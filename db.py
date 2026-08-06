"""
db.py — SQLite query helpers for the HGB MCP server.
"""

import sqlite3
from contextlib import contextmanager
from typing import Any

_DB_PATH: str = "hgb.db"


def set_db_path(path: str):
    global _DB_PATH
    _DB_PATH = path


@contextmanager
def conn():
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only = ON")
    try:
        yield con
    finally:
        con.close()


def row_to_dict(row) -> dict:
    return dict(row) if row else {}


def rows_to_list(rows) -> list[dict]:
    return [dict(r) for r in rows]


# ── Document queries ──────────────────────────────────────────────────────────

def get_document(doc_id: str) -> dict:
    with conn() as c:
        doc = c.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if not doc:
            return {}
        result = row_to_dict(doc)
        result["spans"] = rows_to_list(
            c.execute(
                "SELECT span_id, parent_id, class, element, text, confidence, "
                "token_start, token_end, numerus, specificity, subclass, norm "
                "FROM spans WHERE doc_id = ? ORDER BY token_start",
                (doc_id,),
            )
        )
        result["events"] = rows_to_list(
            c.execute(
                "SELECT event_id, class, token_start, token_end, tense, polarity, modality "
                "FROM events WHERE doc_id = ?",
                (doc_id,),
            )
        )
        return result


def get_dossier(dossier_id: str) -> list[dict]:
    with conn() as c:
        docs = rows_to_list(
            c.execute(
                "SELECT id, year, source, location, text_raw "
                "FROM documents WHERE dossier_id = ? ORDER BY year",
                (dossier_id,),
            )
        )
    return docs


# ── Person queries ────────────────────────────────────────────────────────────

def search_persons(query: str, limit: int = 20) -> list[dict]:
    """FTS search over person span texts."""
    with conn() as c:
        rows = c.execute(
            """
            SELECT s.doc_id, s.span_id, s.class, s.text, s.confidence,
                   s.numerus, s.specificity,
                   d.dossier_id, d.year, d.source, d.location
            FROM fts_spans f
            JOIN spans s  ON s.rowid = f.rowid
            JOIN documents d ON d.id = s.doc_id
            WHERE fts_spans MATCH ? AND s.class = 'per'
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    return rows_to_list(rows)


def get_persons_in_year_range(
    year_from: int, year_to: int, limit: int = 100
) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            """
            SELECT s.text, s.confidence, s.numerus, s.specificity,
                   d.id AS doc_id, d.dossier_id, d.year, d.source, d.location
            FROM spans s
            JOIN documents d ON d.id = s.doc_id
            WHERE s.class = 'per' AND s.element = 'head'
              AND d.year BETWEEN ? AND ?
            ORDER BY d.year, s.text
            LIMIT ?
            """,
            (year_from, year_to, limit),
        ).fetchall()
    return rows_to_list(rows)


def get_cooccurrences(person_name: str, limit: int = 20) -> list[dict]:
    """Other persons mentioned in the same documents as person_name."""
    with conn() as c:
        # Find doc_ids containing the person
        doc_ids = [
            r[0]
            for r in c.execute(
                """
                SELECT DISTINCT doc_id FROM spans
                WHERE class = 'per' AND text LIKE ?
                """,
                (f"%{person_name}%",),
            )
        ]
        if not doc_ids:
            return []
        placeholders = ",".join("?" * len(doc_ids))
        rows = c.execute(
            f"""
            SELECT s.text, COUNT(*) AS freq,
                   GROUP_CONCAT(DISTINCT d.dossier_id) AS dossiers
            FROM spans s
            JOIN documents d ON d.id = s.doc_id
            WHERE s.doc_id IN ({placeholders})
              AND s.class = 'per'
              AND s.text NOT LIKE ?
            GROUP BY s.text
            ORDER BY freq DESC
            LIMIT ?
            """,
            (*doc_ids, f"%{person_name}%", limit),
        ).fetchall()
    return rows_to_list(rows)


# ── Full-text search ──────────────────────────────────────────────────────────

def search_text(query: str, limit: int = 20) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            """
            SELECT d.id, d.dossier_id, d.year, d.source, d.location,
                   snippet(fts_documents, 1, '<b>', '</b>', '…', 20) AS snippet
            FROM fts_documents f
            JOIN documents d ON d.id = f.id
            WHERE fts_documents MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
    return rows_to_list(rows)


# ── Dossier listing ───────────────────────────────────────────────────────────

def list_dossiers(limit: int = 1000) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            """
            SELECT dossier_id,
                   MIN(year) AS year_min, MAX(year) AS year_max,
                   COUNT(*) AS n_docs,
                   location
            FROM documents
            WHERE dossier_id IS NOT NULL
            GROUP BY dossier_id
            ORDER BY dossier_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return rows_to_list(rows)


def db_stats() -> dict[str, Any]:
    with conn() as c:
        return {
            "n_documents": c.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
            "n_spans":     c.execute("SELECT COUNT(*) FROM spans").fetchone()[0],
            "n_persons":   c.execute("SELECT COUNT(*) FROM spans WHERE class='per'").fetchone()[0],
            "n_events":    c.execute("SELECT COUNT(*) FROM events").fetchone()[0],
            "n_dossiers":  c.execute("SELECT COUNT(DISTINCT dossier_id) FROM documents").fetchone()[0],
            "year_min":    c.execute("SELECT MIN(year) FROM documents").fetchone()[0],
            "year_max":    c.execute("SELECT MAX(year) FROM documents").fetchone()[0],
        }
