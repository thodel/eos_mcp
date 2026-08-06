"""
db.py — SQLite query helpers for the HGB MCP server.
"""

import re
import sqlite3
from contextlib import contextmanager
from typing import Any

_DB_PATH: str = "hgb.db"

MAX_LIMIT = 500            # ceiling for any caller-supplied limit
SPAN_LIMIT = 2000          # spans/events attached to a single document
DOSSIER_INDEX_LIMIT = 1000 # rows in the hgb://dossiers resource
COOC_DOC_CAP = 400         # documents scanned by get_cooccurrences, see below


# The schema, owned here rather than in build_db.py so that importing it costs
# nothing (build_db pulls in lxml) and the tests cannot drift from the build.
SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    dossier_id  TEXT,
    year        INTEGER,
    source      TEXT,
    location    TEXT,         -- WKT POINT(E N) or NULL
    language    TEXT,
    pages       INTEGER,
    text_raw    TEXT,         -- full document text from metadata/@text
    checked     INTEGER       -- 0/1
);

CREATE TABLE IF NOT EXISTS spans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT REFERENCES documents(id),
    span_id     TEXT,
    parent_id   TEXT,         -- for nested spans
    class       TEXT,         -- per, loc, org, date, money, …
    element     TEXT,         -- reference, head, value, trigger
    text        TEXT,
    confidence  REAL,
    token_start INTEGER,
    token_end   INTEGER,
    numerus     TEXT,
    specificity TEXT,
    subclass    TEXT,
    norm        TEXT          -- normalised value (money, date)
);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT REFERENCES documents(id),
    event_id    TEXT,
    class       TEXT,
    token_start INTEGER,
    token_end   INTEGER,
    tense       TEXT,
    polarity    TEXT,
    modality    TEXT
);

-- Full-text search tables
CREATE VIRTUAL TABLE IF NOT EXISTS fts_documents USING fts5(
    id UNINDEXED, text_raw, content=documents, content_rowid=rowid
);
CREATE VIRTUAL TABLE IF NOT EXISTS fts_spans USING fts5(
    doc_id UNINDEXED, span_id UNINDEXED, class UNINDEXED,
    text, content=spans, content_rowid=rowid
);
"""

TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
    INSERT INTO fts_documents(rowid, id, text_raw) VALUES (new.rowid, new.id, new.text_raw);
END;
CREATE TRIGGER IF NOT EXISTS spans_ai AFTER INSERT ON spans BEGIN
    INSERT INTO fts_spans(rowid, doc_id, span_id, class, text)
    VALUES (new.rowid, new.doc_id, new.span_id, new.class, new.text);
END;
"""

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


def clamp(limit, default, cap=MAX_LIMIT) -> int:
    """Constrain a caller-supplied limit. SQLite reads LIMIT -1 as unbounded, so an
    unchecked negative value would return the whole table — and min(limit, 200),
    which this module used before, passes every negative straight through."""
    try:
        n = int(limit)
    except (TypeError, ValueError):
        return default
    return min(n, cap) if n >= 1 else default


def like_pattern(query) -> str:
    """Substring pattern for LIKE, with the wildcards escaped so a query of '%' or
    '_' matches those characters literally instead of the whole table. Pairs with
    ESCAPE '\\' in the SQL."""
    escaped = (query or "").replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def quote_fts(query: str) -> str:
    """Rewrite a query as quoted FTS5 phrases, one per word (implicit AND).
    Strips the characters FTS5 treats as syntax so no input can be a syntax error."""
    tokens = [t for t in re.split(r'\s+', re.sub(r'["\*\(\):^-]', ' ', query or "")) if t]
    return ' '.join(f'"{t}"' for t in tokens)


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
                "FROM spans WHERE doc_id = ? ORDER BY token_start LIMIT ?",
                (doc_id, SPAN_LIMIT),
            )
        )
        result["events"] = rows_to_list(
            c.execute(
                "SELECT event_id, class, token_start, token_end, tense, polarity, modality "
                "FROM events WHERE doc_id = ? LIMIT ?",
                (doc_id, SPAN_LIMIT),
            )
        )
        return result


def get_dossier(dossier_id: str, limit: int = 100) -> list[dict]:
    """Documents in one dossier. Bounded: a dossier carries every document's full
    text_raw, so an unbounded result runs past what a client will accept."""
    with conn() as c:
        docs = rows_to_list(
            c.execute(
                "SELECT id, year, source, location, text_raw "
                "FROM documents WHERE dossier_id = ? ORDER BY year LIMIT ?",
                (dossier_id, clamp(limit, 100)),
            )
        )
    return docs


# ── Person queries ────────────────────────────────────────────────────────────

_PERSON_FTS_SQL = """
    SELECT s.doc_id, s.span_id, s.class, s.text, s.confidence,
           s.numerus, s.specificity,
           d.dossier_id, d.year, d.source, d.location
    FROM fts_spans f
    JOIN spans s  ON s.rowid = f.rowid
    JOIN documents d ON d.id = s.doc_id
    WHERE fts_spans MATCH ? AND s.class = 'per'
    ORDER BY rank
    LIMIT ?
"""

_PERSON_LIKE_SQL = """
    SELECT s.doc_id, s.span_id, s.class, s.text, s.confidence,
           s.numerus, s.specificity,
           d.dossier_id, d.year, d.source, d.location
    FROM spans s
    JOIN documents d ON d.id = s.doc_id
    WHERE s.class = 'per' AND s.text LIKE ? ESCAPE '\\'
    LIMIT ?
"""


def search_persons(query: str, limit: int = 20) -> list[dict]:
    """Search person span texts. Honours FTS5 operators when the query is well
    formed, falls back to quoted phrases and then to a literal substring search,
    rather than raising at the caller."""
    limit = clamp(limit, 20)
    if not query or not query.strip():
        return [{"error": "Empty query."}]
    with conn() as c:
        for q in (query, quote_fts(query)):
            if not q:
                continue
            try:
                return rows_to_list(c.execute(_PERSON_FTS_SQL, (q, limit)).fetchall())
            except sqlite3.OperationalError:
                continue
        return rows_to_list(
            c.execute(_PERSON_LIKE_SQL, (like_pattern(query), limit)).fetchall())


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
            (year_from, year_to, clamp(limit, 100)),
        ).fetchall()
    return rows_to_list(rows)


def get_cooccurrences(person_name: str, limit: int = 20) -> list[dict]:
    """Other persons mentioned in the same documents as person_name.

    The matching documents are capped at COOC_DOC_CAP because their ids are bound
    one-per-placeholder in the second query: a common name matching thousands of
    documents would otherwise exceed SQLite's variable limit and raise
    'too many SQL variables' instead of returning results.
    """
    pattern = like_pattern(person_name)
    with conn() as c:
        doc_ids = [
            r[0]
            for r in c.execute(
                """
                SELECT DISTINCT doc_id FROM spans
                WHERE class = 'per' AND text LIKE ? ESCAPE '\\'
                LIMIT ?
                """,
                (pattern, COOC_DOC_CAP),
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
              AND s.text NOT LIKE ? ESCAPE '\\'
            GROUP BY s.text
            ORDER BY freq DESC
            LIMIT ?
            """,
            (*doc_ids, pattern, clamp(limit, 20)),
        ).fetchall()
    return rows_to_list(rows)


# ── Full-text search ──────────────────────────────────────────────────────────

_TEXT_FTS_SQL = """
    SELECT d.id, d.dossier_id, d.year, d.source, d.location,
           snippet(fts_documents, 1, '<b>', '</b>', '…', 20) AS snippet
    FROM fts_documents f
    JOIN documents d ON d.id = f.id
    WHERE fts_documents MATCH ?
    ORDER BY rank
    LIMIT ?
"""

_TEXT_LIKE_SQL = """
    SELECT id, dossier_id, year, source, location,
           SUBSTR(text_raw, 1, 200) AS snippet
    FROM documents
    WHERE text_raw LIKE ? ESCAPE '\\'
    LIMIT ?
"""


def search_text(query: str, limit: int = 20) -> list[dict]:
    """Full-text search over the transcriptions, with the same fallback ladder as
    search_persons so a stray quote returns results instead of an error."""
    limit = clamp(limit, 20)
    if not query or not query.strip():
        return [{"error": "Empty query."}]
    with conn() as c:
        for q in (query, quote_fts(query)):
            if not q:
                continue
            try:
                return rows_to_list(c.execute(_TEXT_FTS_SQL, (q, limit)).fetchall())
            except sqlite3.OperationalError:
                continue
        return rows_to_list(
            c.execute(_TEXT_LIKE_SQL, (like_pattern(query), limit)).fetchall())


# ── Dossier listing ───────────────────────────────────────────────────────────

_DOSSIER_SQL = """
    SELECT dossier_id,
           MIN(year) AS year_min, MAX(year) AS year_max,
           COUNT(*) AS n_docs,
           location
    FROM documents
    WHERE dossier_id IS NOT NULL
    GROUP BY dossier_id
    ORDER BY dossier_id
    LIMIT ?
"""


def list_dossiers(limit: int = 200) -> list[dict]:
    with conn() as c:
        return rows_to_list(c.execute(_DOSSIER_SQL, (clamp(limit, 200),)).fetchall())


def dossier_index(limit: int = DOSSIER_INDEX_LIMIT) -> dict:
    """Brief dossier index for the hgb://dossiers resource. Says so when it is
    truncated rather than silently returning a prefix."""
    limit = clamp(limit, DOSSIER_INDEX_LIMIT, cap=DOSSIER_INDEX_LIMIT)
    with conn() as c:
        total = c.execute(
            "SELECT COUNT(DISTINCT dossier_id) FROM documents WHERE dossier_id IS NOT NULL"
        ).fetchone()[0]
        rows = rows_to_list(c.execute(_DOSSIER_SQL, (limit,)).fetchall())
    out = {"total": total, "returned": len(rows), "truncated": len(rows) < total,
           "dossiers": rows}
    if out["truncated"]:
        out["note"] = (f"Showing the first {len(rows)} of {total} dossiers. "
                       "Use list_dossiers(limit) for a different slice.")
    return out


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
