#!/usr/bin/env python3
"""
vectorstore.py — the vector schema and search every MCP server here shares.

VENDORED FILE, byte-identical across hls_mcp, kf_mcp, dodis_mcp and
infoclio_mcp. Corpus-specific behaviour is injected, never edited in.

Vectors are stored L2-normalised, so cosine similarity is a dot product and a
search is one matrix multiply. At these corpus sizes that is exact and fast
enough that an approximate index would add a recall parameter to get wrong in
exchange for nothing.

What a repo supplies:
  * ``set_db_path`` / a ``conn()`` context manager (its own db module)
  * ``JOIN_SQL`` — how to fetch its own metadata for a set of chunk ids
  * ``row_to_hit`` — how to turn one joined row into a result dict

Everything else — the schema, the packing, the matrix, the per-parent cap — is
the same everywhere, which is the point.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Optional

# The chunk/embedding tables. Kept separate from each repo's corpus schema so
# rebuilding a corpus cannot silently drop its vectors, and adding vectors
# cannot touch the corpus.
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT    PRIMARY KEY,   -- "<parent_id>#<chunk_index>"
    parent_id   TEXT    NOT NULL,      -- document / article / entry id
    chunk_index INTEGER NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    UNIQUE (parent_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT    PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model    TEXT    NOT NULL,
    dims     INTEGER NOT NULL,
    vector   BLOB    NOT NULL          -- float32, little-endian, L2-normalised
);
-- One row per pipeline run. Provenance is a project requirement, not
-- bookkeeping: an answer's evidence has to be traceable to the process that
-- indexed it, not only to the document.
CREATE TABLE IF NOT EXISTS embedding_runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    model       TEXT NOT NULL,
    dims        INTEGER,
    base_url    TEXT,
    chunk_chars INTEGER,
    chunk_overlap INTEGER,
    n_parents   INTEGER,
    n_chunks    INTEGER,
    notes       TEXT
);
CREATE INDEX IF NOT EXISTS ix_chunks_parent ON chunks(parent_id);
CREATE INDEX IF NOT EXISTS ix_embeddings_model ON embeddings(model);
"""

_CACHE: dict[tuple[str, str], Any] = {}


def default_model() -> str:
    prefix = os.environ.get("MCP_ENV_PREFIX", "MCP")
    return os.environ.get(f"{prefix}_EMBED_MODEL",
                          os.environ.get("MCP_EMBED_MODEL", "qwen3-embedding-4b"))


def load_matrix(conn_factory: Callable, db_path: str, model: str):
    """(chunk_ids, matrix) for a model, loaded once per (database, model)."""
    key = (db_path, model)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for semantic search") from exc

    # Zeilenweise in eine vorab angelegte Matrix, statt erst alle Zeilen zu holen und
    # dann zu einem Puffer zu verketten (07.10.2026).
    #
    # Der Grund ist gemessen, nicht geahnt: `fetchall()` hielt 1,98 Mio. Blobs (8,1 GB),
    # `b"".join` legte dieselben 8,1 GB noch einmal daneben — Spitze also 16,2 GB. Die
    # Unit erlaubte 16 G, und genau daran ist der Reindex sieben Nächte in Folge
    # gescheitert: der Dienst verbrauchte in zwölf Minuten 18,8 s CPU, er rechnete nicht,
    # er steckte in der Reclaim-Drossel. Die Grenzen sind inzwischen hochgesetzt; diese
    # Fassung braucht sie nicht mehr auszuschöpfen.
    #
    # `ORDER BY rowid` statt `ORDER BY chunk_id`: die Reihenfolge liest niemand (`search`
    # nimmt argpartition über die Punktzahlen und verlangt nur, dass `chunk_ids[i]` zu
    # `matrix[i]` passt), aber rowid ist die Anordnung der Tabelle selbst — ein
    # fortlaufender Lauf über die Platte, ohne den temporären B-Baum, den der Abfrageplan
    # für chunk_id aufbauen musste (24,8 der 28,4 Sekunden).
    with conn_factory() as c:
        n = c.execute("SELECT COUNT(*) FROM embeddings WHERE model = ?",
                      (model,)).fetchone()[0]
        if not n:
            raise RuntimeError(
                f"no embeddings for model {model!r} in {db_path}. "
                "Run embed_db.py to build the semantic index.")
        dims = c.execute("SELECT dims FROM embeddings WHERE model = ? LIMIT 1",
                         (model,)).fetchone()[0]
        matrix = np.empty((n, dims), dtype="<f4")
        chunk_ids = []
        for i, (chunk_id, vector) in enumerate(c.execute(
                "SELECT chunk_id, vector FROM embeddings WHERE model = ? ORDER BY rowid",
                (model,))):
            if i >= n:          # zwischen COUNT und Lauf ist etwas dazugekommen
                break
            matrix[i] = np.frombuffer(vector, dtype="<f4")
            chunk_ids.append(chunk_id)
        if len(chunk_ids) < n:  # oder weggefallen: die Matrix auf das Gelesene kürzen
            matrix = matrix[:len(chunk_ids)]
    _CACHE[key] = (chunk_ids, matrix)
    return chunk_ids, matrix


def clear_cache() -> None:
    _CACHE.clear()


def warm(conn_factory: Callable, db_path: str, model: str) -> dict:
    """Load the vectors now and report what was loaded, or why it could not be.

    Called at startup so the first user question is not the request that waits
    for the matrix to be assembled.
    """
    try:
        chunk_ids, matrix = load_matrix(conn_factory, db_path, model)
    except RuntimeError as exc:
        return {"ready": False, "model": model, "reason": str(exc)}
    return {"ready": True, "model": model, "n_chunks": len(chunk_ids),
            "dims": int(matrix.shape[1]),
            "megabytes": round(matrix.nbytes / 1_048_576, 1)}


def search(conn_factory: Callable, db_path: str, query_vector, *,
           join_sql: str, row_to_hit: Callable[[Any, str, float], Optional[dict]],
           limit: int = 20, model: Optional[str] = None,
           per_parent: int = 2, fetch_multiplier: int = 8) -> list[dict]:
    """Passages closest in meaning to an already-embedded query.

    ``per_parent`` caps how many passages one document may contribute, so a
    single long document cannot fill the result set and crowd out the others
    that answer the question.

    ``row_to_hit`` may return None to filter a row out — that is how a caller
    applies its own restrictions (a category, a year range) without this
    function needing to know they exist.
    """
    import numpy as np

    model = model or default_model()
    chunk_ids, matrix = load_matrix(conn_factory, db_path, model)

    query = np.asarray(query_vector, dtype="float32")
    if query.shape[0] != matrix.shape[1]:
        raise ValueError(
            f"query has {query.shape[0]} dimensions, index has {matrix.shape[1]}")
    norm = float(np.linalg.norm(query)) or 1.0
    scores = matrix @ (query / norm)

    # Take a generous slice before filtering: row_to_hit and the per-parent cap
    # both discard candidates, so topping up here avoids a second matrix pass.
    fetch = min(len(chunk_ids), max(limit * fetch_multiplier, limit + 50))
    candidates = np.argpartition(-scores, fetch - 1)[:fetch]
    candidates = candidates[np.argsort(-scores[candidates])]
    picked = [(chunk_ids[i], float(scores[i])) for i in candidates]

    by_id = {}
    with conn_factory() as c:
        for start in range(0, len(picked), 400):
            window = picked[start:start + 400]
            sql = join_sql.format(placeholders=",".join("?" * len(window)))
            for row in c.execute(sql, [cid for cid, _ in window]).fetchall():
                by_id[row[0]] = row

    out, seen = [], {}
    for chunk_id, score in picked:
        row = by_id.get(chunk_id)
        if row is None:
            continue                      # a vector outlived its chunk
        hit = row_to_hit(row, chunk_id, round(score, 4))
        if hit is None:
            continue
        parent = hit.get("id")
        if seen.get(parent, 0) >= per_parent:
            continue
        seen[parent] = seen.get(parent, 0) + 1
        out.append(hit)
        if len(out) >= limit:
            break
    return out


def stats(conn_factory: Callable, parent_table: str, parent_id_col: str = "id") -> dict:
    """Coverage of the semantic index, and the runs that produced it."""
    with conn_factory() as c:
        if not c.execute("SELECT name FROM sqlite_master WHERE type='table' "
                         "AND name='embeddings'").fetchone():
            return {"indexed": False, "reason": "no embeddings table; run embed_db.py"}
        by_model = c.execute(
            "SELECT model, COUNT(*), MAX(dims) FROM embeddings GROUP BY model").fetchall()
        n_chunks = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_total = c.execute(f"SELECT COUNT(*) FROM {parent_table}").fetchone()[0]
        n_indexed = c.execute("SELECT COUNT(DISTINCT parent_id) FROM chunks").fetchone()[0]
        runs = c.execute(
            "SELECT run_id, started_at, finished_at, model, dims, n_chunks, notes "
            "FROM embedding_runs ORDER BY started_at DESC LIMIT 5").fetchall()
    return {
        "indexed": bool(by_model),
        "n_chunks": n_chunks,
        "n_indexed": n_indexed,
        "n_total": n_total,
        "coverage": round(n_indexed / n_total, 4) if n_total else 0.0,
        "models": [{"model": m, "n_vectors": n, "dims": d} for m, n, d in by_model],
        "recent_runs": [
            {"run_id": r[0], "started_at": r[1], "finished_at": r[2], "model": r[3],
             "dims": r[4], "n_chunks": r[5], "notes": r[6]} for r in runs],
    }
