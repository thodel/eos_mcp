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

Truncated search (ch-h-bot#594, L-6). With ``<PREFIX>_SEARCH_DIMS`` set (e.g. 512),
the matrix holds only the first d components of each vector, re-normalised — the
Matryoshka prefix the embedding model is trained to make meaningful — and a search
re-scores its best ``RERANK`` candidates with the full vectors read from the
database. Measured on bar_mcp stage 1 (411 163 vectors, 04.11.2026): 512 d alone
loses a third of the top 10 (overlap .64 with 2 560 d), with re-scoring of the
best 200 the ranking is the full one (MRR .894 = .894, overlap .98) at a fifth of
the memory. Unset, nothing changes.
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

_CACHE: dict[tuple, Any] = {}

#: Candidates re-scored with the full vectors in a truncated search.
RERANK = 200


def default_model() -> str:
    prefix = os.environ.get("MCP_ENV_PREFIX", "MCP")
    return os.environ.get(f"{prefix}_EMBED_MODEL",
                          os.environ.get("MCP_EMBED_MODEL", "qwen3-embedding-4b"))


def search_dims() -> Optional[int]:
    """The truncated width from ``<PREFIX>_SEARCH_DIMS``, or None for the full vectors."""
    prefix = os.environ.get("MCP_ENV_PREFIX", "MCP")
    raw = os.environ.get(f"{prefix}_SEARCH_DIMS", os.environ.get("MCP_SEARCH_DIMS", "")).strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def load_matrix(conn_factory: Callable, db_path: str, model: str, dims: Optional[int] = None):
    """(chunk_ids, matrix) for a model, loaded once per (database, model, width).

    With ``dims`` below the stored width, each row keeps its first ``dims``
    components, re-normalised; the full vectors stay on disk."""
    key = (db_path, model, dims)
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
        stored = c.execute("SELECT dims FROM embeddings WHERE model = ? LIMIT 1",
                           (model,)).fetchone()[0]
        width = dims if dims and dims < stored else stored
        matrix = np.empty((n, width), dtype="<f4")
        chunk_ids = []
        for i, (chunk_id, vector) in enumerate(c.execute(
                "SELECT chunk_id, vector FROM embeddings WHERE model = ? ORDER BY rowid",
                (model,))):
            if i >= n:          # zwischen COUNT und Lauf ist etwas dazugekommen
                break
            if width == stored:
                matrix[i] = np.frombuffer(vector, dtype="<f4")
            else:               # das Präfix, neu normiert — nur `width` Werte bleiben im RAM
                head = np.frombuffer(vector, dtype="<f4", count=width)
                matrix[i] = head / (float(np.linalg.norm(head)) or 1.0)
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
        chunk_ids, matrix = load_matrix(conn_factory, db_path, model, search_dims())
    except RuntimeError as exc:
        return {"ready": False, "model": model, "reason": str(exc)}
    out = {"ready": True, "model": model, "n_chunks": len(chunk_ids),
           "dims": int(matrix.shape[1]),
           "megabytes": round(matrix.nbytes / 1_048_576, 1)}
    if search_dims():
        out["rerank"] = RERANK
    return out


def _full_vectors(conn_factory: Callable, model: str, chunk_ids: list) -> dict:
    """The stored vectors of a few chunks, for re-scoring a truncated search.

    The model is checked here, not in SQL: with ``model = ?`` beside a long IN list the
    planner takes ``ix_embeddings_model`` and walks every row of the model — measured on
    bar.db (411 163 vectors, 04.11.2026), 524 ms for 200 vectors instead of 2 ms."""
    import numpy as np
    out = {}
    with conn_factory() as c:
        for start in range(0, len(chunk_ids), 400):
            window = chunk_ids[start:start + 400]
            for cid, m, vector in c.execute(
                    f"SELECT chunk_id, model, vector FROM embeddings WHERE chunk_id IN "
                    f"({','.join('?' * len(window))})", window):
                if m == model:
                    out[cid] = np.frombuffer(vector, dtype="<f4")
    return out


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
    dims = search_dims()
    chunk_ids, matrix = load_matrix(conn_factory, db_path, model, dims)

    query = np.asarray(query_vector, dtype="float32")
    truncated = matrix.shape[1] < query.shape[0] and dims == matrix.shape[1]
    if query.shape[0] != matrix.shape[1] and not truncated:
        raise ValueError(
            f"query has {query.shape[0]} dimensions, index has {matrix.shape[1]}")
    full = query / (float(np.linalg.norm(query)) or 1.0)
    head = query[:matrix.shape[1]]
    scores = matrix @ (head / (float(np.linalg.norm(head)) or 1.0))

    # Take a generous slice before filtering: row_to_hit and the per-parent cap
    # both discard candidates, so topping up here avoids a second matrix pass.
    fetch = min(len(chunk_ids), max(limit * fetch_multiplier, limit + 50))
    if truncated:
        fetch = min(len(chunk_ids), max(fetch, RERANK))
    candidates = np.argpartition(-scores, fetch - 1)[:fetch]
    candidates = candidates[np.argsort(-scores[candidates])]
    picked = [(chunk_ids[i], float(scores[i])) for i in candidates]
    if truncated:
        # The prefix found the candidates; the full vectors order them. A row whose
        # full vector is gone keeps its prefix score rather than vanishing.
        vectors = _full_vectors(conn_factory, model, [cid for cid, _ in picked[:RERANK]])
        rescored = [(cid, float(vectors[cid] @ full) if cid in vectors else s) for cid, s in picked[:RERANK]]
        picked = sorted(rescored, key=lambda p: -p[1]) + picked[RERANK:]

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
