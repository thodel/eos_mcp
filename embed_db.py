#!/usr/bin/env python3
"""
embed_db.py — build the semantic index. Shared, vendored, byte-identical.

    GPUSTACK_API_KEY=... python embed_db.py                 # whole corpus
    GPUSTACK_API_KEY=... python embed_db.py --limit 500     # a sample first
    GPUSTACK_API_KEY=... python embed_db.py --recompute     # after a model change
    python embed_db.py --status

Resumable: chunks already embedded with the same model are skipped and every
batch is committed, so an interrupted run continues rather than starting over.
Chunking is deterministic, so re-running produces the same chunk ids.

A repo supplies `corpus.py` with:
    DB_ENV, DEFAULT_DB, PARENT_TABLE   what and where its corpus is
    iter_documents(conn, limit)        -> (parent_id, embedding_text)
                                       or (parent_id, heading, embedding_text)
    clean(text)                        -> optional corpus-specific cleanup
Everything below is the same everywhere.

A heading — title, year, pages, whatever says what a passage is from — goes on
the first line of EVERY passage of its document, not only the first: a window
lifted from page 14 of a twenty-page article otherwise embeds as if it were
anonymous. It is not part of the text that is cleaned and windowed, so offsets
point into clean(text) and a hit can be located in its document, and it does
not count against the window size. Corpora that yield pairs get no heading and
exactly the chunks they got before.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone

import corpus
import embeddings as emb
import vectorstore


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def split_document(item) -> tuple:
    """(parent_id, heading, text) from what iter_documents yielded."""
    if len(item) == 2:
        return item[0], "", item[1]
    if len(item) == 3:
        # Verbatim: the corpus formats its heading, and the stored text has to
        # stay what the vectors were made from. Blank means none.
        heading = item[1] or ""
        return item[0], heading if heading.strip() else "", item[2]
    raise ValueError("iter_documents must yield (parent_id, text) or "
                     f"(parent_id, heading, text), not {len(item)} values")


def build_chunks(conn, limit, size, overlap, quiet) -> int:
    written, batch = 0, []
    clean = getattr(corpus, "clean", None)
    for item in corpus.iter_documents(conn, limit):
        parent_id, heading, text = split_document(item)
        for index, (start, end, piece) in enumerate(
                emb.chunk_document(text, size, overlap, clean=clean)):
            batch.append((f"{parent_id}#{index}", str(parent_id), index,
                          start, end, f"{heading}\n{piece}" if heading else piece))
        if len(batch) >= 5000:
            conn.executemany("INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?)", batch)
            conn.commit()
            written += len(batch)
            batch.clear()
            if not quiet:
                print(f"  chunked {written:,} …", file=sys.stderr)
    if batch:
        conn.executemany("INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?)", batch)
        conn.commit()
        written += len(batch)
    return written


def reuse_vectors(conn, reuse_db: str, model: str, quiet: bool) -> int:
    """Copy vectors from a previous index for chunks whose text is unchanged.

    A nightly rebuild starts from an empty staging database and used to
    re-embed the whole corpus — 647,000 chunks, forty minutes of GPU — for
    the few thousand pages read since yesterday, and a dropped connection
    in minute thirty threw all of it away (2026-09-10). Chunk ids are
    deterministic (parent#index) and the text is compared, so a chunk gets
    its old vector only if it is the same chunk; what changed or is new is
    embedded as before.
    """
    conn.execute("ATTACH DATABASE ? AS prev", (reuse_db,))
    try:
        copied = conn.execute(
            "INSERT OR IGNORE INTO main.embeddings (chunk_id, model, dims, vector) "
            "SELECT e.chunk_id, e.model, e.dims, e.vector "
            "FROM prev.embeddings e "
            "JOIN prev.chunks pc ON pc.chunk_id = e.chunk_id "
            "JOIN main.chunks c ON c.chunk_id = pc.chunk_id AND c.text = pc.text "
            "WHERE e.model = ?", (model,)).rowcount
        conn.commit()
    finally:
        conn.execute("DETACH DATABASE prev")
    if not quiet:
        print(f"  reused {copied:,} vectors from {reuse_db}", file=sys.stderr)
    return copied


def embed_pending(conn, model, batch_size, quiet) -> tuple[int, int]:
    pending = conn.execute(
        "SELECT c.chunk_id, c.text FROM chunks c "
        "LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id AND e.model = ? "
        "WHERE e.chunk_id IS NULL ORDER BY c.chunk_id", (model,)).fetchall()
    total = len(pending)
    if not total:
        return 0, 0
    if not quiet:
        print(f"  {total:,} chunks to embed with {model}", file=sys.stderr)

    client = emb.get_client()
    done, dims, t0 = 0, 0, time.time()
    for offset in range(0, total, batch_size):
        window = pending[offset:offset + batch_size]
        vectors = emb.embed_texts([t for _, t in window], model=model, client=client)
        if len(vectors) != len(window):
            raise emb.EmbeddingError(
                f"asked for {len(window)} vectors, got {len(vectors)}")
        dims = len(vectors[0])
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (chunk_id, model, dims, vector) "
            "VALUES (?,?,?,?)",
            [(cid, model, dims, emb.pack(v)) for (cid, _), v in zip(window, vectors)])
        conn.commit()          # per batch: an interrupted run keeps its work
        done += len(window)
        if not quiet and (offset // batch_size) % 20 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            eta = (total - done) / rate if rate else 0
            print(f"  embedded {done:,}/{total:,} ({rate:.0f}/s, eta {eta/60:.1f} min)",
                  file=sys.stderr)
    return done, dims


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.environ.get(corpus.DB_ENV, corpus.DEFAULT_DB))
    parser.add_argument("--model", default=emb.DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=emb.DEFAULT_BATCH)
    parser.add_argument("--chunk-chars", type=int, default=emb.CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=emb.CHUNK_OVERLAP)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--reuse", metavar="DB",
                        help="copy vectors from this earlier index for chunks "
                             "whose text is unchanged, then embed the rest")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"ERROR: database not found at {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(vectorstore.SCHEMA_SQL)
    conn.commit()

    if args.status:
        import db as db_module
        db_module.set_db_path(args.db)
        stats = vectorstore.stats(db_module.conn, corpus.PARENT_TABLE)
        for key, value in stats.items():
            if key != "recent_runs":
                print(f"  {key}: {value}")
        return 0

    if args.recompute:
        conn.executescript("DELETE FROM embeddings; DELETE FROM chunks;")
        conn.commit()

    run_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO embedding_runs (run_id, started_at, model, base_url, "
        "chunk_chars, chunk_overlap) VALUES (?,?,?,?,?,?)",
        (run_id, utcnow(), args.model, emb.DEFAULT_BASE_URL,
         args.chunk_chars, args.chunk_overlap))
    conn.commit()

    t0 = time.time()
    try:
        n_new = build_chunks(conn, args.limit, args.chunk_chars,
                             args.chunk_overlap, args.quiet)
        if args.reuse and os.path.exists(args.reuse):
            reuse_vectors(conn, args.reuse, args.model, args.quiet)
        n_embedded, dims = embed_pending(conn, args.model, args.batch_size, args.quiet)
    except emb.EmbeddingError as exc:
        conn.execute("UPDATE embedding_runs SET finished_at=?, notes=? WHERE run_id=?",
                     (utcnow(), f"failed: {exc}", run_id))
        conn.commit()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        conn.execute("UPDATE embedding_runs SET finished_at=?, notes=? WHERE run_id=?",
                     (utcnow(), "interrupted — rerun to resume", run_id))
        conn.commit()
        print("\ninterrupted; embedded chunks are committed, rerun to resume",
              file=sys.stderr)
        return 130

    n_parents = conn.execute("SELECT COUNT(DISTINCT parent_id) FROM chunks").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.execute(
        "UPDATE embedding_runs SET finished_at=?, dims=?, n_parents=?, n_chunks=? "
        "WHERE run_id=?", (utcnow(), dims or None, n_parents, n_chunks, run_id))
    conn.commit()
    print(f"Done in {time.time()-t0:.0f}s: +{n_new:,} chunks, +{n_embedded:,} vectors "
          f"({dims or '?'}d, {args.model}); corpus now {n_chunks:,} chunks over "
          f"{n_parents:,} documents.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
