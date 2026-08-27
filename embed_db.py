#!/usr/bin/env python3
"""
embed_db.py — build the semantic index into kf.db.

Resolves each entry's transcription to its expanded reading, windows it into
passages, embeds them on GPUStack, and stores the vectors alongside the corpus.
Run it after build_db.py:

    GPUSTACK_API_KEY=... python embed_db.py                    # full corpus
    GPUSTACK_API_KEY=... python embed_db.py --limit 100        # a sample first
    GPUSTACK_API_KEY=... python embed_db.py --recompute        # after a model change

The expansion vocabulary is built from the whole corpus before chunking, not
per entry — deciding whether "Hein" is a fragment or a name needs evidence from
everywhere the token occurs (see embeddings.build_vocab).

The run is **resumable**: chunks already embedded with the same model are
skipped, so an interrupted run continues where it stopped rather than starting
over. Chunking is deterministic, so re-running produces the same chunk ids.

Each run is recorded in `embedding_runs` with the model, dimensions and window
settings that produced the vectors — an answer's evidence should be traceable
to the process that indexed it, not just to the article.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone

import db as db_module
import embeddings as emb


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the chunk/embedding tables in a database built before they existed."""
    conn.executescript(db_module.EMBEDDING_SCHEMA_SQL)
    conn.commit()


def build_chunks(conn: sqlite3.Connection, limit: int | None,
                 size: int, overlap: int, quiet: bool) -> int:
    """Window every entry that has no chunks yet. Returns the number written."""
    # Built over the entire corpus, including entries already chunked: the
    # fragment/word judgement needs every occurrence of a token, not just the
    # ones in this batch.
    if not quiet:
        print("  building expansion vocabulary …", file=sys.stderr)
    vocab = emb.build_vocab(
        t for (t,) in conn.execute(
            "SELECT text_raw FROM documents WHERE text_raw IS NOT NULL"))

    rows = conn.execute(
        """
        SELECT e.id, e.dossier_id, e.year, e.text_raw
        FROM documents e
        WHERE e.text_raw IS NOT NULL AND e.text_raw <> ''
          AND NOT EXISTS (SELECT 1 FROM chunks c WHERE c.doc_id = e.id)
        ORDER BY e.id
        """ + (" LIMIT ?" if limit else ""),
        (limit,) if limit else ()).fetchall()

    written = 0
    batch: list[tuple] = []
    for doc_id, dossier_id, year, text in rows:
        cleaned = emb.clean_entry_text(text, vocab)
        for index, (start, end, piece) in enumerate(
                emb.chunk_article(cleaned, size, overlap)):
            # Title and year ride on the first line of every passage. A charter
            # passage otherwise carries no trace of which document or century it
            # belongs to, and "the farm at Lind" reads the same in 1360 as 1644.
            # No title in this corpus; the dossier and year are what a
            # reader has to place an entry by.
            head = " ".join(str(x) for x in (dossier_id, year) if x)
            body = f"{head}\n{piece}" if head else piece
            batch.append((f"{doc_id}#{index}", doc_id, index, start, end, body))
        if len(batch) >= 5000:
            conn.executemany(
                "INSERT OR IGNORE INTO chunks VALUES (?,?,?,?,?,?)", batch)
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


def embed_pending(conn: sqlite3.Connection, model: str, batch_size: int,
                  quiet: bool) -> tuple[int, int]:
    """Embed every chunk lacking a vector for ``model``. Returns (n_chunks, dims)."""
    pending = conn.execute(
        """
        SELECT c.chunk_id, c.text
        FROM chunks c
        LEFT JOIN embeddings e ON e.chunk_id = c.chunk_id AND e.model = ?
        WHERE e.chunk_id IS NULL
        ORDER BY c.chunk_id
        """, (model,)).fetchall()

    total = len(pending)
    if not total:
        return 0, 0
    if not quiet:
        print(f"  {total:,} chunks to embed with {model}", file=sys.stderr)

    client = emb.get_client()
    done, dims, t0 = 0, 0, time.time()
    for offset in range(0, total, batch_size):
        window = pending[offset:offset + batch_size]
        vectors = emb.embed_texts([text for _, text in window],
                                  model=model, client=client)
        if len(vectors) != len(window):
            raise emb.EmbeddingError(
                f"asked for {len(window)} vectors, got {len(vectors)}")
        dims = len(vectors[0])
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (chunk_id, model, dims, vector) "
            "VALUES (?,?,?,?)",
            [(chunk_id, model, dims, emb.pack(vector))
             for (chunk_id, _), vector in zip(window, vectors)])
        conn.commit()          # commit per batch — an interrupted run keeps its work
        done += len(window)
        if not quiet and (offset // batch_size) % 20 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            eta = (total - done) / rate if rate else 0
            print(f"  embedded {done:,}/{total:,}  "
                  f"({rate:.0f} chunks/s, eta {eta / 60:.1f} min)", file=sys.stderr)
    return done, dims


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.environ.get("KF_DB", "/data/kf.db"))
    parser.add_argument("--model", default=emb.DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=emb.DEFAULT_BATCH)
    parser.add_argument("--chunk-chars", type=int, default=emb.CHUNK_CHARS)
    parser.add_argument("--chunk-overlap", type=int, default=emb.CHUNK_OVERLAP)
    parser.add_argument("--limit", type=int, default=None,
                        help="only chunk this many not-yet-chunked documents "
                             "(for a trial run on a sample)")
    parser.add_argument("--recompute", action="store_true",
                        help="drop existing chunks and vectors and rebuild")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"ERROR: database not found at {args.db}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    ensure_schema(conn)

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
        n_new_chunks = build_chunks(conn, args.limit, args.chunk_chars,
                                    args.chunk_overlap, args.quiet)
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

    n_articles = conn.execute(
        "SELECT COUNT(DISTINCT doc_id) FROM chunks").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    conn.execute(
        "UPDATE embedding_runs SET finished_at=?, dims=?, n_articles=?, n_chunks=? "
        "WHERE run_id=?", (utcnow(), dims or None, n_articles, n_chunks, run_id))
    conn.commit()

    print(f"Done in {time.time() - t0:.0f}s: +{n_new_chunks:,} chunks, "
          f"+{n_embedded:,} vectors ({dims or '?'}d, {args.model}); "
          f"corpus now {n_chunks:,} chunks over {n_articles:,} entries.")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
