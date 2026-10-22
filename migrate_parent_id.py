#!/usr/bin/env python3
"""
migrate_parent_id.py — hgb.db von der älteren Bauart auf das geteilte Schema (#6).

Das geteilte embed_db.py/vectorstore.py nennt den Elternteil einer Passage
`chunks.parent_id`; dieser Index nannte ihn `doc_id` (Index `ix_chunks_entry`),
und `embedding_runs` zählte `n_articles` statt `n_parents`. Umbenannt wird in
EINER Transaktion; die Vektoren bleiben (ch-h-bot#555: alle 76 737 Passagen
zeilengleich, kein Re-Embed):

    ALTER TABLE chunks RENAME COLUMN doc_id TO parent_id;
    ALTER TABLE embedding_runs RENAME COLUMN n_articles TO n_parents;
    DROP INDEX ix_chunks_entry;
    CREATE INDEX ix_chunks_parent ON chunks(parent_id);

danach PRAGMA integrity_check. DROP INDEX ist nötig: RENAME COLUMN biegt den
alten Index um, sein Name bleibt, und vectorstore.SCHEMA_SQL legte dann
ix_chunks_parent als zweiten gleichen Index an. Idempotent: was schon
umbenannt ist, wird übersprungen, ein zweiter Lauf ändert nichts.

    python migrate_parent_id.py --db /home/tobias/mcp/data/hgb.db

Die Sicherung macht der Betreiber vorher selbst (hgb.db hat 638 MB):

    cp hgb.db hgb.db.pre-parent-id

Der Server muss stehen, und Code und Umbenennung kommen zusammen: das alte
db.py fragt doc_id ab und antwortete auf einer umbenannten DB jedem
search_semantic mit «no such column»; das neue fragt parent_id.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

OLD_COLUMN, NEW_COLUMN = "doc_id", "parent_id"
OLD_RUNS_COLUMN, NEW_RUNS_COLUMN = "n_articles", "n_parents"
OLD_INDEX, NEW_INDEX = "ix_chunks_entry", "ix_chunks_parent"


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def indexes(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA index_list({table})")]


def has_table(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (table,)).fetchone() is not None


def counts(conn: sqlite3.Connection) -> dict:
    """Was zählt, vor und nach der Umbenennung dasselbe sein muss."""
    parent = NEW_COLUMN if NEW_COLUMN in columns(conn, "chunks") else OLD_COLUMN
    return {
        "chunks": conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
        "parents": conn.execute(
            f"SELECT COUNT(DISTINCT {parent}) FROM chunks").fetchone()[0],
        "embeddings": conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0],
        "runs": conn.execute("SELECT COUNT(*) FROM embedding_runs").fetchone()[0],
    }


def plan(conn: sqlite3.Connection) -> list[str]:
    """Die Anweisungen, die diese Datenbank noch braucht — leer, wenn keine."""
    steps = []
    if OLD_COLUMN in columns(conn, "chunks"):
        steps.append(f"ALTER TABLE chunks RENAME COLUMN {OLD_COLUMN} TO {NEW_COLUMN}")
    if OLD_RUNS_COLUMN in columns(conn, "embedding_runs"):
        steps.append(f"ALTER TABLE embedding_runs RENAME COLUMN "
                     f"{OLD_RUNS_COLUMN} TO {NEW_RUNS_COLUMN}")
    if OLD_INDEX in indexes(conn, "chunks"):
        steps.append(f"DROP INDEX {OLD_INDEX}")
    if NEW_INDEX not in indexes(conn, "chunks"):
        steps.append(f"CREATE INDEX {NEW_INDEX} ON chunks({NEW_COLUMN})")
    return steps


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Run the plan in one transaction. Returns the statements that ran."""
    steps = plan(conn)
    if not steps:
        return []
    conn.isolation_level = None          # explicit transaction, nothing implicit
    conn.execute("BEGIN")
    try:
        for statement in steps:
            conn.execute(statement)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return steps


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.environ.get("EOS_DB", "/data/hgb.db"))
    args = parser.parse_args(argv)

    if not os.path.exists(args.db):
        print(f"ERROR: database not found at {args.db}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(args.db)
    if not (has_table(conn, "chunks") and has_table(conn, "embedding_runs")):
        print(f"{args.db}: no semantic index (no chunks/embedding_runs table), "
              "nothing to rename; embed_db.py creates the shared schema itself.")
        return 0

    before = counts(conn)
    print(f"before: {before}")
    steps = migrate(conn)
    if steps:
        for statement in steps:
            print(f"  {statement};")
    else:
        print("  already migrated, nothing to do")

    check = conn.execute("PRAGMA integrity_check").fetchone()[0]
    after = counts(conn)
    print(f"after:  {after}")
    print(f"integrity_check: {check}")
    conn.close()
    if check != "ok":
        print("ERROR: integrity_check failed — restore the backup", file=sys.stderr)
        return 2
    if after != before:
        print("ERROR: the counts changed — restore the backup", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
