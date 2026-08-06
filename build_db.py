"""
build_db.py — parse hgb_full_*.xml into hgb.db (SQLite + FTS5)

Run once before starting the server:
    python build_db.py --xml ../hgb_full_26_05_29_05.xml --db hgb.db

The XML is ~800 MB; streaming iterparse keeps memory usage low (~200 MB peak).
"""

import argparse
import sqlite3
import sys
import time
from lxml import etree


# ── Schema ────────────────────────────────────────────────────────────────────

DDL = """
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

TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS docs_ai AFTER INSERT ON documents BEGIN
    INSERT INTO fts_documents(rowid, id, text_raw) VALUES (new.rowid, new.id, new.text_raw);
END;
CREATE TRIGGER IF NOT EXISTS spans_ai AFTER INSERT ON spans BEGIN
    INSERT INTO fts_spans(rowid, doc_id, span_id, class, text)
    VALUES (new.rowid, new.doc_id, new.span_id, new.class, new.text);
END;
"""


def init_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.executescript(DDL)
    con.executescript(TRIGGERS)
    con.commit()
    return con


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse(xml_path: str, db_path: str, batch: int = 500):
    con = init_db(db_path)
    cur = con.cursor()

    doc_rows, span_rows, event_rows = [], [], []
    n_docs = 0
    t0 = time.time()

    context = etree.iterparse(xml_path, events=("end",), tag="document")

    for _, doc_el in context:
        doc_id = doc_el.get("id")
        if not doc_id:
            doc_el.clear()
            continue

        # metadata element
        meta = doc_el.find("metadata")
        if meta is None:
            doc_el.clear()
            continue

        doc_rows.append((
            doc_id,
            meta.get("dossierid"),
            _int(meta.get("year")),
            meta.get("source"),
            meta.get("location"),
            meta.get("language"),
            _int(meta.get("pages")),
            meta.get("text", ""),
            1 if meta.get("checked_by_gpt") == "true" else 0,
        ))

        # spans
        for sp in doc_el.findall(".//spans/span"):
            parent = sp.getparent()
            parent_id = parent.get("id") if parent.tag == "span" else None
            span_rows.append((
                doc_id,
                sp.get("id"),
                parent_id,
                sp.get("class"),
                sp.get("element"),
                sp.get("text", ""),
                _float(sp.get("confidence")),
                _int(sp.get("start")),
                _int(sp.get("end")),
                sp.get("numerus"),
                sp.get("specificity"),
                sp.get("subclass"),
                sp.get("norm"),
            ))

        # events
        for ev in doc_el.findall(".//eventGroups/eventGroup"):
            event_rows.append((
                doc_id,
                ev.get("event_id"),
                ev.get("class"),
                _int(ev.get("start")),
                _int(ev.get("end")),
                ev.get("tense"),
                ev.get("polarity"),
                ev.get("modality"),
            ))

        doc_el.clear()
        n_docs += 1

        if n_docs % batch == 0:
            _flush(cur, doc_rows, span_rows, event_rows)
            con.commit()
            doc_rows, span_rows, event_rows = [], [], []
            elapsed = time.time() - t0
            print(f"  {n_docs:,} documents … {elapsed:.0f}s", end="\r", flush=True)

    _flush(cur, doc_rows, span_rows, event_rows)
    con.commit()

    elapsed = time.time() - t0
    print(f"\nDone: {n_docs:,} documents in {elapsed:.1f}s → {db_path}")
    con.close()


def _flush(cur, docs, spans, events):
    cur.executemany(
        "INSERT OR IGNORE INTO documents VALUES (?,?,?,?,?,?,?,?,?)", docs
    )
    cur.executemany(
        "INSERT INTO spans(doc_id,span_id,parent_id,class,element,text,"
        "confidence,token_start,token_end,numerus,specificity,subclass,norm) VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?)", spans
    )
    cur.executemany(
        "INSERT INTO events(doc_id,event_id,class,token_start,token_end,"
        "tense,polarity,modality) VALUES (?,?,?,?,?,?,?,?)", events
    )


def _int(v):
    try: return int(v)
    except: return None

def _float(v):
    try: return float(v)
    except: return None


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="../hgb_full_26_05_29_05.xml")
    ap.add_argument("--db",  default="hgb.db")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()
    print(f"Parsing {args.xml} → {args.db}")
    parse(args.xml, args.db, args.batch)
