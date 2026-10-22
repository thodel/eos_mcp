#!/usr/bin/env python3
"""
test_eos_mcp.py — test suite for the EOS / HGB Basel MCP server.

Runs two ways. Under pytest, failures fail the run; the CLI keeps the grouped
output and sets the exit code.

    pytest test_eos_mcp.py                                     # unit tests only
    EOS_DB=/data/hgb.db pytest test_eos_mcp.py                 # + DB tests
    EOS_SERVER=http://localhost:8000 pytest test_eos_mcp.py    # + server tests

    python test_eos_mcp.py --unit
    python test_eos_mcp.py --db /data/hgb.db --server http://localhost:8000

Unit tests build their own throwaway database from db.SCHEMA_SQL, so they need
no setup. Tests needing the real DB or a live server skip when it isn't configured.
"""
import argparse, json, os, sqlite3, sys, tempfile
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── Helpers ───────────────────────────────────────────────────────────────────

RED   = "\033[91m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RESET = "\033[0m"

def ok(msg):   print(f"{GREEN}✅ {msg}{RESET}")
def fail(msg): print(f"{RED}❌ {msg}{RESET}")
def warn(msg): print(f"{YELLOW}⚠️  {msg}{RESET}")
def info(msg): print(f"   {msg}")

class Checks:
    """Collects several checks so one run reports them all, then fails as a unit.

    Not named Test* — pytest would try to collect it as a test class.
    """
    def __init__(self): self.passed = self.failed = 0; self.failures = []
    def check(self, cond, msg):
        if cond:
            self.passed += 1
            ok(msg)
        else:
            self.failed += 1
            self.failures.append(msg)
            fail(msg)
    def assert_ok(self):
        total = self.passed + self.failed
        print(f"\n{'─'*50}")
        print(f"Ran {total} checks: {GREEN}{self.passed} passed{RESET}", end="")
        if self.failed: print(f", {RED}{self.failed} failed{RESET}", end="")
        print()
        if self.failed:
            raise AssertionError(
                f"{self.failed} of {total} checks failed:\n  - " + "\n  - ".join(self.failures))


@pytest.fixture
def db_path():
    return os.environ.get("EOS_DB", "")

@pytest.fixture
def base_url():
    return os.environ.get("EOS_SERVER", "")


# ── Fixture database ──────────────────────────────────────────────────────────

# id, dossier_id, year, source, location, language, pages, text_raw, checked
DOCUMENTS = [
    ("doc1", "dos-A", 1450, "StABS A1", "POINT(7.58 47.55)", "gmh", 2,
     "Item Hans Meier hat geben zwei pfund pfennig", 1),
    ("doc2", "dos-A", 1455, "StABS A2", None, "gmh", 1,
     "Anna Sicher und Hans Meier bekennen sich schuldig", 1),
    ("doc3", "dos-B", 1600, "StABS B1", None, "gmh", 3,
     "Ulrich Zwingli wird genannt in diesem Eintrag", 0),
]

# doc_id, span_id, parent_id, class, element, text, confidence,
# token_start, token_end, numerus, specificity, subclass, norm
SPANS = [
    ("doc1", "s1", None, "per", "head", "Hans Meier",  0.9, 2, 3, "sg", "spec", None, None),
    ("doc1", "s2", None, "money", "value", "zwei pfund", 0.8, 5, 6, None, None, None, "2 lb"),
    ("doc2", "s3", None, "per", "head", "Anna 100%",   0.9, 0, 1, "sg", "spec", None, None),
    ("doc2", "s4", None, "per", "head", "Hans_Meier",  0.9, 3, 4, "sg", "spec", None, None),
    ("doc3", "s5", None, "per", "head", "Ulrich Zwingli", 0.7, 0, 1, "sg", "spec", None, None),
]

# doc_id, event_id, class, token_start, token_end, tense, polarity, modality
EVENTS = [
    ("doc1", "e1", "payment", 2, 6, "past", "pos", None),
]


def make_fixture_db(path):
    import db as db_module
    con = sqlite3.connect(path)
    con.executescript(db_module.SCHEMA_SQL)
    con.executescript(db_module.TRIGGERS_SQL)
    con.executemany(f"INSERT INTO documents VALUES ({','.join('?' * 9)})", DOCUMENTS)
    con.executemany(
        "INSERT INTO spans (doc_id, span_id, parent_id, class, element, text, confidence,"
        " token_start, token_end, numerus, specificity, subclass, norm) "
        f"VALUES ({','.join('?' * 13)})", SPANS)
    con.executemany(
        "INSERT INTO events (doc_id, event_id, class, token_start, token_end, tense,"
        f" polarity, modality) VALUES ({','.join('?' * 8)})", EVENTS)
    con.commit(); con.close()
    db_module.set_db_path(path)
    return db_module


# ── 1. Unit tests ─────────────────────────────────────────────────────────────

def test_limits_are_clamped():
    """min(limit, 200) — what this server used before — passes every negative
    straight through, and SQLite reads LIMIT -1 as unbounded."""
    tr = Checks()
    with tempfile.TemporaryDirectory() as tmp:
        db = make_fixture_db(f"{tmp}/hgb.db")
        tr.check(db.clamp(-1, 50) == 50, "negative limit falls back to the default")
        tr.check(min(-1, 200) == -1, "the old min(limit, 200) guard let -1 through")
        tr.check(db.clamp(0, 50) == 50, "zero limit falls back to the default")
        tr.check(db.clamp("many", 50) == 50, "non-numeric limit falls back to the default")
        tr.check(db.clamp(10**9, 50) == db.MAX_LIMIT, f"huge limit capped at {db.MAX_LIMIT}")

        tr.check(len(db.list_dossiers(-1)) == 2, "list_dossiers(-1) is bounded")
        tr.check(len(db.get_dossier("dos-A", -1)) == 2, "get_dossier(-1) is bounded")
        tr.check(len(db.search_persons("Meier", -1)) <= db.MAX_LIMIT,
                 "search_persons(-1) is bounded")
        tr.check(len(db.get_persons_in_year_range(1400, 1700, -1)) <= db.MAX_LIMIT,
                 "get_persons_in_year_range(-1) is bounded")
    tr.assert_ok()


def test_like_wildcards_are_escaped():
    """A '%' or '_' in a query must match itself, not act as a wildcard."""
    tr = Checks()
    with tempfile.TemporaryDirectory() as tmp:
        db = make_fixture_db(f"{tmp}/hgb.db")
        tr.check(db.like_pattern("a%b_c") == "%a\\%b\\_c%", "like_pattern escapes both wildcards")

        # get_cooccurrences matches the person by LIKE: '%' must not select every doc.
        cooc = db.get_cooccurrences("100%")
        names = sorted(r["text"] for r in cooc)
        tr.check(names == ["Hans_Meier"],
                 f"'100%' matches only its own document's co-occurrences (got {names})")
        # A bare '%' finds the one person whose name contains a literal percent —
        # so its co-occurrences are that document's, not every document's. Unescaped
        # it would have selected the whole corpus, dragging in Zwingli and Hans Meier.
        bare = sorted(r["text"] for r in db.get_cooccurrences("%"))
        tr.check(bare == ["Hans_Meier"],
                 f"a bare '%' stays literal instead of selecting every document (got {bare})")
        tr.check(db.get_cooccurrences("Zwingli") == [],
                 "a person alone in a document has no co-occurrences")
    tr.assert_ok()


def test_fulltext_survives_hostile_queries():
    """FTS5 syntax errors must degrade to a literal search, not raise."""
    tr = Checks()
    with tempfile.TemporaryDirectory() as tmp:
        db = make_fixture_db(f"{tmp}/hgb.db")
        hits = db.search_persons("Meier")
        tr.check(hits and hits[0]["class"] == "per", f"plain query finds a person ({hits[:1]})")
        texts = db.search_text("pfund")
        tr.check(texts and texts[0]["id"] == "doc1", f"text search finds the document ({texts[:1]})")

        for q in ['Meier"', "Hans AND", "(unbalanced", "Mei*", "%"]:
            for fn, name in ((db.search_persons, "search_persons"), (db.search_text, "search_text")):
                try:
                    tr.check(isinstance(fn(q, 5), list), f"{name}({q!r}) returned a list")
                except Exception as e:
                    tr.check(False, f"{name}({q!r}) raised {type(e).__name__}: {e}")
        tr.check("error" in db.search_persons("")[0], "an empty query is reported as an error")
    tr.assert_ok()


def test_document_and_dossier_shape():
    tr = Checks()
    with tempfile.TemporaryDirectory() as tmp:
        db = make_fixture_db(f"{tmp}/hgb.db")
        doc = db.get_document("doc1")
        tr.check(doc.get("id") == "doc1", "get_document returns the document")
        tr.check(len(doc["spans"]) == 2, f"spans are attached (got {len(doc['spans'])})")
        tr.check(len(doc["events"]) == 1, f"events are attached (got {len(doc['events'])})")
        tr.check(db.get_document("nope") == {}, "an unknown document id returns {}")

        dossier = db.get_dossier("dos-A")
        tr.check([d["id"] for d in dossier] == ["doc1", "doc2"], "dossier documents are ordered by year")
        tr.check(db.get_dossier("nope") == [], "an unknown dossier returns []")
    tr.assert_ok()


def test_dossier_index_reports_truncation():
    """hgb://dossiers must say when it is only showing a prefix."""
    tr = Checks()
    with tempfile.TemporaryDirectory() as tmp:
        db = make_fixture_db(f"{tmp}/hgb.db")
        full = db.dossier_index()
        tr.check(full["total"] == 2 and full["returned"] == 2, "full index returns everything")
        tr.check(full["truncated"] is False, "full index is not flagged truncated")
        tr.check("note" not in full, "no truncation note when nothing is cut")

        cut = db.dossier_index(limit=1)
        tr.check(cut["total"] == 2 and cut["returned"] == 1, "truncated index reports both counts")
        tr.check(cut["truncated"] is True, "truncation is flagged")
        tr.check("list_dossiers" in cut.get("note", ""), "note points at list_dossiers")
    tr.assert_ok()


def test_connection_is_read_only():
    tr = Checks()
    with tempfile.TemporaryDirectory() as tmp:
        db = make_fixture_db(f"{tmp}/hgb.db")
        with db.conn() as c:
            try:
                c.execute("DELETE FROM documents")
                tr.check(False, "a write through db.conn() was accepted")
            except sqlite3.OperationalError as e:
                tr.check(True, f"writes are rejected ({e})")
        tr.check(db.db_stats()["n_documents"] == 3, "corpus is intact after the attempted write")
    tr.assert_ok()


def test_server_module_registers_tools():
    """server.py must import without reading sys.argv — it used to call
    parse_args() at import time, which hijacks the arguments of any process that
    imports it, pytest included."""
    pytest.importorskip("mcp", reason="mcp SDK not installed")
    import anyio
    import server as server_module

    tr = Checks()
    expected = {"corpus_stats", "search_persons", "get_document", "get_dossier",
                "search_text", "get_persons_in_year_range", "get_cooccurrences",
                "list_dossiers"}
    tools = anyio.run(server_module.mcp.list_tools)
    names = {t.name for t in tools}
    tr.check(not expected - names, f"all tools registered (missing: {sorted(expected - names)})")
    undocumented = sorted(t.name for t in tools if not (t.description or "").strip())
    tr.check(not undocumented,
             f"every tool carries a description for the client (missing: {undocumented})")

    n = server_module.normalise_path
    tr.check(n("/mcp/eos/mcp") == "/mcp/eos/mcp", "an already-correct path is unchanged")
    tr.check(n("mcp/eos/mcp/") == "/mcp/eos/mcp", "slashes are normalised")
    tr.check(n("") == "/mcp" and n(None) == "/mcp", "an empty path falls back to /mcp")
    tr.check(server_module.parse_args([]).http_path == "/mcp", "default endpoint path is /mcp")
    tr.check(server_module.parse_args(["--http-path", "mcp/eos/mcp/"]).http_path
             == "/mcp/eos/mcp", "--http-path is normalised on the way in")

    bad = server_module.get_persons_in_year_range(1600, 1500)
    tr.check(isinstance(bad, list) and "error" in bad[0], "an inverted year range is rejected")
    wide = server_module.get_persons_in_year_range(1000, 1900)
    tr.check(isinstance(wide, list) and "error" in wide[0], "an over-wide year range is rejected")
    tr.assert_ok()


# ── 2. DB tests — against the real corpus ─────────────────────────────────────

def test_db_layer_against_real_db(db_path):
    if not db_path or not os.path.exists(db_path):
        pytest.skip(f"hgb.db not found at {db_path!r} — set EOS_DB or pass --db")

    import db as db_module
    db_module.set_db_path(db_path)
    tr = Checks()

    s = db_module.db_stats()
    info(f"documents={s['n_documents']} spans={s['n_spans']} persons={s['n_persons']} "
         f"dossiers={s['n_dossiers']} years={s['year_min']}–{s['year_max']}")
    tr.check(s["n_documents"] > 100, f"documents: >100 (got {s['n_documents']})")
    tr.check(len(db_module.list_dossiers(-1)) <= db_module.MAX_LIMIT, "list_dossiers(-1) is bounded")

    for q in ["Meier", "%", "_", 'quote"mark', "Hans AND"]:
        for fn, name in ((db_module.search_persons, "search_persons"),
                         (db_module.search_text, "search_text")):
            try:
                tr.check(isinstance(fn(q, 5), list), f"{name}({q!r}) returned a list")
            except Exception as e:
                tr.check(False, f"{name}({q!r}) raised {type(e).__name__}: {e}")

    # A common name is the case that used to break get_cooccurrences with
    # 'too many SQL variables'.
    for name in ["Hans", "a"]:
        try:
            res = db_module.get_cooccurrences(name, 5)
            tr.check(isinstance(res, list), f"get_cooccurrences({name!r}) returned {len(res)} rows")
        except Exception as e:
            tr.check(False, f"get_cooccurrences({name!r}) raised {type(e).__name__}: {e}")

    idx = db_module.dossier_index()
    payload = json.dumps(idx, ensure_ascii=False)
    tr.check(len(payload) < 150_000,
             f"dossier index stays under the 150k result limit ({len(payload):,} chars)")
    tr.assert_ok()


# ── 3. Server integration test ────────────────────────────────────────────────

def _tool_payload(result):
    structured = getattr(result, "structured_content", None)
    if structured:
        return structured.get("result", structured)
    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if text:
            try: return json.loads(text)
            except json.JSONDecodeError: return text
    return None


def test_server(base_url):
    """Drive the running server over streamable HTTP using the official client."""
    if not base_url:
        pytest.skip("no server URL — set EOS_SERVER or pass --server")
    try:
        import anyio
        from mcp import ClientSession
        from mcp.client.streamable_http import streamable_http_client
    except ImportError as e:
        pytest.skip(f"mcp client library not available: {e}")

    tr = Checks()
    url = base_url.rstrip("/")
    if url.rsplit("/", 1)[-1] != "mcp":
        url += "/mcp"

    async def exercise():
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                names = {t.name for t in (await session.list_tools()).tools}
                tr.check("corpus_stats" in names, f"tools/list returned {len(names)} tools")

                stats = _tool_payload(await session.call_tool("corpus_stats", {}))
                tr.check(isinstance(stats, dict) and stats.get("n_documents", 0) > 0,
                         f"corpus_stats returns documents (got {stats})")

                for tool, args in [
                    ("search_persons",   {"query": "Meier", "limit": 3}),
                    ("search_text",      {"query": "pfund", "limit": 3}),
                    ("list_dossiers",    {"limit": 3}),
                    ("get_cooccurrences", {"person_name": "Hans", "limit": 3}),
                    ("get_persons_in_year_range", {"year_from": 1450, "year_to": 1500, "limit": 3}),
                ]:
                    res = await session.call_tool(tool, args)
                    tr.check(not res.is_error, f"{tool} call succeeded")

                res = await session.call_tool("search_text", {"query": 'pfund"', "limit": 3})
                tr.check(not res.is_error, "search_text survives an unbalanced quote")

                payload = _tool_payload(await session.call_tool("get_document",
                                                               {"doc_id": "definitely_not_an_id"}))
                tr.check(isinstance(payload, dict) and "error" in payload,
                         f"unknown document id returns an error object (got {payload})")
                # Der Text allein ist nicht maschinell von einem Ausfall zu
                # unterscheiden; das Feld ist es (ch-h-bot#497).
                tr.check(isinstance(payload, dict) and payload.get("not_found") is True,
                         f"unknown document id says so in `not_found` (got {payload})")

                payload = _tool_payload(await session.call_tool(
                    "get_persons_in_year_range", {"year_from": 1600, "year_to": 1500}))
                tr.check(isinstance(payload, list) and "error" in payload[0],
                         "an inverted year range comes back as data")

                resources = {str(r.uri) for r in (await session.list_resources()).resources}
                tr.check("hgb://stats" in resources, f"hgb://stats listed (got {sorted(resources)})")

    anyio.run(exercise)
    tr.assert_ok()


# ── CLI ───────────────────────────────────────────────────────────────────────

def cli_run(label, fn, *fn_args):
    print(f"\n{label}")
    try:
        fn(*fn_args)
        return True
    except pytest.skip.Exception as e:
        warn(f"skipped: {e}")
        return True
    except AssertionError as e:
        fail(str(e))
        return False
    except Exception as e:
        fail(f"{type(e).__name__}: {e}")
        return False


# ── The vendored embeddings.py ───────────────────────────────────────────────

@pytest.mark.parametrize("entry", ["server", "embed_db"])
def test_the_vendored_embeddings_read_the_eos_settings(entry):
    # embeddings.py is byte-identical across the fleet and finds EOS_EMBED_MODEL
    # and the other EOS_* settings only through MCP_ENV_PREFIX, which corpus.py
    # sets before importing it. Without that, an override would be read by
    # nothing and queries would be embedded with a model the index was not
    # built with.
    import subprocess
    env = {k: v for k, v in os.environ.items() if k != "MCP_ENV_PREFIX"}
    env["EOS_EMBED_MODEL"] = "probe-model"
    out = subprocess.run(
        [sys.executable, "-c",
         # server.py imports corpus inside search_semantic, the first place
         # it needs a vector; this is that order.
         f"import {entry}, corpus, embeddings; "
         "print(embeddings.ENV_PREFIX, embeddings.DEFAULT_MODEL)"],
        cwd=os.path.dirname(os.path.abspath(__file__)), env=env,
        capture_output=True, text=True)
    assert out.stdout.split() == ["EOS", "probe-model"], out.stderr


def test_queries_carry_this_corpus_instruction():
    # The shared embed_query only knows "Swiss history"; corpus.embed_query
    # keeps the instruction this index has always been queried with.
    import corpus as corpus_module
    seen = {}

    class FakeClient:
        class embeddings:
            @staticmethod
            def create(model, input):
                seen["input"] = input
                return type("R", (), {"data": [type("D", (), {"embedding": [1.0, 0.0]})()]})()

    corpus_module.embed_query("Wer?", client=FakeClient())
    assert seen["input"][0].startswith("Instruct:")
    assert "land-register" in seen["input"][0]
    assert seen["input"][0].endswith("Wer?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="EOS MCP test suite")
    ap.add_argument("--unit", action="store_true", help="Run unit tests")
    ap.add_argument("--db", default=os.environ.get("EOS_DB", ""), help="Path to hgb.db")
    ap.add_argument("--server", default=os.environ.get("EOS_SERVER", ""), help="Server URL")
    args = ap.parse_args()

    if not args.unit and not args.db and not args.server:
        ap.print_help(); sys.exit(0)

    print(f"{'═'*50}\nEOS MCP test suite\n{'═'*50}")
    ok_all = True

    if args.unit:
        ok_all &= cli_run("[1] Unit: limits are clamped", test_limits_are_clamped)
        ok_all &= cli_run("[2] Unit: LIKE wildcards are escaped", test_like_wildcards_are_escaped)
        ok_all &= cli_run("[3] Unit: full-text survives hostile queries",
                          test_fulltext_survives_hostile_queries)
        ok_all &= cli_run("[4] Unit: document and dossier shape", test_document_and_dossier_shape)
        ok_all &= cli_run("[5] Unit: dossier index reports truncation",
                          test_dossier_index_reports_truncation)
        ok_all &= cli_run("[6] Unit: connection is read-only", test_connection_is_read_only)
        ok_all &= cli_run("[7] Unit: server registers its tools", test_server_module_registers_tools)

    if args.db:
        ok_all &= cli_run(f"[8] DB: query layer ({args.db})", test_db_layer_against_real_db, args.db)

    if args.server:
        ok_all &= cli_run(f"[9] Server: MCP integration ({args.server})", test_server, args.server)

    print(f"\n{'═'*50}")
    print(f"{GREEN}ALL PASSED{RESET}" if ok_all else f"{RED}FAILURES{RESET}")
    sys.exit(0 if ok_all else 1)


# ── Semantic search ───────────────────────────────────────────────────────────
# Written before this shipped, deliberately. Porting the same machinery into
# ssrq_mcp produced four failures in a row — a module missing from the image, a
# missing module constant, another corpus's columns in the SQL, and a row key
# that did not match the alias. Every one surfaced to the caller as the same
# opaque "Error executing tool search_semantic". One test over a real database
# with real vectors catches all four.

def _semantic_db(tmp_path):
    import sqlite3
    import struct

    import db
    import vectorstore

    path = tmp_path / "semantic.db"
    conn = sqlite3.connect(path)
    conn.executescript(db.SCHEMA_SQL)
    conn.executescript(vectorstore.SCHEMA_SQL)
    conn.execute(
        "INSERT INTO documents (id, dossier_id, year, source, location, language, "
        "pages, text_raw, checked) VALUES "
        "('doc-1','HGB 42',1512,'StABS',NULL,'de',2,"
        "'Item hat Hans Meyer sin hus by dem Rhyn verkoufft.',1)")
    # The passage as embed_db.py writes it: the head on the first line, the
    # offsets into the cleaned text below it.
    conn.execute(
        "INSERT INTO chunks (chunk_id, parent_id, chunk_index, char_start, char_end, text) "
        "VALUES ('doc-1#0','doc-1',0,0,48,"
        "'HGB 42 1512' || char(10) || 'Item hat Hans Meyer sin hus by dem Rhyn verkoufft.')")
    dims = 8
    vector = [1.0] + [0.0] * (dims - 1)
    conn.execute(
        "INSERT INTO embeddings (chunk_id, model, dims, vector) VALUES (?,?,?,?)",
        ("doc-1#0", "test-model", dims, struct.pack(f"<{dims}f", *vector)))
    conn.commit(); conn.close()
    db.set_db_path(str(path))        # also drops vectorstore's matrix cache
    return db, vector


def test_search_semantic_end_to_end(tmp_path):
    db, vector = _semantic_db(tmp_path)

    hits = db.search_semantic(vector, limit=5, model="test-model")

    assert len(hits) == 1
    hit = hits[0]
    assert hit["id"] == "doc-1"
    assert hit["year"] == 1512
    # The HGB has no titles: an entry is placed by its dossier.
    assert hit["dossier_id"] == "HGB 42"
    assert hit["source"] == "StABS" and hit["language"] == "de"
    assert hit["score"] > 0.99
    # The snippet starts with the text; the head stays with the vector
    # (ch-h-bot#555). The offsets are untouched by the cut.
    assert hit["snippet"] == "Item hat Hans Meyer sin hus by dem Rhyn verkoufft."
    assert (hit["chunk_index"], hit["char_start"], hit["char_end"]) == (0, 0, 48)


def test_the_snippet_drops_only_this_documents_head():
    import db

    body = "Item hat Hans Meyer sin hus verkoufft."
    assert db.passage_body("HGB 42 1512\n" + body, "HGB 42", 1512) == body
    # Verbatim head: a dossier with a blank at the edge is cut as written.
    assert db.passage_body("HGB 9  1450\n" + body, "HGB 9 ", 1450) == body
    # A passage without a head, or with another document's head, is untouched.
    assert db.passage_body(body, "HGB 42", 1512) == body
    assert db.passage_body("HGB 43 1512\n" + body, "HGB 42", 1512) == "HGB 43 1512\n" + body
    # No dossier and no year means no head — nothing is cut, not even a blank line.
    assert db.passage_body("\n" + body, None, None) == "\n" + body
    # The head alone is not a passage with a head line.
    assert db.passage_body("HGB 42 1512", "HGB 42", 1512) == "HGB 42 1512"


def test_the_year_filter_works(tmp_path):
    import sqlite3
    import struct

    db, vector = _semantic_db(tmp_path)
    # A second document without a year: a year filter must drop it, no filter keeps it.
    conn = sqlite3.connect(db._DB_PATH)
    conn.execute("INSERT INTO documents (id, dossier_id, year, text_raw) "
                 "VALUES ('doc-2','HGB 7',NULL,'Anna Sicher und Hans Meier')")
    conn.execute("INSERT INTO chunks VALUES ('doc-2#0','doc-2',0,0,26,"
                 "'HGB 7' || char(10) || 'Anna Sicher und Hans Meier')")
    conn.execute("INSERT INTO embeddings VALUES (?,?,?,?)",
                 ("doc-2#0", "test-model", 8, struct.pack("<8f", 0.9, 0.1, 0, 0, 0, 0, 0, 0)))
    conn.commit(); conn.close()
    db.set_db_path(db._DB_PATH)

    ids = lambda hits: [h["id"] for h in hits]
    assert ids(db.search_semantic(vector, model="test-model")) == ["doc-1", "doc-2"]
    assert db.search_semantic(vector, model="test-model", year_from=1600) == []
    assert ids(db.search_semantic(vector, model="test-model", year_to=1600)) == ["doc-1"]
    assert ids(db.search_semantic(vector, model="test-model", year_from=1500)) == ["doc-1"]
    assert db.search_semantic(vector, model="test-model", year_from=1500, year_to=1511) == []


def test_semantic_stats_reports_coverage(tmp_path):
    db, _ = _semantic_db(tmp_path)

    stats = db.semantic_stats()

    assert stats["n_chunks"] == 1
    assert stats["indexed"] is True
    # The shared keys (#6): n_indexed/n_total, not n_entries_indexed/n_entries_total.
    assert (stats["n_indexed"], stats["n_total"], stats["coverage"]) == (1, 1, 1.0)
    assert not {k for k in stats if k.startswith("n_entries")}
    assert stats["models"] == [{"model": "test-model", "n_vectors": 1, "dims": 8}]


def test_every_sql_statement_matches_the_schema(tmp_path):
    """A missing column only shows when the statement runs, and inside a tool
    that reaches the caller as an opaque failure."""
    import sqlite3

    import db
    import vectorstore

    conn = sqlite3.connect(tmp_path / "schema.db")
    conn.executescript(db.SCHEMA_SQL)
    conn.executescript(vectorstore.SCHEMA_SQL)
    conn.execute(db._JOIN_SQL.format(placeholders="?"), ("x",)).fetchall()
    conn.close()


def test_every_module_is_copied_into_the_image():
    """The Dockerfile lists modules individually rather than `COPY . .`.

    hls_mcp shipped without embeddings.py: the build succeeded and the
    container crash-looped on the import, taking the largest provider in the
    federation offline. Only COPY lines are scanned — an earlier version of
    this check elsewhere matched the module name in its own comment and passed
    with the COPY removed. embed_db.py and migrate_parent_id.py run against
    the database on the host, not in the server's image.
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent
    copy_lines = [line for line in (root / "Dockerfile").read_text(encoding="utf-8").splitlines()
                  if line.strip().startswith("COPY")]
    copied = set(re.findall(r"([\w]+\.py)", " ".join(copy_lines)))
    shipped = {p.name for p in root.glob("*.py")
               if not p.name.startswith("test_")
               and p.name not in {"embed_db.py", "migrate_parent_id.py"}}
    assert not shipped - copied, f"not COPYed into the image: {sorted(shipped - copied)}"


# ── The index as the old embed_db.py wrote it ─────────────────────────────────
# Frozen from the repo's own embed_db.py at 119a782 on 22.10.2026, before it
# was replaced by the shared one (#6): the same fixture, run through the old
# build_chunks, gave exactly these rows. The production index on idhefix was
# built by that code, and ch-h-bot#555 measured that the shared build_chunks
# reproduces all 76,737 of its passages row for row. These rows are what that
# measurement rests on, in the repo, so the quirks of the old clean
# («verkoufftund», «istder»), the verbatim head («HGB 9  1450», two blanks;
# « HGB 8 1460», a leading one), a
# document without a year or without a dossier, and a document windowed five
# times with the head on every window are each pinned to one line.

_LONG_TEXT = " ".join(
    f"Item {i} der Hein r₎ rich Meyer hat geben un̄ und zwei pfund ✳ pfennig ."
    for i in range(60))

# id, dossier_id, year, source, location, language, pages, text_raw, checked
OLD_INDEX_DOCUMENTS = [
    # «Hein» is a word twice and a fragment once here; across the table it is a
    # fragment 63 times. The old vocabulary saw the table, so this joins.
    ("d0", "HGB 1", 1400, "StABS", None, "de", 1, "Hein und Hein und Hein r₎ rich", 1),
    ("d1", "HGB 42", 1512, "StABS", None, "de", 2,
     "Item hat Hein r₎ rich Meyer sin hus by dem Rhyn verkoufft ✳ un̄ und bezalt .", 1),
    ("d2", "HGB 42", 1513, "StABS", None, "de", 1, "Diz ist dˀ der brief des Hein ✳ Hein r₎ rich", 1),
    ("d3", "HGB 7", None, "StABS", None, "de", 1, "Anna Sicher un̄ und Hans Meier", 1),
    ("d4", None, None, "StABS", None, "de", 1, "Ulrich wird genannt", 0),
    ("d5", "HGB 9 ", 1450, "StABS", None, "de", 1, "stif tˀin terin Oͤs r₎ wald", 1),
    ("d6", "HGB 100", 1600, "StABS", None, "de", 9, _LONG_TEXT, 1),
    ("d7", "HGB 101", 1601, "StABS", None, "de", 1, "", 1),       # empty: no passage
    ("d8", "HGB 102", 1602, "StABS", None, "de", 1, None, 1),     # NULL: no passage
    # A blank at the edge of the dossier id stays in the head: a strip() would
    # write a head the vectors were not made from.
    ("d9", " HGB 8", 1460, "StABS", None, "de", 1, "Peter Ruf wird genannt", 1),
]

OLD_INDEX_CLEANED = {
    "d0": "Hein und Hein und Heinrich",
    "d1": "Item hat Heinrich Meyer sin hus by dem Rhyn verkoufftund bezalt.",
    "d2": "Diz istder brief des Hein Heinrich",
    "d3": "Anna Sicherund Hans Meier",
    "d4": "Ulrich wird genannt",
    "d5": "stifterin Oͤswald",
    "d9": "Peter Ruf wird genannt",
}

OLD_INDEX_HEADS = {"d0": "HGB 1 1400", "d1": "HGB 42 1512", "d2": "HGB 42 1513",
                   "d3": "HGB 7", "d4": "", "d5": "HGB 9  1450", "d6": "HGB 100 1600",
                   "d9": " HGB 8 1460"}

# chunk_id, doc_id, chunk_index, char_start, char_end, text — the short ones in
# full; d6's five windows by length and digest, each starting with its head.
OLD_INDEX_ROWS = [
    ("d0#0", "d0", 0, 0, 26, "HGB 1 1400\nHein und Hein und Heinrich"),
    ("d1#0", "d1", 0, 0, 64, "HGB 42 1512\nItem hat Heinrich Meyer sin hus by dem Rhyn verkoufftund bezalt."),
    ("d2#0", "d2", 0, 0, 34, "HGB 42 1513\nDiz istder brief des Hein Heinrich"),
    ("d3#0", "d3", 0, 0, 25, "HGB 7\nAnna Sicherund Hans Meier"),
    ("d4#0", "d4", 0, 0, 19, "Ulrich wird genannt"),
    ("d5#0", "d5", 0, 0, 17, "HGB 9  1450\nstifterin Oͤswald"),
    ("d9#0", "d9", 0, 0, 22, " HGB 8 1460\nPeter Ruf wird genannt"),
]
OLD_INDEX_D6_ROWS = [   # chunk_id, char_start, char_end, len(text), sha256(text)[:16]
    ("d6#0", 0, 950, 962, "0f2cd8958c6c8683"),
    ("d6#1", 800, 1790, 1001, "0fd3175da292dbd0"),
    ("d6#2", 1640, 2630, 1001, "da1feea09ac64d01"),
    ("d6#3", 2480, 3470, 1001, "e2620a5f719d9b47"),
    ("d6#4", 3320, 3589, 281, "b225f136a8c44505"),
]

# db.EMBEDDING_SCHEMA_SQL as it was at 119a782: the layout of the index on
# idhefix before migrate_parent_id.py.
OLD_EMBEDDING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT    PRIMARY KEY,
    doc_id    TEXT    NOT NULL,
    chunk_index INTEGER NOT NULL,
    char_start  INTEGER NOT NULL,
    char_end    INTEGER NOT NULL,
    text        TEXT    NOT NULL,
    UNIQUE (doc_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT    PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model    TEXT    NOT NULL,
    dims     INTEGER NOT NULL,
    vector   BLOB    NOT NULL
);
CREATE TABLE IF NOT EXISTS embedding_runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    model       TEXT NOT NULL,
    dims        INTEGER,
    base_url    TEXT,
    chunk_chars INTEGER,
    chunk_overlap INTEGER,
    n_articles  INTEGER,
    n_chunks    INTEGER,
    notes       TEXT
);
CREATE INDEX IF NOT EXISTS ix_chunks_entry ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS ix_embeddings_model ON embeddings(model);
"""


def _rows_match_the_old_index(rows):
    """Every chunk row, against what the old embed_db.py wrote."""
    import hashlib

    rows = sorted(tuple(r) for r in rows)
    short = [r for r in rows if not r[0].startswith("d6#")]
    assert short == OLD_INDEX_ROWS
    long = [(r[0], r[3], r[4], len(r[5]), hashlib.sha256(r[5].encode("utf-8")).hexdigest()[:16])
            for r in rows if r[0].startswith("d6#")]
    assert long == OLD_INDEX_D6_ROWS
    for r in rows:
        if r[0].startswith("d6#"):
            assert r[1] == "d6" and r[2] == int(r[0].split("#")[1])
            assert r[5].startswith("HGB 100 1600\n")


def _old_index_db(path):
    """A database in the OLD layout holding the frozen rows, one vector per chunk."""
    import sqlite3
    import struct

    import db

    conn = sqlite3.connect(path)
    conn.executescript(db.SCHEMA_SQL)
    conn.executescript(db.TRIGGERS_SQL)
    conn.executescript(OLD_EMBEDDING_SCHEMA_SQL)
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)", OLD_INDEX_DOCUMENTS)
    conn.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", OLD_INDEX_ROWS)
    cleaned = _cleaned_d6()
    for chunk_id, start, end, _, _ in OLD_INDEX_D6_ROWS:
        conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                     (chunk_id, "d6", int(chunk_id.split("#")[1]), start, end,
                      "HGB 100 1600\n" + cleaned[start:end].strip()))
    n = len(OLD_INDEX_ROWS) + len(OLD_INDEX_D6_ROWS)
    for i, (chunk_id,) in enumerate(conn.execute("SELECT chunk_id FROM chunks ORDER BY chunk_id").fetchall()):
        vector = [0.0] * n
        vector[i] = 1.0
        conn.execute("INSERT INTO embeddings VALUES (?,?,?,?)",
                     (chunk_id, "test-model", n, struct.pack(f"<{n}f", *vector)))
    conn.execute("INSERT INTO embedding_runs (run_id, started_at, finished_at, model, dims, "
                 "n_articles, n_chunks) VALUES ('run-1','2026-10-01T00:00:00Z',"
                 "'2026-10-01T00:10:00Z','test-model',?,8,?)", (n, n))
    conn.commit()
    return conn


def _cleaned_d6():
    import corpus

    vocab = corpus.build_vocab(d[7] for d in OLD_INDEX_DOCUMENTS if d[7])
    return corpus.clean_entry_text(_LONG_TEXT, vocab)


def test_head_and_clean_are_what_the_old_embed_db_wrote(tmp_path):
    """corpus.iter_documents and corpus.clean, against the frozen rows."""
    import hashlib
    import sqlite3

    import corpus
    import db

    conn = sqlite3.connect(tmp_path / "corpus.db")
    conn.executescript(db.SCHEMA_SQL)
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)", OLD_INDEX_DOCUMENTS)

    docs = list(corpus.iter_documents(conn))
    assert [d[0] for d in docs] == ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d9"]
    assert {d[0]: d[1] for d in docs} == OLD_INDEX_HEADS
    cleaned = {d[0]: corpus.clean(d[2]) for d in docs}
    assert {k: cleaned[k] for k in OLD_INDEX_CLEANED} == OLD_INDEX_CLEANED
    assert len(cleaned["d6"]) == 3589
    assert hashlib.sha256(cleaned["d6"].encode("utf-8")).hexdigest()[:16] == "95c5f355a3e066d2"


def test_the_vocabulary_covers_the_whole_table_whatever_the_limit(tmp_path):
    """--limit samples the documents, not the evidence clean() decides on."""
    import sqlite3

    import corpus
    import db

    conn = sqlite3.connect(tmp_path / "corpus.db")
    conn.executescript(db.SCHEMA_SQL)
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)", OLD_INDEX_DOCUMENTS)

    docs = list(corpus.iter_documents(conn, limit=1))
    assert [d[0] for d in docs] == ["d0"]
    assert corpus.clean(docs[0][2]) == "Hein und Hein und Heinrich"
    # With d0's own evidence alone, «Hein» is a word twice and a fragment once,
    # and the expansion would not join.
    alone = corpus.build_vocab([docs[0][2]])
    assert corpus.clean_entry_text(docs[0][2], alone) == "Hein und Hein und Hein rich"


def test_clean_refuses_without_the_vocabulary():
    import corpus

    corpus._VOCAB = None
    with pytest.raises(RuntimeError, match="iter_documents"):
        corpus.clean("Hein r₎ rich")


def test_the_shared_embed_db_writes_the_old_index(tmp_path):
    """The shared build_chunks, through corpus.py, reproduces the frozen rows —
    chunk id, parent, index, offsets, text — on an empty index."""
    import sqlite3

    import db
    import embed_db
    import vectorstore

    conn = sqlite3.connect(tmp_path / "fresh.db")
    conn.executescript(db.SCHEMA_SQL)
    conn.executescript(vectorstore.SCHEMA_SQL)
    conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?)", OLD_INDEX_DOCUMENTS)
    conn.commit()

    assert embed_db.build_chunks(conn, None, 1000, 150, True) == 12
    _rows_match_the_old_index(conn.execute(
        "SELECT chunk_id, parent_id, chunk_index, char_start, char_end, text FROM chunks"))


# ── The migration ─────────────────────────────────────────────────────────────

def test_migration_keeps_every_vector_and_leaves_nothing_pending(tmp_path, capsys):
    """An index in the old layout, renamed in place: afterwards the shared
    embed_db.py finds every chunk with its vector, chunks nothing new and
    embeds nothing, and search_semantic answers through vectorstore."""
    import sqlite3

    import db
    import embed_db
    import migrate_parent_id

    path = tmp_path / "hgb.db"
    _old_index_db(path).close()

    assert migrate_parent_id.main(["--db", str(path)]) == 0
    out = capsys.readouterr().out
    assert "integrity_check: ok" in out
    assert "RENAME COLUMN doc_id TO parent_id" in out
    assert "'chunks': 12" in out and "'embeddings': 12" in out and "'parents': 8" in out

    conn = sqlite3.connect(path)
    assert "parent_id" in migrate_parent_id.columns(conn, "chunks")
    assert "doc_id" not in migrate_parent_id.columns(conn, "chunks")
    assert "n_parents" in migrate_parent_id.columns(conn, "embedding_runs")
    assert sorted(migrate_parent_id.indexes(conn, "chunks")) == sorted(
        ["ix_chunks_parent", "sqlite_autoindex_chunks_1", "sqlite_autoindex_chunks_2"])
    # The renamed column still carries the UNIQUE constraint.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO chunks VALUES ('d0#9','d0',0,0,1,'x')")

    # The shared embed_db.py on the migrated index: every row is already there
    # (INSERT OR IGNORE finds its twin), nothing is pending, no client is needed.
    before = conn.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall()
    embed_db.build_chunks(conn, None, 1000, 150, True)
    assert conn.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall() == before
    _rows_match_the_old_index(before)
    assert embed_db.embed_pending(conn, "test-model", 64, True) == (0, 0)
    conn.close()

    assert embed_db.main(["--db", str(path), "--status"]) == 0
    out = capsys.readouterr().out
    # d7 and d8 have no text, so 8 of 10 documents carry passages.
    assert "n_chunks: 12" in out and "n_indexed: 8" in out and "n_total: 10" in out
    assert "coverage: 0.8" in out

    db.set_db_path(str(path))
    vector = [0.0] * 12
    vector[1] = 1.0                      # d1#0 is second in chunk_id order
    hits = db.search_semantic(vector, model="test-model", limit=3)
    assert hits[0]["id"] == "d1" and hits[0]["chunk_id"] == "d1#0"
    assert hits[0]["snippet"] == "Item hat Heinrich Meyer sin hus by dem Rhyn verkoufftund bezalt."
    assert (hits[0]["char_start"], hits[0]["char_end"]) == (0, 64)
    stats = db.semantic_stats()
    assert (stats["n_chunks"], stats["n_indexed"], stats["n_total"]) == (12, 8, 10)
    assert stats["recent_runs"][0]["run_id"] == "run-1"


def test_migration_is_idempotent(tmp_path, capsys):
    import sqlite3

    import migrate_parent_id

    path = tmp_path / "hgb.db"
    _old_index_db(path).close()
    assert migrate_parent_id.main(["--db", str(path)]) == 0
    schema = sqlite3.connect(path).execute(
        "SELECT sql FROM sqlite_master ORDER BY name").fetchall()
    capsys.readouterr()

    assert migrate_parent_id.main(["--db", str(path)]) == 0
    out = capsys.readouterr().out
    assert "already migrated, nothing to do" in out
    assert sqlite3.connect(path).execute(
        "SELECT sql FROM sqlite_master ORDER BY name").fetchall() == schema
    # One index on parent_id, not two: DROP INDEX is what the shared SCHEMA_SQL needs.
    conn = sqlite3.connect(path)
    conn.executescript(__import__("vectorstore").SCHEMA_SQL)
    assert [i for i in migrate_parent_id.indexes(conn, "chunks") if i.startswith("ix_")] \
        == ["ix_chunks_parent"]


def test_migration_rolls_back_as_one_transaction(tmp_path):
    """Half a rename is worse than none: if a statement fails, nothing changed."""
    import sqlite3

    import migrate_parent_id

    path = tmp_path / "hgb.db"
    conn = _old_index_db(path)
    # A second index with the target name makes the last CREATE INDEX fail.
    conn.execute("CREATE INDEX ix_chunks_parent ON embeddings(model)")
    conn.commit()
    with pytest.raises(sqlite3.OperationalError):
        migrate_parent_id.migrate(conn)
    assert "doc_id" in migrate_parent_id.columns(conn, "chunks")
    assert "n_articles" in migrate_parent_id.columns(conn, "embedding_runs")
    assert "ix_chunks_entry" in migrate_parent_id.indexes(conn, "chunks")


def test_the_new_code_needs_the_migrated_db(tmp_path):
    """Why the server stops before the rename and starts after the pull: the
    new db.py asks for parent_id, the old index does not have it."""
    import sqlite3

    import db

    path = tmp_path / "hgb.db"
    _old_index_db(path).close()
    db.set_db_path(str(path))
    vector = [1.0] + [0.0] * 11
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        db.search_semantic(vector, model="test-model")


def test_migration_without_an_index_does_nothing(tmp_path, capsys):
    import migrate_parent_id

    db = make_fixture_db(f"{tmp_path}/hgb.db")
    assert migrate_parent_id.main(["--db", db._DB_PATH]) == 0
    assert "no semantic index" in capsys.readouterr().out
    assert migrate_parent_id.main(["--db", f"{tmp_path}/missing.db"]) == 1
