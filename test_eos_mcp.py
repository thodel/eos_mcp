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
