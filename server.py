"""server.py — EOS / HGB Basel MCP server (mcp 2.0 MCPServer, streamable HTTP)."""
import argparse, json, logging, os
from mcp.server.mcpserver import MCPServer
import db as db_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Defaults come from the environment so that importing this module never touches
# sys.argv — argparse at import time would hijack the arguments of any process that
# imports the server (tests, an ASGI loader). The CLI overrides these in main().
DEFAULT_DB   = os.environ.get("EOS_DB", "/data/hgb.db")
DEFAULT_HOST = os.environ.get("EOS_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.environ.get("EOS_PORT", "8000"))

MAX_YEAR_SPAN = 300


def normalise_path(path):
    """A single leading slash, no trailing slash — the form the ASGI route wants.

    Behind a reverse proxy the server must answer on its *public* path: the
    streamable-HTTP transport builds no URLs of its own, but the route only
    matches what it was mounted at. Setting this to the public path (e.g.
    /mcp/eos/mcp) lets nginx proxy_pass without rewriting, which is the mismatch
    that makes a sub-path deployment 404."""
    cleaned = (path or "").strip().strip("/")
    return f"/{cleaned}" if cleaned else "/mcp"


DEFAULT_HTTP_PATH = normalise_path(os.environ.get("EOS_HTTP_PATH", "/mcp"))

db_module.set_db_path(DEFAULT_DB)

mcp = MCPServer(
    name="EOS HGB Basel",
    version="1.0.0",
    instructions=(
        "This server provides access to the Historisches Grundbuch Basel (HGB), "
        "a corpus of historical land-register documents from Basel, ca. 1400–1700. "
        "Documents contain tokenised medieval German text annotated with persons, "
        "dates, money amounts, and legal events (payments, transfers, etc.). "
        "Use search_persons to find person mentions; get_document for full detail; "
        "search_text for keyword search across the raw transcriptions."
    ),
)

# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_semantic(query: str, limit: int = 20, year_from: int = 0,
                    year_to: int = 0, per_document: int = 2) -> list[dict]:
    """Land-register passages that answer a question, matched by meaning.

    The reason this server embeds its own corpus. Measured before it was built:
    a question in modern German returned zero keyword hits against this
    material, because the entries are tokenised medieval German with high
    edition noise and the caller would have to know the scribe's spelling.

    `year_from`/`year_to` restrict to a period. `per_document` caps how many
    passages one entry may contribute.
    """
    import embeddings as emb

    vector = emb.embed_query(query)
    return db_module.search_semantic(
        vector, limit=limit, year_from=year_from or None,
        year_to=year_to or None, per_document=per_document)


@mcp.tool()
def semantic_index_stats() -> dict:
    """Whether the semantic index is built, and over how much of the corpus.

    Worth checking before trusting an empty result: coverage below 1.0 means
    passages are missing, not that the corpus has nothing to say.
    """
    return db_module.semantic_stats()



@mcp.tool()
def corpus_stats() -> dict:
    """High-level counts for the HGB corpus, plus the year range covered."""
    return db_module.db_stats()

@mcp.tool()
def search_persons(query: str, limit: int = 20) -> list[dict]:
    """Full-text search over person mentions. Returns the span, its document, year,
    source, and location."""
    return db_module.search_persons(query, limit)

@mcp.tool()
def get_document(doc_id: str) -> dict:
    """Full document: metadata, raw text, all annotated spans, and all events."""
    result = db_module.get_document(doc_id)
    if not result:
        # `not_found`: «kenne ich nicht» maschinell von einem Ausfall zu
        # unterscheiden (ch-h-bot#497). Der Text bleibt, weil dieser Server
        # auch direkt von Menschen und Modellen gefragt wird.
        return {"error": f"Document '{doc_id}' not found.", "not_found": True}
    return result

@mcp.tool()
def get_dossier(dossier_id: str, limit: int = 100) -> list[dict]:
    """All documents belonging to one dossier (a property's file), oldest first."""
    results = db_module.get_dossier(dossier_id, limit)
    if not results:
        return [{"error": f"Dossier '{dossier_id}' not found.", "not_found": True}]
    return results

@mcp.tool()
def search_text(query: str, limit: int = 20) -> list[dict]:
    """Full-text search across the raw transcriptions. Returns snippets with highlights."""
    return db_module.search_text(query, limit)

@mcp.tool()
def get_persons_in_year_range(year_from: int, year_to: int, limit: int = 100) -> list[dict]:
    """Person mentions in documents dated within a year range (inclusive)."""
    if year_to < year_from:
        return [{"error": "year_to must be >= year_from"}]
    if year_to - year_from > MAX_YEAR_SPAN:
        return [{"error": f"Year range too large; max {MAX_YEAR_SPAN} years."}]
    return db_module.get_persons_in_year_range(year_from, year_to, limit)

@mcp.tool()
def get_cooccurrences(person_name: str, limit: int = 20) -> list[dict]:
    """Other persons mentioned in the same documents as a given person, by frequency."""
    return db_module.get_cooccurrences(person_name, limit)

@mcp.tool()
def list_dossiers(limit: int = 200) -> list[dict]:
    """List dossiers with their document counts and year ranges."""
    return db_module.list_dossiers(limit)

# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("hgb://stats")
def resource_stats() -> str:
    return json.dumps(db_module.db_stats(), indent=2)

@mcp.resource("hgb://dossiers")
def resource_dossiers() -> str:
    """Dossier index: id, year range, document count. Flags its own truncation."""
    return json.dumps(db_module.dossier_index(), indent=2, ensure_ascii=False)

@mcp.resource("hgb://document/{doc_id}")
def resource_document(doc_id: str) -> str:
    result = db_module.get_document(doc_id)
    if not result:
        return json.dumps({"error": f"Document '{doc_id}' not found.",
                           "not_found": True})
    return json.dumps(result, indent=2, ensure_ascii=False)

# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="EOS / HGB Basel MCP server")
    ap.add_argument("--db",   default=DEFAULT_DB,   help="Path to hgb.db (env EOS_DB)")
    ap.add_argument("--host", default=DEFAULT_HOST, help="Bind address (env EOS_HOST)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port (env EOS_PORT)")
    ap.add_argument("--http-path", default=DEFAULT_HTTP_PATH,
                    help="Path the MCP endpoint is served at; set it to the public "
                         "path when behind a reverse proxy (env EOS_HTTP_PATH)")
    args = ap.parse_args(argv)
    args.http_path = normalise_path(args.http_path)
    return args

def main(argv=None):
    args = parse_args(argv)
    db_module.set_db_path(args.db)

    logger.info(f"Database: {args.db}")
    try:
        s = db_module.db_stats()
        logger.info(f"Corpus: {s['n_documents']:,} documents, {s['n_persons']:,} person spans, "
                    f"{s['n_dossiers']:,} dossiers")
    except Exception as e:
        logger.warning(f"Could not read DB stats: {e}")
    logger.info(f"Starting EOS MCP server on {args.host}:{args.port}{args.http_path}")
    mcp.run(transport="streamable-http", host=args.host, port=args.port,
            streamable_http_path=args.http_path)

if __name__ == "__main__":
    main()
