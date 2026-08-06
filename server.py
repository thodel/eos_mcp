# This is the patched server.py with host/port in __init__ not run()

import argparse
import json
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

import db as db_module

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ap = argparse.ArgumentParser()
ap.add_argument("--db",   default="/data/hgb.db",  help="Path to hgb.db")
ap.add_argument("--host", default="0.0.0.0", help="Bind host")
ap.add_argument("--port", type=int, default=8000, help="Bind port")
args = ap.parse_args()

db_module.set_db_path(args.db)

mcp = FastMCP(
    name="EOS HGB Basel",
    host=args.host,
    port=args.port,
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
def corpus_stats() -> dict:
    """Return high-level statistics about the HGB corpus."""
    return db_module.db_stats()

@mcp.tool()
def search_persons(query: str, limit: int = 20) -> list[dict]:
    limit = min(limit, 200)
    results = db_module.search_persons(query, limit)
    return results

@mcp.tool()
def get_document(doc_id: str) -> dict:
    result = db_module.get_document(doc_id)
    if not result:
        return {"error": f"Document '{doc_id}' not found."}
    return result

@mcp.tool()
def get_dossier(dossier_id: str) -> list[dict]:
    results = db_module.get_dossier(dossier_id)
    if not results:
        return [{"error": f"Dossier '{dossier_id}' not found."}]
    return results

@mcp.tool()
def search_text(query: str, limit: int = 20) -> list[dict]:
    limit = min(limit, 100)
    return db_module.search_text(query, limit)

@mcp.tool()
def get_persons_in_year_range(year_from: int, year_to: int, limit: int = 100) -> list[dict]:
    if year_to < year_from:
        return [{"error": "year_to must be >= year_from"}]
    if year_to - year_from > 300:
        return [{"error": "Year range too large; max 300 years."}]
    limit = min(limit, 500)
    return db_module.get_persons_in_year_range(year_from, year_to, limit)

@mcp.tool()
def get_cooccurrences(person_name: str, limit: int = 20) -> list[dict]:
    return db_module.get_cooccurrences(person_name, min(limit, 100))

@mcp.tool()
def list_dossiers(limit: int = 200) -> list[dict]:
    return db_module.list_dossiers(min(limit, 2000))

# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("hgb://stats")
def resource_stats() -> str:
    return json.dumps(db_module.db_stats(), indent=2)

@mcp.resource("hgb://dossiers")
def resource_dossiers() -> str:
    return json.dumps(db_module.list_dossiers(9999), indent=2)

@mcp.resource("hgb://document/{doc_id}")
def resource_document(doc_id: str) -> str:
    return json.dumps(db_module.get_document(doc_id), indent=2, ensure_ascii=False)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(f"Database: {args.db}")
    try:
        stats = db_module.db_stats()
        logger.info(f"Corpus: {stats['n_documents']:,} docs, {stats['n_persons']:,} person spans")
    except Exception as e:
        logger.warning(f"Could not read DB stats: {e}")

    logger.info(f"Starting EOS MCP server on {args.host}:{args.port}")
    mcp.run(transport="sse")
