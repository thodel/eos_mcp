# HGB Basel — MCP Server

An [MCP](https://modelcontextprotocol.io) server that exposes the Historisches Grundbuch Basel (HGB) corpus for use with Claude and other MCP-compatible clients.

## Architecture

```
hgb_full_*.xml  ──► build_db.py ──► hgb.db (SQLite + FTS5)
                                         │
                                    server.py  (FastMCP / HTTP SSE)
                                         │
                              http://<host>:8000/sse
```

The 800 MB XML is parsed once into a ~100 MB SQLite database. The server then runs stateless queries against it.

## Setup

### 1. Install dependencies

```bash
cd mcp_server
pip install -r requirements.txt
```

### 2. Build the database

```bash
python build_db.py --xml ../hgb_full_26_05_29_05.xml --db hgb.db
```

This takes ~10 minutes and produces `hgb.db`. Run it once; repeat only when the XML changes.

### 3. Start the server

```bash
python server.py --db hgb.db --host 0.0.0.0 --port 8000
```

### 4. Connect a client

Add to your `claude_desktop_config.json` (or equivalent):

```json
{
  "mcpServers": {
    "hgb": {
      "url": "http://<server-ip>:8000/sse"
    }
  }
}
```

Or for Claude Code:
```bash
claude mcp add hgb --transport sse --url http://<server-ip>:8000/sse
```

---

## Docker deployment (recommended for the vServer)

### Build image

```bash
docker compose build
```

### First-time: build the database

```bash
# Copy XML to /data/hgb/ on the server, then:
docker run --rm \
  -v /data/hgb:/data \
  hgb-mcp \
  python build_db.py --xml /data/hgb_full_26_05_29_05.xml --db /data/hgb.db
```

### Run

```bash
docker compose up -d
```

Update `/data/hgb` in `docker-compose.yml` to match the actual path on the vServer.

### Reverse proxy (nginx, optional but recommended)

```nginx
server {
    listen 443 ssl;
    server_name hgb-mcp.example.unibe.ch;

    location / {
        proxy_pass         http://localhost:8000;
        proxy_http_version 1.1;
        # Required for SSE
        proxy_set_header   Connection '';
        proxy_buffering    off;
        proxy_cache        off;
        chunked_transfer_encoding on;
    }
}
```

---

## Available tools

| Tool | Description |
|------|-------------|
| `corpus_stats` | Document/span/person/event counts and year range |
| `search_persons(query, limit)` | FTS search for person names (FTS5 syntax) |
| `get_document(doc_id)` | Full document: text, all spans, events |
| `get_dossier(dossier_id)` | All documents for a property, ordered by year |
| `search_text(query, limit)` | Keyword search over raw transcriptions with snippets |
| `get_persons_in_year_range(year_from, year_to, limit)` | Person mentions filtered by year |
| `get_cooccurrences(person_name, limit)` | Other persons in the same documents |
| `list_dossiers(limit)` | All properties with coordinates and year ranges |

## Available resources

| URI | Description |
|-----|-------------|
| `hgb://stats` | Corpus statistics (JSON) |
| `hgb://dossiers` | All dossiers (JSON) |
| `hgb://document/{doc_id}` | Single document (JSON) |
