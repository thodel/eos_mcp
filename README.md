# HGB Basel — MCP Server

An [MCP](https://modelcontextprotocol.io) server that exposes the Historisches Grundbuch Basel (HGB) corpus for use with Claude and other MCP-compatible clients.

## Architecture

```
hgb_full_*.xml  ──► build_db.py ──► hgb.db (SQLite + FTS5)
                                         │
                                    server.py  (mcp 2.0 MCPServer,
                                                streamable HTTP)
                                         │
                              http://<host>:8000/mcp
```

The 800 MB XML is parsed once into a ~100 MB SQLite database. The server then runs stateless queries against it (`PRAGMA query_only`).

The server targets **mcp 2.0**, which removed `mcp.server.fastmcp` — the high-level class is now `MCPServer` in `mcp.server.mcpserver`. `requirements.txt` pins the major version accordingly; the previous `mcp[cli]>=1.0.0` floor meant a rebuild silently jumped major versions and broke the import.

**The transport is streamable HTTP** (`/mcp`), not the HTTP+SSE (`/sse`) this server used previously. SSE is deprecated, and its handshake hands the client an absolute `/messages/` path computed from the app's own mount point — unreachable behind a reverse-proxy sub-path without rewriting the event stream in the proxy. **Existing clients pointed at `/sse` must be repointed.**

`db.SCHEMA_SQL` and `db.TRIGGERS_SQL` own the schema; `build_db.py` uses them, so the build and the tests cannot drift apart.

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

Each flag also has an environment variable — `EOS_DB`, `EOS_HOST`, `EOS_PORT`,
`EOS_HTTP_PATH` — which the flags override. Importing `server.py` never reads
`sys.argv`, so it is safe to import from tests or an ASGI loader.

`--http-path` (default `/mcp`) is the path the MCP endpoint is served at. **Behind a
reverse proxy, set it to the public path** — see [Reverse proxy](#reverse-proxy-nginx).

### 4. Connect a client

**Claude Code** — the name and URL are positional; there is no `--url` flag:

```bash
claude mcp add --transport http hgb http://<server-ip>:8000/mcp -s user
```

`-s user` makes the server available in every project; `-s project` writes it to
`.mcp.json` to share with a repository; the default `local` scope is just you, in the
current project. `claude mcp list` then reports the connection status.

**Claude Desktop, Cowork, claude.ai** — Customize → Connectors → **+** → *Add custom
connector*, and paste the same URL. These clients connect from Anthropic's cloud rather
than from your machine, so the server has to be reachable over the public internet;
`claude_desktop_config.json` only configures local stdio servers, not remote URLs.

**Project-scoped `.mcp.json`:**

```json
{
  "mcpServers": {
    "hgb": {
      "type": "http",
      "url": "http://<server-ip>:8000/mcp"
    }
  }
}
```

`type` is required, and `streamable-http` is accepted as an alias for `http`. An entry
with a `url` but no `type` is read as a stdio server and skipped with an error.

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

### Reverse proxy (nginx)

<a id="reverse-proxy-nginx"></a>

Serving under a sub-path (`https://tei.example.ch/mcp/eos/mcp`) has exactly one rule:
**the app's `--http-path` and the nginx `location` must be the same string.** The
endpoint is one path answering `POST`, `GET`, and `DELETE`; it builds no URLs of its
own, so nginx only has to forward the path unchanged.

```nginx
server {
    listen 443 ssl;
    server_name tei.example.ch;

    # EOS_HTTP_PATH=/mcp/eos/mcp — same string, no trailing slash on proxy_pass.
    location /mcp/eos/mcp {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host $host;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_set_header   Connection '';
        proxy_buffering    off;
        proxy_cache        off;
        proxy_read_timeout 3600s;
        chunked_transfer_encoding on;
    }
}
```

Under SSE this deployment needed a `sub_filter` rewriting `data: /messages/` in the
event stream, because the handshake advertised a path outside the proxy prefix.
Streamable HTTP removes that workaround entirely.

Two failure modes, both returning a bare `Not Found` or `405`:

- **A trailing slash on `proxy_pass`** strips the location prefix, so the app sees `/`.
- **`location` and `--http-path` disagree** — every request 404s. The startup line
  prints the path actually served:
  `Starting EOS MCP server on 0.0.0.0:8000/mcp/eos/mcp`.

> **Note:** the server has no authentication. By default `docker-compose.yml` publishes
> port 8000 on all interfaces; if a proxy fronts it, bind it to loopback instead:
>
> ```bash
> EOS_BIND=127.0.0.1 docker compose up -d
> ```

---

## Available tools

| Tool | Description |
|------|-------------|
| `corpus_stats` | Document/span/person/event counts and year range |
| `search_persons(query, limit)` | FTS search for person names (FTS5 syntax) |
| `get_document(doc_id)` | Full document: text, all spans, events |
| `get_dossier(dossier_id, limit=100)` | All documents for a property, ordered by year |
| `search_text(query, limit)` | Keyword search over raw transcriptions with snippets |
| `get_persons_in_year_range(year_from, year_to, limit)` | Person mentions filtered by year |
| `get_cooccurrences(person_name, limit)` | Other persons in the same documents |
| `list_dossiers(limit)` | All properties with coordinates and year ranges |

## Available resources

| URI | Description |
|-----|-------------|
| `hgb://stats` | Corpus statistics (JSON) |
| `hgb://dossiers` | Dossier index — `{total, returned, truncated, dossiers: [...]}`, capped at 1000 rows and flagged when truncated |
| `hgb://document/{doc_id}` | Single document (JSON) |

## Query behaviour

**Limits.** Every `limit` is clamped to at most 500; a negative, zero, or non-numeric
value falls back to that tool's own default. The previous `min(limit, 200)` guard let
every negative through, and SQLite reads `LIMIT -1` as unbounded.

**Full-text search.** `search_persons` and `search_text` pass the query to FTS5, so
operators work — `Hans OR Anna`, `Mei*`, `NEAR(...)`. An invalid FTS5 query falls back
to quoted phrases and then to a literal substring search instead of raising.

**Name search.** SQL wildcards are escaped, so `get_cooccurrences("100%")` looks for a
literal "100%" rather than matching every document.

**Co-occurrences.** `get_cooccurrences` caps the matching documents at
`COOC_DOC_CAP` (400). Their ids are bound one per placeholder, so a common name
matching thousands of documents would otherwise exceed SQLite's variable limit and
raise `too many SQL variables`.

**Result size.** Claude.ai and Claude Desktop truncate a tool or resource result at
roughly 150,000 characters. `get_dossier` is bounded because it carries every
document's full `text_raw`, and `get_document` caps spans and events at 2000 each.

## Tests

```bash
pip install pytest
pytest test_eos_mcp.py
```

Unit tests build their own throwaway database from `db.SCHEMA_SQL` and need no setup.
DB and server tests skip unless pointed at them:

```bash
EOS_DB=/data/hgb.db EOS_SERVER=http://localhost:8000 pytest test_eos_mcp.py
```
