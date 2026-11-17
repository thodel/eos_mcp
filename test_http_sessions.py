"""The HTTP endpoint keeps no session per call (stateless streamable HTTP).

The Swiss History Bot (ch-h-bot api/mcp_client.py) opens one MCP session per
tool call and never closes it. A stateful server keeps every such session
until restart; on hsso_mcp that was about 74 KB per call and, by 16.11.2026,
enough to push the unit past MemoryHigh (hsso_mcp PR #2). These tests drive
the real ASGI app in the bot's pattern (initialize, notifications/initialized,
a request, no DELETE) and count what the session manager still holds.

The counter-test sends the same traffic to the stateful app and must find one
session per call, so the first test cannot pass by counting the wrong thing.
The last test checks that main() really hands the flag to the transport.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

starlette_testclient = pytest.importorskip("starlette.testclient")

import server  # noqa: E402

CALLS = 40
EXPECTED_TOOL = "get_document"
HEADERS = {"Content-Type": "application/json",
           "Accept": "application/json, text/event-stream"}


def _bot_call(client) -> dict:
    """One round as ch-h-bot's api/mcp_client.py makes it: three POSTs, no DELETE.

    The third request is tools/list rather than tools/call so the test needs no
    corpus database; the session is opened at initialize either way, which is
    what the bot leaks.
    """
    init = client.post("/mcp", headers=HEADERS, json={
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "1"}}})
    assert init.status_code == 200, init.text
    headers = dict(HEADERS)
    sid = init.headers.get("mcp-session-id")
    if sid:
        headers["Mcp-Session-Id"] = sid
    note = client.post("/mcp", headers=headers,
                       json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    assert note.status_code == 202, note.text
    resp = client.post("/mcp", headers=headers, json={
        "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert resp.status_code == 200, resp.text
    return {"session_id": sid, "body": resp.text}


def _held_sessions_after(app) -> tuple[int, list[dict]]:
    # base_url on loopback: the app enables DNS-rebinding protection for
    # 127.0.0.1 and would answer 421 to TestClient's default Host «testserver».
    with starlette_testclient.TestClient(app, base_url="http://127.0.0.1:8000") as client:
        calls = [_bot_call(client) for _ in range(CALLS)]
        # The manager's own registry of live sessions, the one that grows.
        held = len(server.mcp.session_manager._server_instances)
    return held, calls


def _production_app():
    return server.mcp.streamable_http_app(streamable_http_path="/mcp", host="127.0.0.1",
                                          stateless_http=server.STATELESS_HTTP)


def test_bot_traffic_leaves_no_session_behind():
    assert server.STATELESS_HTTP is True
    held, calls = _held_sessions_after(_production_app())
    assert all(EXPECTED_TOOL in c["body"] for c in calls)  # the server really answered
    assert all(c["session_id"] is None for c in calls)     # no session was ever issued
    assert held == 0


def test_counter_stateful_server_keeps_one_session_per_call():
    app = server.mcp.streamable_http_app(streamable_http_path="/mcp", host="127.0.0.1",
                                         stateless_http=False)
    held, calls = _held_sessions_after(app)
    assert all(c["session_id"] for c in calls)
    assert held == CALLS


def test_main_starts_the_transport_stateless(tmp_path):
    """main() passes STATELESS_HTTP on to mcp.run, not just defines it.

    Runs in a subprocess so main()'s set_db_path and start-up checks touch an
    empty database in tmp_path and leave this process's state alone.
    """
    code = (
        "import json, server\n"
        "seen = {}\n"
        "server.mcp.run = lambda **kw: seen.update(kw)\n"
        # An empty database has no vectors; warming would raise, not report.
        "if hasattr(server.db_module, 'warm_semantic_index'):\n"
        "    server.db_module.warm_semantic_index = lambda *a, **k: {'reason': 'test'}\n"
        f"server.main(['--db', {str(tmp_path / 'empty.db')!r}, '--port', '18999'])\n"
        "print('RUN ' + json.dumps({k: v for k, v in seen.items()\n"
        "                           if isinstance(v, (str, int, bool))}))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    line = next(l for l in out.stdout.splitlines() if l.startswith("RUN "))
    kwargs = json.loads(line[4:])
    assert kwargs["transport"] == "streamable-http"
    assert kwargs["stateless_http"] is True
