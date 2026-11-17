#!/usr/bin/env python3
"""Drive a running EOS server the way the bot does and sample its memory.

The Swiss History Bot (ch-h-bot ``api/mcp_client.py``) makes every tool call
in a fresh MCP session — ``initialize``, ``notifications/initialized``,
``tools/call`` — and never sends the closing ``DELETE``. A stateful
streamable-HTTP server keeps each of those sessions until it is restarted;
on 16.11.2026 about 16,500 of them had pushed ``mcp-hsso`` into its
``MemoryHigh`` (ch-h-bot#584, hsso_mcp PR #2), and the other stateful
providers, ``mcp-eos`` among them, collect them the same way. This script repeats that pattern N times
against a server and prints the resident memory of its process after every
``--every`` calls, so the growth per call can be read off, before and after a
fix.

Standard library only, so it runs with any python3 on the box:

    python3 eval/session_memory.py --url http://127.0.0.1:18000/mcp \\
        --pid 12345 --calls 3000 --every 500

The default tool is ``corpus_stats``: it reads SQLite and never calls the
embedding endpoint, so the measurement puts no load on GPUStack. A get_*
tool with a known id (``--tool``, ``--arguments``) works as well.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HEADERS = {"Content-Type": "application/json",
           "Accept": "application/json, text/event-stream"}


def post(url: str, body: dict, session_id: str | None) -> tuple[int, str | None, bytes]:
    headers = dict(HEADERS)
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, resp.headers.get("mcp-session-id"), resp.read()


def one_call(url: str, tool: str, arguments: dict) -> bool:
    """One tool call in the bot's pattern: three POSTs, no DELETE."""
    status, sid, _ = post(url, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                           "clientInfo": {"name": "session-memory", "version": "1"}}},
                          None)
    post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    status, _, body = post(url, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                 "params": {"name": tool, "arguments": arguments}}, sid)
    return status == 200 and b'"error"' not in body[:200]


def rss_mb(pid: int) -> float:
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError(f"no VmRSS for pid {pid}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--pid", type=int, required=True, help="server process whose VmRSS is sampled")
    ap.add_argument("--calls", type=int, default=3000)
    ap.add_argument("--every", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--tool", default="corpus_stats")
    ap.add_argument("--arguments", default='{}')
    args = ap.parse_args(argv)
    arguments = json.loads(args.arguments)

    print(f"{'calls':>6} {'rss_mb':>8} {'delta_mb':>9} {'kb_per_call':>11} {'failed':>6} {'sec':>6}")
    start_rss = rss_mb(args.pid)
    print(f"{0:>6} {start_rss:>8.1f} {0:>9.1f} {'':>11} {0:>6} {0:>6}")
    done = failed = 0
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        while done < args.calls:
            batch = min(args.every, args.calls - done)
            results = list(pool.map(lambda _: one_call(args.url, args.tool, arguments), range(batch)))
            done += batch
            failed += results.count(False)
            time.sleep(1.0)            # let the server finish what the last responses started
            rss = rss_mb(args.pid)
            delta = rss - start_rss
            print(f"{done:>6} {rss:>8.1f} {delta:>9.1f} {delta * 1024 / done:>11.1f} "
                  f"{failed:>6} {time.monotonic() - t0:>6.0f}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
