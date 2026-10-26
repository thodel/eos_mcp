#!/usr/bin/env python3
"""
embeddings.py — the embedding layer every MCP server in this fleet shares.

VENDORED FILE. The canonical copy lives in hls_mcp; the copies in kf_mcp,
dodis_mcp and infoclio_mcp are meant to be byte-identical, so that a drift check
is a checksum rather than a diff review. Corpus-specific behaviour goes in the
calling repo, never in here — that is what keeps them identical.

The MCP server is where an institution's embeddings belong: a client asks a
question and the server answers with the passages that mean the same thing,
rather than the ones that happen to share keywords. A client cannot embed a
corpus it never receives, so this cannot live on the client side.

Vectors come from GPUStack (`qwen3-embedding-4b`, 2560 dimensions), are
L2-normalised on the way in, and are stored as float32 BLOBs in SQLite. At the
scale of these corpora — tens of thousands of passages — similarity is an honest
brute-force dot product over a numpy array: exact results, no index, and no
recall parameter to get wrong.

The fleet ran on `qwen3-embedding-0.6b` (1024 dimensions) until GPUStack
withdrew it on 24.10.2026; every provider was re-embedded with 4b on 25./26.10.
(ch-h-bot#570). A query vector and the stored vectors must come from the same
model — the model name selects both, here and in `vectorstore.default_model`.

Per-repo configuration is the ENV_PREFIX below and the `clean` callable passed
to `chunk_document`; nothing else should differ.
"""

from __future__ import annotations

import time

import sys

import os
import re
import struct
from typing import Callable, Iterable, Optional

# Set by each repo: "HLS", "KF", "DODIS", "INFOCLIO". Everything else is shared.
ENV_PREFIX = os.environ.get("MCP_ENV_PREFIX", "MCP")


def _env(name: str, default: str) -> str:
    """Read <PREFIX>_<NAME>, falling back to MCP_<NAME> and then the default."""
    return os.environ.get(f"{ENV_PREFIX}_{name}",
                          os.environ.get(f"MCP_{name}", default))


DEFAULT_MODEL = _env("EMBED_MODEL", "qwen3-embedding-4b")
DEFAULT_BASE_URL = os.environ.get("GPUSTACK_BASE_URL", "https://gpustack.unibe.ch/v1")
DEFAULT_BATCH = int(_env("EMBED_BATCH", "64"))

CHUNK_CHARS = int(_env("CHUNK_CHARS", "1000"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "150"))

# Qwen3-Embedding is instruction-aware: queries carry a task instruction and
# documents do not. Skipping this costs real retrieval quality.
QUERY_PREFIX = _env(
    "EMBED_QUERY_PREFIX",
    "Instruct: Given a question about Swiss history, retrieve passages that "
    "answer it\nQuery: ",
)


class EmbeddingError(RuntimeError):
    """Raised when embeddings are requested but cannot be produced."""


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_document(text: str, size: int = CHUNK_CHARS,
                   overlap: int = CHUNK_OVERLAP,
                   clean: Optional[Callable[[str], str]] = None
                   ) -> list[tuple[int, int, str]]:
    """Window a document into ``(char_start, char_end, text)`` passages.

    ``clean`` is the one corpus-specific hook: HLS strips a byline, Königsfelden
    resolves abbreviations to their expanded reading, dodis and infoclio need
    nothing. Offsets are into the *cleaned* text, so a hit can be located in the
    document it came from.

    Windows break at a paragraph or sentence boundary when one falls in the back
    half, so passages do not end mid-clause.
    """
    text = (clean(text) if clean else (text or "")).strip()
    if not text:
        return []
    if len(text) <= size:
        return [(0, len(text), text)]

    out: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            for marker in ("\n\n", ". ", "\n"):
                cut = window.rfind(marker)
                if cut > size * 0.5:
                    end = start + cut + len(marker)
                    break
        piece = text[start:end].strip()
        if piece:
            out.append((start, end, piece))
        if end >= len(text) or end <= start:
            break
        nxt = end - overlap
        start = nxt if nxt > start else end
    return out


# ── Vector storage ───────────────────────────────────────────────────────────

def pack(vector: Iterable[float]) -> bytes:
    """L2-normalise and pack a vector as little-endian float32.

    Normalising once at write time makes every later similarity a plain dot
    product, which is what keeps brute-force search cheap.
    """
    values = [float(v) for v in vector]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return struct.pack(f"<{len(values)}f", *(v / norm for v in values))


def unpack(blob: bytes) -> list[float]:
    """Unpack a stored vector (for tests and inspection)."""
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


# ── GPUStack client ──────────────────────────────────────────────────────────

_client = None


def get_client(base_url: Optional[str] = None, api_key: Optional[str] = None):
    """OpenAI-compatible GPUStack client, created once.

    GPUStack is reachable only from inside the UniBE network; from anywhere else
    it answers 403 *before* checking the key, so a 403 means the wrong network
    rather than a bad credential. The message says so, because the alternative
    is an afternoon spent rotating a key that was never the problem.
    """
    global _client
    if _client is not None and base_url is None and api_key is None:
        return _client
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise EmbeddingError("the `openai` package is required — pip install openai") from exc
    key = api_key or os.environ.get("GPUSTACK_API_KEY")
    if not key:
        raise EmbeddingError(
            "GPUSTACK_API_KEY is not set; semantic search needs it to embed the query")
    client = OpenAI(base_url=base_url or DEFAULT_BASE_URL, api_key=key)
    if base_url is None and api_key is None:
        _client = client
    return client


# A transient failure mid-run — GPUStack dropping one connection, a timeout
# — used to end a forty-minute embedding pass at chunk 306,000 of 647,000
# (2026-09-10). These waits, in seconds, sit between attempts; 401/403 are
# not transient and fail at once.
RETRY_WAITS = (5, 15, 45, 90, 180)


def embed_texts(texts: list[str], model: str = DEFAULT_MODEL,
                client=None, waits=RETRY_WAITS) -> list[list[float]]:
    """Embed a batch of texts. Returns raw (un-normalised) vectors."""
    if not texts:
        return []
    client = client or get_client()
    attempt = 0
    while True:
        try:
            response = client.embeddings.create(model=model, input=texts)
            return [item.embedding for item in response.data]
        except Exception as exc:
            detail = str(exc)
            if "403" in detail:
                detail += ("  — gpustack.unibe.ch denies requests from outside the "
                           "UniBE network before checking the API key; connect the VPN.")
            fatal = "401" in detail or "403" in detail
            if fatal or attempt >= len(waits):
                raise EmbeddingError(f"embedding request failed: {detail}") from exc
            wait = waits[attempt]
            attempt += 1
            print(f"  embedding request failed ({detail[:80]}); retry {attempt} "
                  f"in {wait}s", file=sys.stderr)
            time.sleep(wait)


def embed_query(query: str, model: str = DEFAULT_MODEL, client=None) -> list[float]:
    """Embed one search query, with the instruction prefix applied."""
    vectors = embed_texts([QUERY_PREFIX + query], model=model, client=client)
    if not vectors:
        raise EmbeddingError("the embedding service returned no vector for the query")
    return vectors[0]
