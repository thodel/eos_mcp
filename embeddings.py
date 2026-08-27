#!/usr/bin/env python3
"""
embeddings.py — the embedding layer shared by the pipeline and the server.

The MCP server is where an institution's embeddings live: a client asks a
question in natural language and the server answers with the passages that mean
the same thing, rather than the ones that happen to share keywords.

For the Historisches Grundbuch Basel the gap is at its widest. The corpus is
75,447 land-register entries, 1400–1700, in tokenised medieval German with high
edition noise: a user asking "who owned the houses by the Rhine?" in modern
German shares almost no surface forms with the scribe's spelling. Measured
before this was built: a natural-language question returned zero keyword hits
against this corpus.

Vectors come from GPUStack (`qwen3-embedding-0.6b`, 1024 dimensions), are
L2-normalised on the way in, and are stored as float32 BLOBs in SQLite. The
corpus is 75,447 documents, so similarity is a brute-force dot product over a
numpy array — no index, nothing to tune, exact results.

Mirrors kf_mcp/embeddings.py; the corpus-specific part is the transcription
cleanup below.
"""

from __future__ import annotations

import os
import re
import struct
import unicodedata
from typing import Iterable, Optional

DEFAULT_MODEL = os.environ.get("EOS_EMBED_MODEL", "qwen3-embedding-0.6b")
DEFAULT_BASE_URL = os.environ.get("GPUSTACK_BASE_URL", "https://gpustack.unibe.ch/v1")
DEFAULT_BATCH = int(os.environ.get("EOS_EMBED_BATCH", "64"))

# Passage windowing. 1000 characters is roughly a paragraph of an HLS article —
# small enough that a hit points at the relevant passage, large enough to carry
# its own context into an answer.
CHUNK_CHARS = int(os.environ.get("KF_CHUNK_CHARS", "1000"))
CHUNK_OVERLAP = int(os.environ.get("KF_CHUNK_OVERLAP", "150"))

# Qwen3-Embedding is instruction-aware: queries carry a task instruction and
# documents do not. Skipping this costs real retrieval quality, so it is applied
# by default and can be overridden for a differently-trained model.
QUERY_PREFIX = os.environ.get(
    "EOS_EMBED_QUERY_PREFIX",
    "Instruct: Given a question about property, families and legal acts in "
    "early modern Basel, retrieve passages from land-register entries, "
    "15th to 18th century.\nQuery: ",
)


class EmbeddingError(RuntimeError):
    """Raised when embeddings are requested but cannot be produced."""


# ── Chunking ─────────────────────────────────────────────────────────────────

# ── Transcription cleanup: resolve abbreviations to their expanded reading ────
#
# The edition transcribes diplomatically. Segment boundaries are marked with ✳
# (82,062 of them), and every abbreviated word is written as the raw manuscript
# form immediately followed by the editor's expansion:
#
#     "un̄ und"           the raw form is a whole word            -> und
#     "Hein r₎ rich"      the raw form continues a fragment       -> Heinrich
#     "stif tˀin terin"                                           -> stifterin
#     "Diz ist dˀ der"                                            -> Diz ist der
#
# Embedded as-is, this is close to unusable: names arrive split in half, every
# abbreviated word is doubled, and 82k segment markers punctuate the text at
# random. Embedding the expanded reading is the decision this implements.

SEGMENT_MARKER = "✳"
# Expansion bracket, abbreviation hook, combining macron — the three ways this
# edition marks a raw form.
_MARKER_RE = re.compile("[₎ˀ\u0304]")
_PUNCT = ",.;:()[]«»\"\'"


def has_marker(token: str) -> bool:
    """True when a token is a raw manuscript form rather than a reading."""
    return bool(_MARKER_RE.search(token))


def build_vocab(texts) -> dict:
    """For each token, how often it is a fragment versus a word in its own right.

    Deciding whether an expansion joins the token before it needs to know
    whether that token is already a word: "ist" is, "Hein" and "stif" are not.

    Counting total occurrences does not separate them — "Hein" recurs
    constantly, because "Heinrich" is abbreviated the same way every time — and
    neither does a rare/common threshold, because "Hein" is also a name on its
    own. What separates them is the *ratio*: "ist" appears overwhelmingly in
    ordinary positions, "Hein" almost only before a raw form. The corpus
    answers the question itself, which beats a word list for a language with no
    settled orthography.

    Returns ``{token: (n_fragment, n_solo)}``.
    """
    counts: dict[str, list[int]] = {}
    for text in texts:
        tokens = (text or "").replace(SEGMENT_MARKER, " ").split()
        for i, token in enumerate(tokens):
            if has_marker(token):
                continue
            bare = token.strip(_PUNCT).lower()
            if not bare:
                continue
            slot = counts.setdefault(bare, [0, 0])
            followed = i + 1 < len(tokens) and has_marker(tokens[i + 1])
            slot[0 if followed else 1] += 1
    return {k: tuple(v) for k, v in counts.items()}


def _is_wordlike(token: str) -> bool:
    """Alphabetic, tolerating this edition's combining diacritics.

    ``str.isalpha()`` is False for a combining mark, so "Oͤs" (O + U+0364 + s)
    would otherwise never be recognised as a word fragment.
    """
    return bool(token) and all(
        c.isalpha() or unicodedata.combining(c) for c in token)


def clean_entry_text(text: str, vocab: dict) -> str:
    """Transcription reduced to its expanded reading (see the note above)."""
    tokens = (text or "").replace(SEGMENT_MARKER, " ").split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if has_marker(token):
            if i + 1 < len(tokens):
                expansion = tokens[i + 1]
                previous = out[-1] if out else ""
                bare = previous.strip(_PUNCT).lower()
                fragment, solo = vocab.get(bare, (0, 0))
                if _is_wordlike(bare) and fragment > solo:
                    out[-1] = previous + expansion
                else:
                    out.append(expansion)
                i += 2
            else:
                i += 1        # trailing raw form with no expansion after it
            continue
        out.append(token)
        i += 1
    return re.sub(r"\s+([,.;:])", r"\1", " ".join(out)).strip()


def chunk_article(text: str, size: int = CHUNK_CHARS,
                  overlap: int = CHUNK_OVERLAP) -> list[tuple[int, int, str]]:
    """Window an article into ``(char_start, char_end, text)`` passages.

    Offsets are into the *cleaned* text, so a hit can be located in the article
    it came from. Windows break at a paragraph or sentence boundary when one
    falls in the back half, so passages do not end mid-clause.
    """
    text = (text or "").strip()
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
    """Unpack a stored vector (mostly for tests and inspection)."""
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
        raise EmbeddingError(
            "the `openai` package is required for embeddings — pip install openai"
        ) from exc
    key = api_key or os.environ.get("GPUSTACK_API_KEY")
    if not key:
        raise EmbeddingError(
            "GPUSTACK_API_KEY is not set; semantic search needs it to embed the query")
    client = OpenAI(base_url=base_url or DEFAULT_BASE_URL, api_key=key)
    if base_url is None and api_key is None:
        _client = client
    return client


def embed_texts(texts: list[str], model: str = DEFAULT_MODEL,
                client=None) -> list[list[float]]:
    """Embed a batch of texts. Returns raw (un-normalised) vectors."""
    if not texts:
        return []
    client = client or get_client()
    try:
        response = client.embeddings.create(model=model, input=texts)
    except Exception as exc:
        detail = str(exc)
        if "403" in detail:
            detail += ("  — gpustack.unibe.ch denies requests from outside the "
                       "UniBE network before checking the API key; connect the VPN.")
        raise EmbeddingError(f"embedding request failed: {detail}") from exc
    return [item.embedding for item in response.data]


def embed_query(query: str, model: str = DEFAULT_MODEL, client=None) -> list[float]:
    """Embed one search query, with the instruction prefix applied."""
    vectors = embed_texts([QUERY_PREFIX + query], model=model, client=client)
    if not vectors:
        raise EmbeddingError("the embedding service returned no vector for the query")
    return vectors[0]
