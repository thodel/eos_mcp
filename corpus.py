#!/usr/bin/env python3
"""
corpus.py — what makes this repo the Historisches Grundbuch Basel rather than
one of the others.

embeddings.py is vendored byte-identical across the fleet (see ch-h-bot
docs/MCP_PROVIDER_PATTERN.md); whatever is particular to this corpus lives
here instead, so that the shared file can stay a checksum.

This repo is on the older layout: embed_db.py and the semantic search in db.py
are its own, not the shared ones. Its index stores `chunks.doc_id` (the
shared schema says `parent_id`) and puts a heading (title, year) on the first
line of every passage, which the shared embed_db.py cannot do. Moving to the shared files is
a schema migration plus a full re-embed, so it is a deployment, not a sync.
"""

from __future__ import annotations

import os
import re
import unicodedata

# The shared embeddings.py reads EOS_EMBED_MODEL, EOS_CHUNK_CHARS, ... only
# if it knows the prefix. Set before it is imported; the Dockerfile sets it too.
os.environ.setdefault("MCP_ENV_PREFIX", "EOS")

import embeddings as emb  # noqa: E402

# Qwen3-Embedding is instruction-aware: queries carry a task instruction and
# documents do not. This one names the corpus; the shared default only says
# "Swiss history".
QUERY_PREFIX = os.environ.get(
    "EOS_EMBED_QUERY_PREFIX",
    "Instruct: Given a question about property, families and legal acts in "
    "early modern Basel, retrieve passages from land-register entries, "
    "15th to 18th century.\nQuery: ",
)


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


def embed_query(query: str, model: str = emb.DEFAULT_MODEL, client=None) -> list[float]:
    """Embed one search query with this corpus's instruction prefix."""
    vectors = emb.embed_texts([QUERY_PREFIX + query], model=model, client=client)
    if not vectors:
        raise emb.EmbeddingError("the embedding service returned no vector for the query")
    return vectors[0]
