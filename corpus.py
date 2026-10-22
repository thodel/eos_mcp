#!/usr/bin/env python3
"""
corpus.py — what makes this repo the Historisches Grundbuch Basel rather than
one of the others.

embeddings.py is vendored byte-identical across the fleet (see ch-h-bot
docs/MCP_PROVIDER_PATTERN.md); whatever is particular to this corpus lives
here instead, so that the shared file can stay a checksum.

embed_db.py and vectorstore.py are shared the same way (since #6). What they
ask of a corpus is at the bottom of this file: where the database is, which
table the passages belong to, how to iterate it, and how to clean a text.
"""

from __future__ import annotations

import os
import re
import unicodedata

# The shared embeddings.py reads EOS_EMBED_MODEL, EOS_CHUNK_CHARS, ... only
# if it knows the prefix. Set before it is imported; the Dockerfile sets it too.
os.environ.setdefault("MCP_ENV_PREFIX", "EOS")

import embeddings as emb  # noqa: E402

# What the shared embed_db.py reads: where the corpus is and what a passage
# belongs to. server.py has the same default for the same reason.
DB_ENV = "EOS_DB"
DEFAULT_DB = "/data/hgb.db"
PARENT_TABLE = "documents"

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


# ── What the shared embed_db.py asks of this corpus ─────────────────────────
#
# iter_documents yields (id, head, text); embed_db.py cleans and windows the
# text and puts the head, verbatim, on the first line of every passage. The
# vocabulary for clean() is corpus-wide state, built by iter_documents before
# its first document — not per entry, and not over the sample a --limit asks
# for: deciding whether "Hein" is a fragment or a name needs evidence from
# everywhere the token occurs, and that is how the index on idhefix was built
# and how ch-h-bot#555 measured all 76,737 passages row-identical.

_VOCAB: dict | None = None


def head_line(dossier_id, year) -> str:
    """Die Kopfzeile jeder Passage: Dossier und Jahr, sonst nichts.

    Das HGB hat keine Titel; ein Eintrag wird über sein Dossier und sein Jahr
    eingeordnet, und «der Hof zu Lind» liest sich 1360 wie 1644. Wörtlich,
    ohne strip(): die Vektoren sind aus genau dieser Zeile gemacht, und ein
    Dossier mit Leerzeichen am Rand ergäbe sonst eine andere (ch-h-bot#555:
    ein strip() kostete hls 11 Passagen). Leer, wenn beides fehlt — dann trägt
    die Passage keinen Kopf. Eine Funktion, damit db.passage_body genau den
    Kopf abschneidet, den embed_db.py schreibt.
    """
    return " ".join(str(x) for x in (dossier_id, year) if x)


def iter_documents(conn, limit=None):
    """Yield (id, head, text_raw) for every document that has a text.

    Builds the expansion vocabulary over the WHOLE table first, whatever
    ``limit`` says (see above); the shared build_chunks calls this before its
    first clean(), so the order holds.
    """
    global _VOCAB
    _VOCAB = build_vocab(
        t for (t,) in conn.execute(
            "SELECT text_raw FROM documents WHERE text_raw IS NOT NULL"))
    sql = ("SELECT id, dossier_id, year, text_raw FROM documents "
           "WHERE text_raw IS NOT NULL AND text_raw <> '' ORDER BY id")
    rows = conn.execute(sql + (" LIMIT ?" if limit else ""),
                        (limit,) if limit else ()).fetchall()
    for doc_id, dossier_id, year, text in rows:
        yield doc_id, head_line(dossier_id, year), text


def clean(text: str) -> str:
    """The expanded reading, with the vocabulary iter_documents built.

    Refuses rather than guessing: with an empty vocabulary nothing would ever
    be joined, and the passages would differ from the index without a word.
    """
    if _VOCAB is None:
        raise RuntimeError("corpus.clean needs the corpus vocabulary; "
                           "iter_documents builds it before its first document")
    return clean_entry_text(text, _VOCAB)
