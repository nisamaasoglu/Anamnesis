"""Field-weighted BM25 with light stemming and domain query expansion.

The offline hashing vectoriser matches exact surface words only, so "visit"
misses "visiting", "ICU" misses "Intensive Care Unit", and a section whose body
happens to repeat a query word can outrank the section whose heading names the
topic. This module fixes those three failure modes without any model, network
or new dependency:

* a small suffix stemmer folds inflected forms onto one stem;
* a reviewed list of hospital abbreviations and lay phrasings expands the
  query (never the documents), so expansion cannot change what is indexed;
* BM25 scores the section heading with a higher field weight than the body,
  because a heading is the author's own statement of what a passage is about.

Everything is deterministic and runs in microseconds on this corpus.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .embeddings import _STOPWORDS

_WORD_RE = re.compile(r"[a-z0-9]+")

_PHRASES: dict[str, str] = {
    "a&e": "emergency department",
    "a & e": "emergency department",
    "emergency room": "emergency department",
    "go home": "discharge",
    "leave hospital": "discharge",
    "sent home": "discharge",
    "seen first": "triage priority",
    "end of life": "end-of-life compassionate",
    "blood test": "laboratory test",
    "wash their hands": "hand hygiene",
    "wash my hands": "hand hygiene",
    "wash hands": "hand hygiene",
    "health information": "confidentiality personal information",
    "medical records": "records access",
}

_WORDS: dict[str, str] = {
    "icu": "intensive care unit",
    "er": "emergency department",
    "casualty": "emergency department",
    "lab": "laboratory",
    "labs": "laboratory",
    "meds": "medication pharmacy",
    "medicine": "medication pharmacy",
    "medicines": "medication pharmacy",
    "drugs": "medication pharmacy",
    "prescription": "pharmacy medication",
    "operation": "procedure surgery",
    "surgery": "procedure",
    "moved": "transfer",
    "move": "transfer reschedule",
    "complain": "complaints",
    "complaint": "complaints",
    "gloves": "personal protective equipment",
    "gowns": "personal protective equipment",
    "masks": "personal protective equipment",
    "apron": "personal protective equipment",
    "ppe": "personal protective equipment",
    "dying": "end-of-life compassionate",
    "palliative": "end-of-life compassionate",
    "worse": "deteriorate escalation concern",
    "deteriorates": "escalation concern",
    "child": "paediatric minor",
    "children": "paediatric minor",
    "kid": "paediatric minor",
    "kids": "paediatric minor",
    "alone": "unaccompanied",
    "pediatric": "paediatric",
    "translator": "interpreter",
    "text": "reminder",
    "sms": "reminder",
    "infected": "infection isolation",
    "norovirus": "outbreak",
    "flu": "outbreak",
    "partner": "partners",
}


def stem(word: str) -> str:
    """Strip common English inflections so related forms share one key."""
    if len(word) <= 3:
        return word
    suffixes = (("sses", "ss"), ("ies", "y"), ("ings", ""), ("ing", ""), ("edly", ""), ("ed", ""))
    for suffix, repl in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: len(word) - len(suffix)] + repl
            break
    if word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        word = word[:-1]
    if word.endswith("or") and len(word) > 6:
        word = word[:-2]
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def terms(text: str) -> list[str]:
    """Stemmed content words of a passage (no expansion)."""
    return [stem(w) for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def expand_query(question: str) -> list[str]:
    """Stemmed query terms plus their reviewed abbreviation and lay expansions."""
    lowered = question.lower()
    extra: list[str] = []
    for phrase, replacement in _PHRASES.items():
        if phrase in lowered:
            extra.append(replacement)
    for word in _WORD_RE.findall(lowered):
        if word in _WORDS:
            extra.append(_WORDS[word])
    base = terms(lowered)
    expanded = terms(" ".join(extra))
    return base + [t for t in expanded if t not in base]


@dataclass(frozen=True)
class Document:
    heading: str
    title: str
    body: str


class BM25:
    """BM25F-style scorer: per-field term frequencies combined with field weights."""

    def __init__(
        self,
        documents: Sequence[Document],
        heading_weight: float = 3.0,
        title_weight: float = 1.0,
        body_weight: float = 1.0,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.weighted: list[Counter[str]] = []
        self.heading_terms: list[frozenset[str]] = []
        for doc in documents:
            self.heading_terms.append(frozenset(terms(doc.heading)))
            tf: Counter[str] = Counter()
            for term in terms(doc.heading):
                tf[term] += heading_weight
            for term in terms(doc.title):
                tf[term] += title_weight
            for term in terms(doc.body):
                tf[term] += body_weight
            self.weighted.append(tf)
        self.lengths = [sum(tf.values()) for tf in self.weighted]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self.weighted:
            df.update(tf.keys())
        n = len(self.weighted)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def is_evidence(self, index: int, query_terms: Sequence[str]) -> bool:
        """A passage counts as evidence only with real overlap, not one stray word.

        It must either share a term with the query in its heading, or share at
        least two distinct terms anywhere. One incidental body word ("best" in
        "best interests", "exposure" in "exposure to body fluids") is exactly
        the kind of match that would let an off-topic question borrow a citation.
        """
        unique = set(query_terms)
        if unique & self.heading_terms[index]:
            return True
        return len(unique & self.weighted[index].keys()) >= 2

    def scores(self, query_terms: Sequence[str]) -> list[float]:
        unique = list(dict.fromkeys(query_terms))
        out: list[float] = []
        for tf, length in zip(self.weighted, self.lengths, strict=True):
            ratio = length / self.avg_len if self.avg_len else 1.0
            norm = self.k1 * (1 - self.b + self.b * ratio)
            score = 0.0
            for term in unique:
                freq = tf.get(term, 0.0)
                if freq:
                    score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / (freq + norm)
            out.append(score)
        return out
