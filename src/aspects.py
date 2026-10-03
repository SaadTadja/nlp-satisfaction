"""Aspect detection and sentence attribution (§11) — the "why" layer.

"62% satisfied" is not actionable. "62% satisfied; among the dissatisfied, 41%
mention delivery" is. This module provides the cheap, honest middle ground
between no aspect analysis and full ABSA (which needs aspect-annotated
training data this project does not have).

Method
------
1. Split each review into sentences.
2. Attribute an aspect to a sentence when it contains a lexicon term.
3. Score sentiment **per sentence**, not per review.

Step 3 is what makes this work on mixed feedback: "produit super mais
livraison catastrophique" yields positive-quality and negative-delivery
rather than one muddled label.

Known limitations -- state these in the model card rather than letting a
reader discover them:

* **No implicit aspects.** A complaint with no lexicon term is missed
  entirely. Measure the unattributed rate and report it.
* **No aspect-opinion dependency parsing.** A sentence mentioning two aspects
  is attributed to both, with the same sentiment.
* **Lexicons are hand-built and domain-specific.** They must be rebuilt from
  frequent terms in the target corpus, not reused blindly.

No torch dependency: pure Python so it is testable without a model.
"""

from __future__ import annotations

import re
from collections import Counter

__all__ = [
    "ASPECT_LEXICONS",
    "split_sentences",
    "detect_aspects",
    "attribute_sentences",
    "suggest_lexicon_terms",
]

# Starter lexicons. §11 step 1 says to REBUILD these from the most frequent
# nouns in your own negative reviews -- `suggest_lexicon_terms` below helps.
# Treat what follows as a scaffold, not a finished artifact.
ASPECT_LEXICONS: dict[str, set[str]] = {
    "livraison": {
        "livraison", "livré", "livrée", "colis", "expédition", "transporteur",
        "délai", "délais", "retard", "attente", "reçu", "acheminement",
    },
    "prix": {
        "prix", "tarif", "cher", "chère", "coût", "coûteux", "rapport",
        "qualité-prix", "facturation", "remboursement", "promotion", "arnaque",
    },
    "qualité": {
        "qualité", "solide", "fragile", "robuste", "finition", "matériau",
        "matériaux", "cassé", "cassée", "défectueux", "défectueuse", "abîmé",
        "usure", "durable",
    },
    "service_client": {
        "service", "sav", "conseiller", "conseillère", "support", "assistance",
        "réponse", "injoignable", "accueil", "personnel", "équipe", "contact",
    },
    "facilité_usage": {
        "facile", "simple", "intuitif", "intuitive", "compliqué", "compliquée",
        "pratique", "ergonomie", "notice", "installation", "configuration",
        "utilisation",
    },
    "conformité": {
        "conforme", "description", "annonce", "photo", "taille", "couleur",
        "correspond", "différent", "différente", "trompeur", "trompeuse",
    },
}

# Sentence boundary: terminal punctuation followed by whitespace, or a newline.
# Deliberately simple -- French abbreviations ("M.", "etc.") will occasionally
# cause an over-split. That is acceptable here: an over-split sentence is still
# scored, it just carries less context. Pulling in spaCy for this would add a
# ~50MB model dependency to the Space for a marginal gain.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")
_WORD = re.compile(r"\w+", re.UNICODE)


def split_sentences(text: str) -> list[str]:
    """Split French review text into sentences, dropping empties."""
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def detect_aspects(
    text: str, lexicons: dict[str, set[str]] | None = None
) -> set[str]:
    """Return the aspects mentioned in a piece of text.

    Matching is on lowercased whole words, so "cherche" does not match "cher".
    A text may mention several aspects, or none.
    """
    lex = ASPECT_LEXICONS if lexicons is None else lexicons
    words = _tokens(text)
    return {aspect for aspect, terms in lex.items() if words & terms}


def attribute_sentences(
    text: str, lexicons: dict[str, set[str]] | None = None
) -> list[tuple[str, set[str]]]:
    """Split ``text`` into (sentence, aspects) pairs.

    Sentences with no detected aspect are returned with an empty set rather
    than dropped -- the share of unattributed text is itself a number worth
    reporting (see the limitations above).
    """
    return [(s, detect_aspects(s, lexicons)) for s in split_sentences(text)]


def suggest_lexicon_terms(
    texts, top_k: int = 60, min_length: int = 4, stopwords: set[str] | None = None
) -> list[tuple[str, int]]:
    """Most frequent content words, to seed lexicon building (§11 step 1).

    Run this over your *negative* reviews specifically: the vocabulary of
    complaints is what you want aspects to cover. Group the output by hand
    into aspect buckets -- the grouping is a judgement call, not something to
    automate away.
    """
    stop = _FRENCH_STOPWORDS if stopwords is None else stopwords
    counts: Counter[str] = Counter()
    for t in texts:
        counts.update(
            w for w in _tokens(t) if len(w) >= min_length and w not in stop
        )
    return counts.most_common(top_k)


# Minimal French stopword list -- enough to make `suggest_lexicon_terms`
# useful without pulling in NLTK for one list.
_FRENCH_STOPWORDS = {
    "alors", "aucun", "aussi", "autre", "avec", "avoir", "bien", "cela",
    "cette", "ceux", "chaque", "comme", "dans", "depuis", "deux", "donc",
    "dont", "elle", "elles", "encore", "entre", "etre", "être", "faire",
    "fait", "faut", "hors", "ils", "juste", "leur", "leurs", "mais", "meme",
    "même", "moins", "nous", "parce", "pour", "pourquoi", "quand", "quel",
    "quelle", "quelles", "quels", "qui", "quoi", "sans", "selon", "seulement",
    "sont", "sous", "suis", "sur", "tous", "tout", "toute", "toutes", "tres",
    "très", "trop", "vers", "votre", "vous", "étaient", "était", "étais",
    "cest", "jai", "nest", "plus", "peu", "tant", "déjà", "deja",
}
