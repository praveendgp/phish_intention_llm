"""Retrieval module for the RAG layer.

Uses sentence-transformers embeddings when configured and available, otherwise
falls back to a dependency-free TF-IDF + cosine index so the project runs on a
machine that only has Ollama installed.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from ..config import Config, load_config
from ..utils.logging import get_logger
from .knowledge_base import KnowledgeBase, Passage, get_knowledge_base

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9']+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "that", "this", "with", "as", "at", "by", "it", "be", "from", "not", "but",
    "their", "its", "into", "than", "then", "so", "such", "can", "may", "will",
    "page", "user", "users", "site", "website", "pages",
}


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(str(text).lower())
            if t not in _STOP and len(t) > 1]


class TfidfIndex:
    def __init__(self, passages: Sequence[Passage]):
        self.passages = list(passages)
        self.docs = [_tokenize(p.text) for p in self.passages]
        self.df: Counter = Counter()
        for doc in self.docs:
            self.df.update(set(doc))
        self.n = max(1, len(self.docs))
        self.vectors = [self._vector(doc) for doc in self.docs]

    def _idf(self, term: str) -> float:
        return math.log((self.n + 1) / (self.df.get(term, 0) + 1)) + 1.0

    def _vector(self, tokens: Sequence[str]) -> Dict[str, float]:
        if not tokens:
            return {}
        tf = Counter(tokens)
        longest = max(tf.values())
        vec = {t: (0.5 + 0.5 * c / longest) * self._idf(t) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, top_k: int = 4,
               category: Optional[str] = None) -> List[Tuple[Passage, float]]:
        qvec = self._vector(_tokenize(query))
        if not qvec:
            return []
        scored: List[Tuple[Passage, float]] = []
        for passage, dvec in zip(self.passages, self.vectors):
            if category and passage.category not in (None, category):
                continue
            score = sum(w * dvec.get(t, 0.0) for t, w in qvec.items())
            if score > 0:
                scored.append((passage, round(score, 4)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


class EmbeddingIndex:  # pragma: no cover - optional dependency
    def __init__(self, passages: Sequence[Passage], model_name: str):
        from sentence_transformers import SentenceTransformer
        import numpy as np

        self.np = np
        self.passages = list(passages)
        self.model = SentenceTransformer(model_name)
        self.matrix = self.model.encode([p.text for p in self.passages],
                                        normalize_embeddings=True,
                                        show_progress_bar=False)

    def search(self, query: str, top_k: int = 4,
               category: Optional[str] = None) -> List[Tuple[Passage, float]]:
        qvec = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.matrix @ qvec
        out: List[Tuple[Passage, float]] = []
        for idx in self.np.argsort(-sims):
            passage = self.passages[int(idx)]
            if category and passage.category not in (None, category):
                continue
            out.append((passage, round(float(sims[int(idx)]), 4)))
            if len(out) >= top_k:
                break
        return out


class KnowledgeRetriever:
    """Facade used by the agents."""

    def __init__(self, config: Optional[Config] = None,
                 kb: Optional[KnowledgeBase] = None):
        self.cfg = config or load_config()
        self.kb = kb or get_knowledge_base()
        self.top_k = int(self.cfg.get("framework.retrieval_top_k", 4))
        self.backend = "tfidf"
        self.index = TfidfIndex(self.kb.passages)
        model_name = self.cfg.get("framework.embedding_model")
        if model_name:
            try:
                self.index = EmbeddingIndex(self.kb.passages, model_name)
                self.backend = f"embeddings::{model_name}"
            except Exception as exc:  # pragma: no cover
                log.info("Dense retriever unavailable (%s); using TF-IDF.", exc)

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 category: Optional[str] = None) -> List[Passage]:
        return [p for p, _ in self.index.search(query, top_k or self.top_k, category)]

    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None,
                             category: Optional[str] = None
                             ) -> List[Tuple[Passage, float]]:
        return self.index.search(query, top_k or self.top_k, category)

    def as_context(self, query: str, top_k: Optional[int] = None,
                   category: Optional[str] = None) -> str:
        hits = self.retrieve_with_scores(query, top_k, category)
        if not hits:
            return "(no matching knowledge-base entries)"
        return "\n".join(
            f"{i}. [{p.category or 'general'} | {p.facet} | sim={s}] {p.text}"
            for i, (p, s) in enumerate(hits, 1)
        )
