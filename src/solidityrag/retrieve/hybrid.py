"""Hybrid retrieval: reciprocal rank fusion over other retrievers."""

from __future__ import annotations

from typing import Sequence

from ..index.store import Index
from .base import Retriever


def rrf(rankings: Sequence[Sequence[int]], c: int = 60) -> list[tuple[int, float]]:
    """Reciprocal rank fusion.

    Each item scores sum(1 / (c + rank)) over the rankings that contain it
    (rank starts at 1). Ranks are comparable across retrievers even when raw
    scores are not, so there are no weights to tune. Ties keep first-seen order.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (c + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


class HybridRetriever(Retriever):
    name = "hybrid"

    def __init__(self, index: Index, retrievers: Sequence[Retriever], pool: int = 20, c: int = 60):
        super().__init__(index)
        self.retrievers = list(retrievers)
        self.pool = pool
        self.c = c

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        rankings = [[i for i, _ in r.rank(query, self.pool)] for r in self.retrievers]
        return rrf(rankings, self.c)[:k]