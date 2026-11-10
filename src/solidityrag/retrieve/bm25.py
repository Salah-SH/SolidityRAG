from __future__ import annotations

import numpy as np

from ..tokenize import tokenize
from .base import Retriever


class BM25Retriever(Retriever):
    name = "bm25"

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = np.asarray(self.index.bm25.get_scores(tokenize(query)))
        order = np.argsort(-scores, kind="stable")[:k]
        # a zero score means no query term matched: don't return arbitrary chunks
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]