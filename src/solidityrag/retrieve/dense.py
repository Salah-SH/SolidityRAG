from __future__ import annotations

import numpy as np

from ..index.embedder import Embedder
from ..index.store import Index
from .base import Retriever


class DenseRetriever(Retriever):
    name = "dense"

    def __init__(self, index: Index, embedder: Embedder):
        super().__init__(index)
        if index.embeddings is None:
            raise ValueError("This index has no embeddings. Rebuild it with an embedder.")
        self.embedder = embedder

    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.index.embeddings @ self.embedder.encode_query(query)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order]