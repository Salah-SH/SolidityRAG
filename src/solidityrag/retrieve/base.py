"""Common retriever interface: every method returns ranked chunk indices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..index.store import Index
from ..ingest.chunker import Chunk


@dataclass(frozen=True)
class Result:
    chunk: Chunk
    score: float
    rank: int  # 1-based


class Retriever(ABC):
    name: str = "retriever"

    def __init__(self, index: Index):
        self.index = index

    @abstractmethod
    def rank(self, query: str, k: int) -> list[tuple[int, float]]:
        """Return up to k (chunk_index, score) pairs, best first."""

    def search(self, query: str, k: int = 5) -> list[Result]:
        return [
            Result(self.index.chunks[i], score, r)
            for r, (i, score) in enumerate(self.rank(query, k), start=1)
        ]