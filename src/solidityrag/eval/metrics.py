"""Retrieval metrics against SmartBugs line annotations."""

from __future__ import annotations

from typing import Sequence

from ..ingest.chunker import Chunk
from ..ingest.smartbugs import Annotations


def is_relevant(chunk: Chunk, annotations: Annotations, category: str) -> bool:
    """A chunk is relevant to a category if it covers an annotated vulnerable line of it."""
    for ann in annotations.for_file(chunk.file):
        if ann.category != category:
            continue
        if any(chunk.start_line <= line <= chunk.end_line for line in ann.lines):
            return True
    return False


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    """Relevant results in the top k, divided by k (missing results count as misses)."""
    return sum(flags[:k]) / k


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    """1.0 if at least one of the top k results is relevant, else 0.0."""
    return 1.0 if any(flags[:k]) else 0.0