"""Retrieval metrics against SmartBugs line annotations."""

from __future__ import annotations

from typing import Sequence

from ..ingest.chunker import Chunk
from ..ingest.smartbugs import Annotations


def vulnerable_lines(chunk: Chunk, annotations: Annotations, category: str) -> set[tuple[str, int]]:
    """Annotated vulnerable lines of a category that fall inside the chunk, as (file, line)."""
    return {
        (chunk.file, line)
        for ann in annotations.for_file(chunk.file)
        if ann.category == category
        for line in ann.lines
        if chunk.start_line <= line <= chunk.end_line
    }


def is_relevant(chunk: Chunk, annotations: Annotations, category: str) -> bool:
    """A chunk is relevant to a category if it covers an annotated vulnerable line of it."""
    return bool(vulnerable_lines(chunk, annotations, category))


def lines_covered(chunks: Sequence[Chunk]) -> set[tuple[str, int]]:
    """Distinct source lines spanned by the chunks (overlapping chunks count once)."""
    return {(c.file, n) for c in chunks for n in range(c.start_line, c.end_line + 1)}


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    """Relevant results in the top k, divided by k (missing results count as misses)."""
    return sum(flags[:k]) / k


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    """1.0 if at least one of the top k results is relevant, else 0.0."""
    return 1.0 if any(flags[:k]) else 0.0