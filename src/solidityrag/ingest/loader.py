"""Walk a directory of Solidity files and chunk all of them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from typing import Callable

from .chunker import Chunk, chunk_file, chunk_source


@dataclass
class LoadResult:
    chunks: list[Chunk] = field(default_factory=list)
    empty_files: list[str] = field(default_factory=list)  # files that produced zero chunks
    n_files: int = 0


def load_chunks(
    root: str | Path, chunk_fn: Callable[[str, str], list[Chunk]] = chunk_source
) -> LoadResult:
    root = Path(root)
    if root.is_file():
        files, base = [root], root.parent
    else:
        files, base = sorted(root.rglob("*.sol")), root

    result = LoadResult(n_files=len(files))
    for f in files:
        chunks = chunk_file(f, root=base, chunk_fn=chunk_fn)
        if not chunks:
            result.empty_files.append(f.relative_to(base).as_posix())
        result.chunks.extend(chunks)
    return result