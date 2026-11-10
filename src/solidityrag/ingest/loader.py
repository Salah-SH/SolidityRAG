"""Walk a directory of Solidity files and chunk all of them."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .chunker import Chunk, chunk_file


@dataclass
class LoadResult:
    chunks: list[Chunk] = field(default_factory=list)
    empty_files: list[str] = field(default_factory=list)  # files that produced zero chunks
    n_files: int = 0


def load_chunks(root: str | Path) -> LoadResult:
    root = Path(root)
    if root.is_file():
        files, base = [root], root.parent
    else:
        files, base = sorted(root.rglob("*.sol")), root

    result = LoadResult(n_files=len(files))
    for f in files:
        chunks = chunk_file(f, root=base)
        if not chunks:
            result.empty_files.append(f.relative_to(base).as_posix())
        result.chunks.extend(chunks)
    return result