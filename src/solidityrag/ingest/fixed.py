"""Fixed-size line windows: the structure-unaware baseline chunker.

Every file is cut into windows of ``window`` lines, stepping ``window - overlap``
lines each time, with no regard for where functions start or end. Windows cover
the whole file, including pragmas, state variables and comments. Their
``embed_text`` carries no contract or function header, because that metadata is
exactly what the function-level chunker adds and this baseline should not have.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from functools import partial
from typing import Callable

from .chunker import _CONTRACT, Chunk, _blank

ChunkFn = Callable[[str, str], "list[Chunk]"]


def chunk_source_fixed(
    src: str, file: str = "<memory>", window: int = 8, overlap: int = 0
) -> list[Chunk]:
    if window < 1:
        raise ValueError("window must be at least 1 line")
    if not 0 <= overlap < window:
        raise ValueError("overlap must be in [0, window)")
    lines = src.splitlines(keepends=True)
    if not lines:
        return []

    # contract that owns each window: the last contract declared at or before its first line
    clean = _blank(src)
    line_starts = [0] + [m.end() for m in re.finditer("\n", src)]
    decl_lines, decl_names = [], []
    for m in _CONTRACT.finditer(clean):
        decl_lines.append(bisect_right(line_starts, m.start()))
        decl_names.append(m.group(1))

    step = window - overlap
    chunks: list[Chunk] = []
    start = 0
    while True:
        end = min(start + window, len(lines))
        text = "".join(lines[start:end])
        s, e = start + 1, end  # 1-based, inclusive
        owner = bisect_right(decl_lines, s) - 1
        chunk = Chunk(
            id=f"{file}:L{s}-{e}",
            file=file,
            contract=decl_names[owner] if owner >= 0 else "?",
            name=f"L{s}-{e}",
            kind="window",
            start_line=s,
            end_line=e,
            text=text,
        )
        if chunk.clean_text.strip():  # skip blank or annotation-only windows
            chunks.append(chunk)
        if end >= len(lines):
            break
        start += step
    return chunks


def fixed_chunker(window: int = 8, overlap: int = 0) -> ChunkFn:
    return partial(chunk_source_fixed, window=window, overlap=overlap)
