"""Function-level chunking for Solidity source.

This is the v0.1 chunker: comments and string literals are blanked out, then
function-like declarations are found with a regex and their bodies are matched
by counting braces. It needs no compiler, so it works on old contracts with
missing imports. A tree-sitter based parser is the planned replacement; the
``Chunk`` interface is meant to stay the same when that happens.
"""

from __future__ import annotations

import re
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path

from .sanitize import strip_label_leaks

# Comments and string literals, matched left to right so a "//" inside a string
# or an apostrophe inside a comment is handled by whichever starts first.
_BLANKABLE = re.compile(
    r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\\n])*\"|'(?:\\.|[^'\\\n])*'", re.S
)
_DECL = re.compile(
    r"\b(?:(function|modifier)\b\s*(\w*)|(constructor|fallback|receive)\b(?=\s*\())"
)
_CONTRACT = re.compile(r"\b(?:contract|library|interface)\s+(\w+)")


@dataclass(frozen=True)
class Chunk:
    id: str
    file: str
    contract: str
    name: str
    kind: str  # function | modifier | constructor | fallback | receive
    start_line: int
    end_line: int
    text: str  # raw source of the declaration and body

    @property
    def clean_text(self) -> str:
        """Source with benchmark label leaks removed (used for indexing and prompts)."""
        return strip_label_leaks(self.text)

    @property
    def embed_text(self) -> str:
        return f"// {self.contract}.{self.name}\n{self.clean_text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def _blank(src: str) -> str:
    return _BLANKABLE.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), src)


def _match_brace(clean: str, open_idx: int) -> int | None:
    depth = 0
    for j in range(open_idx, len(clean)):
        ch = clean[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return j
    return None


def chunk_source(src: str, file: str = "<memory>") -> list[Chunk]:
    clean = _blank(src)
    n = len(clean)
    newlines = [i for i, ch in enumerate(src) if ch == "\n"]

    def line_of(pos: int) -> int:
        return bisect_left(newlines, pos) + 1

    contract_pos = []
    contract_names = []
    for m in _CONTRACT.finditer(clean):
        contract_pos.append(m.start())
        contract_names.append(m.group(1))

    chunks: list[Chunk] = []
    pos = 0
    while True:
        m = _DECL.search(clean, pos)
        if not m:
            break

        if m.group(1):  # function or modifier
            kind = m.group(1)
            name = m.group(2)
            if kind == "function" and not name:  # legacy unnamed fallback
                kind, name = "fallback", "fallback"
        else:  # constructor, fallback, receive
            kind = name = m.group(3)

        i = m.end()
        while i < n and clean[i] not in "{;":
            i += 1
        if i >= n:
            break
        if clean[i] == ";":  # declaration without a body (interface, abstract)
            pos = i + 1
            continue

        end = _match_brace(clean, i)
        if end is None:  # unbalanced braces: skip this declaration
            pos = i + 1
            continue

        idx = bisect_right(contract_pos, m.start()) - 1
        owner = contract_names[idx] if idx >= 0 else "?"
        start_line = line_of(m.start())
        end_line = line_of(end)
        chunks.append(
            Chunk(
                id=f"{file}:{owner}.{name}:{start_line}",
                file=file,
                contract=owner,
                name=name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                text=src[m.start() : end + 1],
            )
        )
        pos = end + 1

    return chunks


def chunk_file(path: str | Path, root: str | Path | None = None) -> list[Chunk]:
    path = Path(path)
    src = path.read_text(encoding="utf-8", errors="ignore")
    label = path.relative_to(root).as_posix() if root else path.as_posix()
    return chunk_source(src, label)