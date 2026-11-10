"""Syntax-aware chunking with a tree-sitter parse of Solidity.

Two chunkers share one parse:

* ``chunk_source_ast_functions``: one chunk per function, modifier,
  constructor, fallback or receive that has a body. On SmartBugs Curated it
  produces exactly the same chunks as the regex chunker in ``chunker.py``,
  but it does not depend on brace counting.
* ``chunk_source_ast``: size-bounded syntax chunks, following cAST (Zhang et
  al., 2025). A node that fits in ``max_lines`` is kept whole; a larger node is
  split into its children, recursively; consecutive small pieces are merged
  while they still fit (never across contracts). Chunks therefore never cut through a statement, cover
  the whole file (state variables, events and structs included), and are about
  the size of a fixed window. Each chunk is named after the function(s) it
  overlaps, so its ``embed_text`` keeps the ``// Contract.function`` header.

Needs the ``ast`` extra: ``pip install 'solidityrag[ast]'``.
"""

from __future__ import annotations

import warnings
from bisect import bisect_right
from functools import lru_cache, partial
from typing import Callable

from .chunker import Chunk

FUNCTION_NODES = {
    "function_definition": "function",
    "modifier_definition": "modifier",
    "constructor_definition": "constructor",
    "fallback_receive_definition": "fallback",
}
CONTRACT_NODES = {"contract_declaration", "library_declaration", "interface_declaration"}

Span = tuple[int, int]  # 1-based, inclusive line range


@lru_cache(maxsize=1)
def _parser():
    try:
        import tree_sitter_solidity
        from tree_sitter import Language, Parser
    except ImportError as e:
        raise ImportError("AST chunking needs tree-sitter: pip install 'solidityrag[ast]'") from e
    with warnings.catch_warnings():  # the grammar still hands over an int pointer
        warnings.simplefilter("ignore", DeprecationWarning)
        return Parser(Language(tree_sitter_solidity.language()))


def _span(node) -> Span:
    return node.start_point[0] + 1, node.end_point[0] + 1


def _size(span: Span) -> int:
    return span[1] - span[0] + 1


def _name_of(node) -> str:
    if node.type == "constructor_definition":
        return "constructor"
    if node.type == "fallback_receive_definition":
        return "receive" if b"receive" in node.text.split(b"(")[0] else "fallback"
    ident = node.child_by_field_name("name")
    if ident is None:  # legacy unnamed fallback: function() payable { ... }
        return "fallback"
    return ident.text.decode("utf-8", "ignore")


class _Outline:
    """Contracts and functions of one parsed file, looked up by line."""

    def __init__(self, root):
        self.contracts: list[tuple[Span, str]] = []
        self.functions: list[tuple[Span, str, str, object]] = []  # span, name, kind, node
        self._walk(root)
        self.contracts.sort()
        self._contract_starts = [s for (s, _), _ in self.contracts]

    def _walk(self, node) -> None:
        if node.type in CONTRACT_NODES:
            ident = node.child_by_field_name("name")
            self.contracts.append((_span(node), ident.text.decode() if ident is not None else "?"))
        if node.type in FUNCTION_NODES:
            if node.child_by_field_name("body") is not None:
                self.functions.append((_span(node), _name_of(node), FUNCTION_NODES[node.type], node))
            return
        for child in node.children:
            self._walk(child)

    def contract_at(self, line: int) -> str:
        i = bisect_right(self._contract_starts, line) - 1
        if i >= 0 and self.contracts[i][0][0] <= line <= self.contracts[i][0][1]:
            return self.contracts[i][1]
        return "?"

    def functions_in(self, span: Span) -> list[str]:
        return [name for (s, e), name, _, _ in self.functions if s <= span[1] and span[0] <= e]


def _parse(src: str):
    return _parser().parse(src.encode("utf-8")).root_node


def chunk_source_ast_functions(src: str, file: str = "<memory>") -> list[Chunk]:
    outline = _Outline(_parse(src))
    lines = src.splitlines(keepends=True)
    chunks = []
    for span, name, kind, node in outline.functions:
        if kind == "fallback" and name == "receive":
            kind = "receive"
        owner = outline.contract_at(span[0])
        chunks.append(
            Chunk(
                id=f"{file}:{owner}.{name}:{span[0]}",
                file=file,
                contract=owner,
                name=name,
                kind=kind,
                start_line=span[0],
                end_line=span[1],
                # declaration through the closing brace, like the regex chunker
                text=src.encode("utf-8")[node.start_byte : node.end_byte].decode("utf-8", "ignore")
                if lines
                else "",
            )
        )
    return chunks


def _split(node, max_lines: int) -> list[Span]:
    """cAST: keep nodes that fit, recurse into those that don't, merge small siblings."""
    span = _span(node)
    if _size(span) <= max_lines or not node.children:
        return [span]
    groups: list[Span] = []
    current: Span | None = None
    for child in node.children:
        cs = _span(child)
        if _size(cs) > max_lines:
            if current:
                groups.append(current)
                current = None
            groups.extend(_split(child, max_lines))
            continue
        if current is None:
            current = cs
        elif _size((current[0], max(current[1], cs[1]))) <= max_lines:
            current = (current[0], max(current[1], cs[1]))
        else:
            groups.append(current)
            current = cs
    if current:
        groups.append(current)
    return groups


def _merge_neighbors(spans: list[Span], max_lines: int, contract_at) -> list[Span]:
    """Join consecutive pieces of the same contract while the result still fits.

    Recursion leaves fragments such as a lone ``contract A {`` line or a
    closing brace; this pass folds them into their neighbors so chunks come
    close to the budget instead of averaging half of it.
    """
    merged: list[Span] = []
    for span in sorted(set(spans)):
        if merged:
            prev = merged[-1]
            joined = (prev[0], max(prev[1], span[1]))
            same_contract = contract_at(prev[0]) == contract_at(span[0])
            if same_contract and _size(joined) <= max_lines:
                merged[-1] = joined
                continue
        merged.append(span)
    return merged


def chunk_source_ast(src: str, file: str = "<memory>", max_lines: int = 8) -> list[Chunk]:
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")
    root = _parse(src)
    outline = _Outline(root)
    lines = src.splitlines(keepends=True)
    chunks: list[Chunk] = []
    # split per top-level declaration so a contract never shares a chunk with another
    pieces = [span for child in root.children for span in _split(child, max_lines)]
    for span in _merge_neighbors(pieces, max_lines, outline.contract_at):
        s, e = span
        funcs = outline.functions_in(span)
        name = "+".join(funcs) if funcs else "declarations"
        owner = outline.contract_at(s)
        chunk = Chunk(
            id=f"{file}:{owner}.{name}:L{s}-{e}",
            file=file,
            contract=owner,
            name=name,
            kind="ast",
            start_line=s,
            end_line=e,
            text="".join(lines[s - 1 : e]),
        )
        if chunk.clean_text.strip():
            chunks.append(chunk)
    return chunks


def ast_chunker(max_lines: int = 8) -> Callable[[str, str], list[Chunk]]:
    return partial(chunk_source_ast, max_lines=max_lines)
