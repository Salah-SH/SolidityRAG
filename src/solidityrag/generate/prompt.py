"""Prompt construction. Evidence blocks get short labels ([C1]..[Ck]) because
models copy long path-style ids unreliably; labels map back to chunks."""

from __future__ import annotations

from typing import Sequence

from ..ingest.chunker import Chunk
from ..retrieve.base import Result

SYSTEM_PROMPT = """You are a smart contract security auditor.

You are given numbered evidence blocks of Solidity code, each labelled [C1], [C2], and so on.

Rules:
1. Base every statement only on the evidence blocks. Do not use outside knowledge about these contracts.
2. After each claim, cite the supporting block label in square brackets, for example [C2]. Use only labels that appear in the evidence.
3. For each issue, name the function, say what makes it vulnerable, and describe how it could be exploited.
4. If the evidence is not enough to answer, say so plainly instead of guessing. Do not invent function names or line numbers."""


def build_prompt(
    question: str, results: Sequence[Result]
) -> tuple[str, str, dict[str, Chunk]]:
    """Return (system prompt, user message, label -> chunk map)."""
    label_map: dict[str, Chunk] = {}
    blocks = []
    for n, res in enumerate(results, start=1):
        label = f"C{n}"
        c = res.chunk
        label_map[label] = c
        blocks.append(
            f"[{label}] file={c.file} contract={c.contract} {c.kind}={c.name} "
            f"lines={c.start_line}-{c.end_line}\n```solidity\n{c.clean_text}\n```"
        )
    evidence = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    user = f"Evidence:\n\n{evidence}\n\nQuestion: {question}"
    return SYSTEM_PROMPT, user, label_map