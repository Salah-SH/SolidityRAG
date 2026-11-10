"""Retrieve evidence, ask the LLM, and validate the citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..ingest.chunker import Chunk
from ..retrieve.base import Result, Retriever
from .citations import CitationReport, check_citations
from .prompt import build_prompt

DEFAULT_LLM = "claude-sonnet-5-5"


@dataclass
class Answer:
    question: str
    text: str
    results: list[Result]
    label_map: dict[str, Chunk]
    citations: CitationReport


def _default_client() -> Any:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("Generation needs the Anthropic SDK: pip install 'solidityrag[llm]'") from e
    return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY


def ask(
    question: str,
    retriever: Retriever,
    k: int = 5,
    model: str = DEFAULT_LLM,
    client: Any = None,
    max_tokens: int = 1200,
) -> Answer:
    results = retriever.search(question, k)
    system, user, label_map = build_prompt(question, results)

    client = client or _default_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
    return Answer(
        question=question,
        text=text,
        results=results,
        label_map=label_map,
        citations=check_citations(text, set(label_map)),
    )