"""Check that every citation in an answer points at an evidence block."""

from __future__ import annotations

import re
from dataclasses import dataclass

# matches [C1], [C1, C2] and [C1,C3]; consecutive [C1][C2] match as two groups
_GROUP = re.compile(r"\[(C\d+(?:\s*,\s*C\d+)*)\]")


def extract_labels(text: str) -> list[str]:
    labels: list[str] = []
    for group in _GROUP.findall(text):
        labels.extend(part.strip() for part in group.split(","))
    return labels


@dataclass(frozen=True)
class CitationReport:
    cited: list[str]
    valid: list[str]
    invalid: list[str]

    @property
    def validity(self) -> float | None:
        """Share of citations that exist in the evidence; None if nothing was cited."""
        return len(self.valid) / len(self.cited) if self.cited else None


def check_citations(text: str, available: set[str]) -> CitationReport:
    cited = extract_labels(text)
    return CitationReport(
        cited=cited,
        valid=[c for c in cited if c in available],
        invalid=[c for c in cited if c not in available],
    )