"""Ground-truth annotations from SmartBugs Curated (``vulnerabilities.json``).

Each entry looks like::

    {"path": "dataset/reentrancy/simple_dao.sol",
     "vulnerabilities": [{"lines": [19], "category": "reentrancy"}], ...}

Line numbers are 1-based. Chunk file labels depend on which directory was
indexed (``dataset/reentrancy/x.sol`` or ``reentrancy/x.sol``), so lookup
matches on path suffixes instead of requiring an exact path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Annotation:
    path: str
    category: str
    lines: frozenset[int]


class Annotations:
    def __init__(self, annotations: list[Annotation]):
        self.annotations = annotations
        self._by_path: dict[str, list[Annotation]] = {}
        for a in annotations:
            self._by_path.setdefault(a.path, []).append(a)

        # every trailing path suffix -> the full paths it could refer to
        self._suffixes: dict[str, set[str]] = {}
        for p in self._by_path:
            parts = p.split("/")
            for i in range(len(parts)):
                self._suffixes.setdefault("/".join(parts[i:]), set()).add(p)

    @classmethod
    def from_json(cls, path: str | Path) -> "Annotations":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items: list[Annotation] = []
        for entry in data:
            for v in entry.get("vulnerabilities", []):
                items.append(
                    Annotation(
                        path=entry["path"].replace("\\", "/"),
                        category=v["category"],
                        lines=frozenset(int(x) for x in v["lines"]),
                    )
                )
        return cls(items)

    def categories(self) -> set[str]:
        return {a.category for a in self.annotations}

    def for_file(self, file: str) -> list[Annotation]:
        norm = file.replace("\\", "/").removeprefix("./")
        parts = norm.split("/")
        # the chunk path may carry extra leading directories: try longest suffix first
        for i in range(len(parts)):
            hit = self._by_path.get("/".join(parts[i:]))
            if hit:
                return hit
        # or the chunk path may be shorter than the annotation path
        candidates = self._suffixes.get(norm, set())
        if len(candidates) == 1:
            return self._by_path[next(iter(candidates))]
        return []