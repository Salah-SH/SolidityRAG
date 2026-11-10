import sys
from collections import Counter

from solidityrag.index.store import Index
from solidityrag.ingest.smartbugs import Annotations

idx = Index.load(sys.argv[1])
ann = Annotations.from_json(sys.argv[2])

by_file = {}
for c in idx.chunks:
    by_file.setdefault(c.file, []).append(c)

total = covered = 0
missing = Counter()
for a in ann.annotations:
    # chunk labels are relative to the indexed dataset/ directory
    key = a.path.removeprefix("dataset/")
    for line in a.lines:
        total += 1
        if any(c.start_line <= line <= c.end_line for c in by_file.get(key, [])):
            covered += 1
        else:
            missing[a.category] += 1

print(f"annotated lines: {total}, inside a chunk: {covered} ({covered / total:.1%})")
print("uncovered by category:", dict(missing))