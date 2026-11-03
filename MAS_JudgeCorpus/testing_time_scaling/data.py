from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DATASET_FILES = {
    "hotpotqa": "datasets/BrowseComp-Plus/hotpotqa/data/hotpotqa_test.jsonl",
    "bcp": "datasets/BrowseComp-Plus/data/browsecomp_plus_test.jsonl",
}
EXPECTED_COUNTS = {"hotpotqa": 200, "bcp": 168}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(dataset: str, mas_root: Path, *, limit: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if dataset not in DATASET_FILES:
        raise ValueError(f"unsupported dataset: {dataset}")
    path = (mas_root / DATASET_FILES[dataset]).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    full_count = len(rows)
    if limit > 0:
        rows = rows[:limit]
    samples = []
    for idx, row in enumerate(rows):
        question = str(row.get("query") or row.get("question") or "").strip()
        samples.append({
            "dataset": dataset,
            "index": idx,
            "sample_id": str(row.get("query_id") or row.get("id") or idx),
            "question": question,
            "gold": row.get("answer"),
            "raw": row,
        })
    return samples, {
        "path": str(path), "sha256": sha256_file(path), "full_count": full_count,
        "selected_count": len(samples), "expected_count": EXPECTED_COUNTS[dataset],
    }
