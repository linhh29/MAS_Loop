from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

TOKEN_KEYS = ("input_tokens", "output_tokens", "total_tokens")


def normalize_usage(usage: dict[str, Any] | None, *, source: str = "api") -> dict[str, Any]:
    raw = dict(usage or {})
    inp = int(raw.get("input_tokens", raw.get("prompt_tokens", 0)) or 0)
    out = int(raw.get("output_tokens", raw.get("completion_tokens", 0)) or 0)
    total = int(raw.get("total_tokens", inp + out) or (inp + out))
    return {
        **raw,
        "input_tokens": inp,
        "output_tokens": out,
        "total_tokens": total,
        "usage_source": str(raw.get("usage_source") or source),
    }


def sum_usage(items: Iterable[dict[str, Any] | None]) -> dict[str, Any]:
    rows = [normalize_usage(x) for x in items if x]
    sources: set[str] = set()
    for row in rows:
        nested = row.get("usage_sources")
        if isinstance(nested, (list, tuple, set)):
            sources.update(str(source) for source in nested)
        else:
            sources.add(str(row.get("usage_source", "unknown")))
    return {
        "input_tokens": sum(int(x["input_tokens"]) for x in rows),
        "output_tokens": sum(int(x["output_tokens"]) for x in rows),
        "total_tokens": sum(int(x["total_tokens"]) for x in rows),
        "usage_sources": sorted(sources),
        "num_calls": sum(int(x.get("num_calls", 1)) for x in rows),
    }


@dataclass
class TokenLedger:
    calls: list[dict[str, Any]] = field(default_factory=list)

    def add(self, *, role: str, model: str, usage: dict[str, Any] | None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        row = {
            "role": role,
            "model": model,
            **normalize_usage(usage),
            "metadata": dict(metadata or {}),
        }
        self.calls.append(row)
        return row

    def totals(self) -> dict[str, Any]:
        return {**sum_usage(self.calls), "calls": list(self.calls)}
