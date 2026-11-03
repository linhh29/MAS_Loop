from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any

_BOXED = re.compile(r"\\boxed\s*\{")
_LABEL = re.compile(r"(?im)^\s*(?:[#>*_-]+\s*)*(?:\*\*|__)?\s*(?:final\s+answer|answer|conclusion)\s*(?:\*\*|__)?\s*[:：]\s*(.*)$")
_FORMAT = re.compile(r"^(?:</?think>|\*+|_+|\$+|`+|#+|[-=]{2,})$", re.I)


def _last_boxed(text: str) -> str | None:
    values = []
    for match in _BOXED.finditer(text):
        start, depth, idx = match.end(), 1, match.end()
        while idx < len(text) and depth:
            depth += (text[idx] == "{") - (text[idx] == "}")
            idx += 1
        if depth == 0:
            values.append(text[start:idx - 1].strip())
    if not values:
        return None
    val = values[-1]
    wrapper = re.fullmatch(r"\\text\{(.*)\}", val, flags=re.S)
    return (wrapper.group(1) if wrapper else val).strip()


def extract_answer(text: Any) -> str:
    text = str(text or "").strip()
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed
    labelled = list(_LABEL.finditer(text))
    for match in reversed(labelled):
        value = match.group(1).strip().strip("*_ ")
        if value and not _FORMAT.fullmatch(value):
            return value
        for line in text[match.end():].splitlines():
            value = line.strip().strip("*_ ")
            if value and not _FORMAT.fullmatch(value):
                return value
    lines = [line.strip().strip("*_ ") for line in text.splitlines()]
    lines = [line for line in lines if line and not _FORMAT.fullmatch(line)]
    return lines[-1] if lines else ""


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower().strip()
    # Match Hotpot-style article removal without collapsing a legitimate one-token answer such as "A".
    if len(text.split()) > 1:
        text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def is_correct(prediction: Any, gold: Any) -> bool:
    p, g = normalize_answer(extract_answer(prediction)), normalize_answer(gold)
    return bool(g and (p == g or g in p))


def majority_vote(outputs: list[Any]) -> tuple[int, str, dict[str, int]]:
    answers = [extract_answer(x) for x in outputs]
    norms = [normalize_answer(x) for x in answers]
    counts = Counter(norms)
    if not counts:
        return 0, "", {}
    best_count = max(counts.values())
    winner = next(norm for norm in norms if counts[norm] == best_count)
    idx = norms.index(winner)
    return idx, answers[idx], dict(counts)
