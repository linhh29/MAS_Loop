"""Official-style HotpotQA answer exact-match and token F1 metrics."""
from __future__ import annotations
import re
import string
from collections import Counter
from typing import Any


def normalize_answer(text: str) -> str:
    text = "".join(ch for ch in (text or "").lower() if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_metrics(prediction: str, gold: str | None) -> dict[str, Any]:
    if gold is None:
        return {"available": False, "exact_match": None, "f1": None, "precision": None, "recall": None}
    pred, target = normalize_answer(prediction), normalize_answer(gold)
    exact = float(pred == target)
    special = {"yes", "no", "noanswer"}
    if (pred in special or target in special) and pred != target:
        return {"available": True, "exact_match": exact, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    pred_tokens, target_tokens = pred.split(), target.split()
    overlap = sum((Counter(pred_tokens) & Counter(target_tokens)).values())
    if not pred_tokens or not target_tokens:
        precision = recall = f1 = float(pred_tokens == target_tokens)
    elif not overlap:
        precision = recall = f1 = 0.0
    else:
        precision, recall = overlap / len(pred_tokens), overlap / len(target_tokens)
        f1 = 2 * precision * recall / (precision + recall)
    return {"available": True, "exact_match": exact, "f1": f1, "precision": precision, "recall": recall}
