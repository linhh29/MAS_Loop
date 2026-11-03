"""Strict rule-based scoring for AIME-style mathematical answers."""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Tuple

from sympy import E, Float, Integer, Rational, Symbol, pi, simplify, sqrt
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)


_FINAL_MARKER_RE = re.compile(
    r"(?:\*\*)?(?:final\s+answer|answer)(?:\*\*)?\s*(?:is|=|:|：)\s*([^\n]+)",
    re.IGNORECASE,
)
_INTEGER_RE = re.compile(r"(?<![\d.])-?\d+(?:,\d{3})*(?![\d.])")
_UNSAFE_TEXT_COMMAND_RE = re.compile(r"\\(?:textbf|textit|mbox)\s*\{", re.IGNORECASE)
_SIMPLE_LHS_RE = re.compile(r"^[A-Za-z](?:\^\{?-?\d+\}?)?(?:[+\-*/][A-Za-z](?:\^\{?-?\d+\}?)?)*$")
_SAFE_EXPRESSION_RE = re.compile(r"^[0-9A-Za-z+\-*/().^]+$")
_SAFE_MULTI_LETTER_IDENTIFIERS = frozenset({"sqrt", "pi"})
_PARSE_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)


def _boxed_payloads(text: str) -> list[str]:
    """Return balanced ``\\boxed{...}`` payloads in source order."""
    source = str(text or "")
    payloads: list[str] = []
    marker = r"\boxed"
    cursor = 0
    while True:
        start = source.find(marker, cursor)
        if start < 0:
            break
        brace = source.find("{", start + len(marker))
        if brace < 0:
            break
        # Do not let a distant unrelated brace turn malformed ``\boxed`` into
        # an answer candidate.
        if source[start + len(marker):brace].strip():
            cursor = start + len(marker)
            continue
        depth = 0
        for end in range(brace, len(source)):
            if source[end] == "{":
                depth += 1
            elif source[end] == "}":
                depth -= 1
                if depth == 0:
                    payloads.append(source[brace + 1:end].strip())
                    cursor = end + 1
                    break
        else:
            break
    return payloads


def _last_box_is_complete(text: str) -> bool:
    """Require the final ``\\boxed`` assertion itself to be well formed."""
    source = str(text or "")
    last_marker = source.rfind(r"\boxed")
    return last_marker < 0 or bool(_boxed_payloads(source[last_marker:]))


def _strip_wrappers(value: str) -> str:
    text = str(value or "").strip().strip("`*_ ")
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = text.rstrip(".,;:，。 ")
    return text


def _assignment_rhs(value: str) -> str:
    """Accept a simple final assignment such as ``n=4`` or ``r^2+s^2=504``."""
    text = _strip_wrappers(value)
    if text.count("=") != 1:
        return text
    lhs, rhs = (part.strip() for part in text.split("=", 1))
    compact_lhs = re.sub(r"\s+", "", lhs).replace("\\left", "").replace("\\right", "")
    compact_lhs = compact_lhs.replace("{", "").replace("}", "")
    if rhs and len(rhs) <= 120 and _SIMPLE_LHS_RE.fullmatch(compact_lhs):
        return rhs
    return text


def _replace_innermost_latex(text: str) -> str:
    """Translate common nested LaTeX arithmetic into a SymPy-safe form."""
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", text)
        text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", text)
        # Numeric text wrappers are formatting, not answer content.  Lettered
        # wrappers remain untouched and are rejected below instead of guessed.
        text = re.sub(
            r"\\text\s*\{\s*([+\-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\}",
            r"\1",
            text,
            flags=re.IGNORECASE,
        )
    return text


def _normalize_expression(value: Any) -> str:
    text = _assignment_rhs(str(value or ""))
    if not text or _UNSAFE_TEXT_COMMAND_RE.search(text):
        return ""
    text = text.replace("−", "-").replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace(",", "")
    # A degree marker is a presentation unit for the same numeric final value.
    text = re.sub(r"(?:\^\s*)?\\circ\s*$|°\s*$", "", text).strip()
    text = _replace_innermost_latex(text)
    text = re.sub(r"\\(?:mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    # Percent is mathematical notation, so 80% and 0.8 are exactly equivalent.
    percent = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\\?%\s*", text)
    if percent:
        return f"({percent.group(1)})/100"
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("\\pi", "pi")
    # Remaining braces are ordinary arithmetic grouping, e.g. 2^{3}.
    text = text.replace("{", "(").replace("}", ")")
    return re.sub(r"\s+", "", _strip_wrappers(text))


def _safe_parse_expression(value: str):
    """Parse only a small arithmetic grammar; reject arbitrary identifiers."""
    if not value or not _SAFE_EXPRESSION_RE.fullmatch(value):
        return None
    identifiers = re.findall(r"[A-Za-z]+", value)
    if any(len(name) > 1 and name not in _SAFE_MULTI_LETTER_IDENTIFIERS for name in identifiers):
        return None
    local_dict: dict[str, Any] = {"sqrt": sqrt, "pi": pi, "e": E}
    for name in identifiers:
        if len(name) == 1 and name != "e":
            local_dict[name] = Symbol(name)
    # Security: the input grammar and identifier allowlist above prevent SymPy's
    # eval-backed parser from resolving Python builtins or arbitrary callables.
    global_dict = {
        "Integer": Integer,
        "Float": Float,
        "Rational": Rational,
        "Symbol": Symbol,
    }
    try:
        return parse_expr(
            value,
            local_dict=local_dict,
            global_dict=global_dict,
            transformations=_PARSE_TRANSFORMATIONS,
            evaluate=True,
        )
    except Exception:
        return None


def math_equal(prediction: Any, reference: Any) -> bool:
    """Return exact mathematical equivalence; never award partial credit."""
    pred = _normalize_expression(prediction)
    ref = _normalize_expression(reference)
    if not pred or not ref:
        return False
    if pred == ref:
        return True
    try:
        return Fraction(pred) == Fraction(ref)
    except (ValueError, ZeroDivisionError):
        pass
    p = _safe_parse_expression(pred)
    r = _safe_parse_expression(ref)
    if p is None or r is None:
        return False
    try:
        return bool(simplify(p - r) == 0)
    except Exception:
        return False


class AIMEScorer:
    """Extract one final answer and grade it by exact mathematical equivalence."""

    def extract_model_answer(self, text: str) -> str:
        source = str(text or "")
        boxed = _boxed_payloads(source)
        if boxed:
            if not _last_box_is_complete(source):
                return ""
            return _strip_wrappers(boxed[-1])
        if r"\boxed" in source:
            return ""

        marked = _FINAL_MARKER_RE.findall(source)
        if marked:
            return _strip_wrappers(marked[-1])

        # Conservative compatibility fallback: inspect only the final non-empty
        # sentence/line, never arbitrary numbers from the reasoning body.
        lines = [line.strip() for line in source.splitlines() if line.strip()]
        tail = lines[-1] if lines else source.strip()
        sentences = [s.strip() for s in re.split(r"(?<!\d)[.!?]\s+", tail) if s.strip()]
        candidate = sentences[-1] if sentences else tail
        stripped_candidate = _strip_wrappers(candidate)
        if _safe_parse_expression(_normalize_expression(stripped_candidate)) is not None:
            return stripped_candidate
        math_probe = re.sub(
            r"\\(?:frac|dfrac|tfrac|sqrt|cdot|times|left|right|mathrm|operatorname)",
            "",
            stripped_candidate,
        )
        if stripped_candidate and not re.search(r"[A-Za-z]{2,}", math_probe):
            return stripped_candidate
        integers = _INTEGER_RE.findall(candidate)
        return integers[-1].replace(",", "") if integers else stripped_candidate

    def calculate_score(self, expected_output: str, prediction: str) -> Tuple[int, str]:
        expected = self.extract_model_answer(expected_output)
        predicted = self.extract_model_answer(prediction)
        return (int(math_equal(predicted, expected)), predicted)

    def math_equal(self, prediction: Any, reference: Any) -> bool:
        return math_equal(prediction, reference)


def score_answer(gold_answer: str, model_output: str) -> Tuple[int, str]:
    """Convenience: return ``(0|1, extracted_prediction)``."""
    return AIMEScorer().calculate_score(gold_answer, model_output)
