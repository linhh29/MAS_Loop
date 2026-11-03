#!/usr/bin/env python3
"""Generate an auditable eReport for two MAS-Loop training/evaluation runs.

The report deliberately separates four notions that are easy to conflate:
XML parseability, patch translation/application, structural MAS change, and
held-out task-score improvement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from mas_loop_rl.schema import (
    _repair_near_xml,
    extract_json,
    parse_mas_xml,
    parse_refiner_xml,
)

SCHEMA = "mas_loop_dual_run_ereport/4"
STAGES = ("orchestra_only", "orchestra_editor")
PLACEHOLDER_PATTERNS = (
    "observed bounded evidence",
    "earliest failure",
    "downstream consequence",
    "mechanism",
    "falsifiable expected change",
    "expected downstream effect",
    "minimal coherent repair",
    "expected effect",
    "regression risk",
    "improved task-specific role",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def latest_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            value = str(row.get(key) or "")
            if value:
                out[value] = row
    return out


def read_jsonl_glob(pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path().glob(pattern) if not pattern.startswith("/") else []):
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if pattern.startswith("/"):
        import glob
        for raw in sorted(glob.glob(pattern)):
            with Path(raw).open(encoding="utf-8") as handle:
                rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(rows)
    os.replace(tmp, path)


def finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mean(values: Iterable[Any]) -> float | None:
    clean = [x for x in (finite(v) for v in values) if x is not None]
    return statistics.fmean(clean) if clean else None


def population_variance(values: Iterable[Any]) -> float | None:
    clean = [x for x in (finite(v) for v in values) if x is not None]
    return statistics.pvariance(clean) if clean else None


def quantile(values: Iterable[Any], q: float) -> float | None:
    """Deterministic linear-interpolated quantile for compact signal audits."""
    clean = sorted(x for x in (finite(v) for v in values) if x is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = max(0.0, min(1.0, q)) * (len(clean) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def distribution(values: Iterable[Any]) -> dict[str, Any]:
    clean = [x for x in (finite(v) for v in values) if x is not None]
    return {
        "count": len(clean),
        "mean": mean(clean),
        "min": min(clean, default=None),
        "p10": quantile(clean, 0.10),
        "p25": quantile(clean, 0.25),
        "p50": quantile(clean, 0.50),
        "p75": quantile(clean, 0.75),
        "p90": quantile(clean, 0.90),
        "max": max(clean, default=None),
    }


def counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))}


def dominance(counter: Counter[Any]) -> dict[str, Any]:
    """Report both diversity and collapse; raw category counts alone hide dominance."""
    total = sum(counter.values())
    if not total:
        return {"total": 0, "distinct": 0, "top": None, "top_count": 0, "top_share": None}
    top, top_count = max(counter.items(), key=lambda item: (item[1], str(item[0])))
    return {
        "total": int(total),
        "distinct": len(counter),
        "top": str(top),
        "top_count": int(top_count),
        "top_share": top_count / total,
    }


def metric_trends(series: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    keys = sorted({key for row in series for key in (row.get("selected") or {})})
    trends: dict[str, dict[str, Any]] = {}
    for key in keys:
        values = [finite((row.get("selected") or {}).get(key)) for row in series]
        clean = [value for value in values if value is not None]
        if not clean:
            continue
        trends[key] = {
            "first": clean[0],
            "last": clean[-1],
            "delta": clean[-1] - clean[0],
            "mean": mean(clean),
            "min": min(clean),
            "max": max(clean),
            "observations": len(clean),
        }
    return trends


def _last_root_fragment(text: str, root: str) -> str | None:
    matches = list(re.finditer(rf"<{root}\b[^>]*>.*?</{root}\s*>", text, flags=re.I | re.S))
    if matches:
        return matches[-1].group(0)
    raw = text.strip()
    return raw if raw.lower().startswith(f"<{root}") else None


def classify_structured_output(text: Any, root: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    result = {"classification": "unrecoverable", "repairs": [], "error": ""}
    fragment = _last_root_fragment(raw, root)
    if fragment:
        try:
            elem = ET.fromstring(fragment)
            if elem.tag.split("}")[-1].lower() == root:
                (parse_mas_xml if root == "mas" else parse_refiner_xml)(raw)
                result["classification"] = "native_xml"
                return result
        except Exception:
            pass
        repaired, repairs = _repair_near_xml(fragment, root)
        if repairs:
            try:
                elem = ET.fromstring(repaired)
                if elem.tag.split("}")[-1].lower() == root:
                    (parse_mas_xml if root == "mas" else parse_refiner_xml)(raw)
                    result.update(classification="repaired_xml", repairs=repairs)
                    return result
            except Exception:
                pass
    try:
        value = extract_json(raw)
        if isinstance(value, dict):
            result["classification"] = "legacy_json_fallback"
            return result
    except Exception as exc:
        result["error"] = str(exc)
    return result



def _agent_ids(mas: Any) -> set[str]:
    if not isinstance(mas, dict):
        return set()
    return {
        str(agent.get("id"))
        for agent in (mas.get("agents") or [])
        if isinstance(agent, dict) and agent.get("id")
    }


def diagnosis_target_resolves(diagnosis: Any, mas: Any) -> bool:
    """Cheap structural proxy; it does not claim the diagnosis is causally correct."""
    if not isinstance(diagnosis, dict):
        return False
    ids = _agent_ids(mas)
    target_type = str(diagnosis.get("target_type") or "").lower()
    target_id = str(diagnosis.get("target_id") or "")
    if target_type == "agent":
        return target_id in ids
    if target_type == "edge" and "->" in target_id:
        source, target = (part.strip() for part in target_id.split("->", 1))
        return source in ids and target in ids
    if target_type in {"graph", "mas", "aggregation", "workflow"}:
        return bool(ids)
    return False


def evidence_location_resolves(location: Any, mas: Any) -> bool:
    """Resolve supported evidence-location syntax against the authored MAS graph."""
    text = str(location or "").strip()
    ids = _agent_ids(mas)
    if text.startswith("agent:"):
        agent_id = text.removeprefix("agent:").split(".", 1)[0]
        return agent_id in ids
    if text.startswith("edge:") and "->" in text:
        source, target = (part.strip().split(".", 1)[0] for part in text.removeprefix("edge:").split("->", 1))
        return source in ids and target in ids
    return False


def diagnosis_patch_aligned(diagnosis: Any, operations: Any) -> bool:
    """Proxy that checks whether applied operations touch the diagnosed target."""
    if not isinstance(diagnosis, dict) or not isinstance(operations, list) or not operations:
        return False
    target_type = str(diagnosis.get("target_type") or "").lower()
    target_id = str(diagnosis.get("target_id") or "")
    if target_type in {"graph", "mas", "aggregation", "workflow"}:
        return True
    if target_type == "agent":
        return any(
            isinstance(op, dict) and (
                str(op.get("agent_id") or op.get("id") or "") == target_id
                or target_id in {str(op.get("from") or ""), str(op.get("to") or "")}
            )
            for op in operations
        )
    if target_type == "edge" and "->" in target_id:
        source, target = (part.strip() for part in target_id.split("->", 1))
        return any(
            isinstance(op, dict)
            and str(op.get("from") or "") == source
            and str(op.get("to") or "") == target
            for op in operations
        )
    return False


def _contains_placeholder(value: Any) -> bool:
    text = str(value or "").lower()
    return any(phrase in text for phrase in PLACEHOLDER_PATTERNS)

def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stage_score(row: dict[str, Any], stage: str) -> int | None:
    value = row.get(stage)
    if not isinstance(value, dict) or value.get("score") not in (0, 1):
        return None
    return int(value["score"])


def bootstrap_delta(deltas: list[int], seed: int, draws: int) -> tuple[float | None, float | None]:
    if not deltas:
        return None, None
    rng = random.Random(seed)
    n = len(deltas)
    samples = sorted(sum(deltas[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return samples[int(0.025 * (draws - 1))], samples[int(0.975 * (draws - 1))]


def mcnemar_exact(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def compare_records(
    reference: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    reference_stage: str,
    current_stage: str,
    *,
    label: str,
    run: str,
    seed: int,
    draws: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    wins = losses = ties = 0
    deltas: list[int] = []
    for key in sorted(set(reference) & set(current)):
        before = stage_score(reference[key], reference_stage)
        after = stage_score(current[key], current_stage)
        if before is None or after is None:
            continue
        delta = after - before
        wins += delta == 1
        losses += delta == -1
        ties += delta == 0
        deltas.append(delta)
        transition = (
            "wrong_to_correct" if delta == 1 else
            "correct_to_wrong" if delta == -1 else
            "both_correct" if before == 1 else "both_wrong"
        )
        pairs.append({
            "run": run,
            "comparison": label,
            "sample_key": key,
            "sample_id": current[key].get("sample_id"),
            "reference_stage": reference_stage,
            "current_stage": current_stage,
            "reference_score": before,
            "current_score": after,
            "delta": delta,
            "transition": transition,
            "question": current[key].get("task"),
            "gold_answer": current[key].get("gold_answer"),
            "reference_prediction": (reference[key].get(reference_stage) or {}).get("prediction"),
            "current_prediction": (current[key].get(current_stage) or {}).get("prediction"),
        })
    low, high = bootstrap_delta(deltas, seed, draws)
    stats = {
        "run": run,
        "comparison": label,
        "paired_n": len(deltas),
        "wrong_to_correct": wins,
        "correct_to_wrong": losses,
        "both_correct": sum(x["transition"] == "both_correct" for x in pairs),
        "both_wrong": sum(x["transition"] == "both_wrong" for x in pairs),
        "paired_delta_mean": mean(deltas),
        "bootstrap_95ci_low": low,
        "bootstrap_95ci_high": high,
        "mcnemar_exact_p": mcnemar_exact(wins, losses),
    }
    return stats, pairs


def training_metric(root: Path, role: str) -> dict[str, Any]:
    path = root / "round_1" / role / "metrics" / "training_metrics.jsonl"
    latest = latest_jsonl(path, "global_step")
    if not latest:
        return {}
    ordered_rows = [latest[key] for key in sorted(latest, key=lambda x: int(x))]
    selected = [
        "actor/lr", "actor/loss", "actor/pg_loss", "actor/entropy", "actor/grad_norm",
        "actor/ppo_kl", "actor/kl_loss", "actor/kl_coef", "actor/pg_clipfrac",
        "critic/rewards/mean", "critic/rewards/min", "critic/rewards/max",
        "critic/advantages/mean", "critic/advantages/min", "critic/advantages/max",
        "perf/time_per_step", "perf/throughput", "perf/total_num_tokens",
        "timing_s/gen", "timing_s/update_actor", "timing_s/update_weights", "timing_s/save_checkpoint",
        "actor/perf/max_memory_allocated_gb", "actor/perf/max_memory_reserved_gb",
        "training/rollout_actor_probs_pearson_corr", "training/rollout_probs_diff_valid",
        "response_length/mean", "prompt_length/mean", "response/aborted_ratio",
    ]
    series = []
    for row in ordered_rows:
        metrics = dict(row.get("metrics") or {})
        series.append({
            "global_step": int(row.get("global_step")),
            "timestamp_utc": row.get("timestamp_utc"),
            "selected": {key: metrics.get(key) for key in selected},
            "all": metrics,
        })
    result = dict(series[-1])
    result["num_steps"] = len(series)
    result["series"] = series
    result["trends"] = metric_trends(series)
    return result


def latest_joint_step(root: Path) -> dict[str, Any]:
    candidates: list[tuple[int, Path]] = []
    for path in (root / "round_1" / "joint_steps").glob("joint_step_*.json"):
        match = re.fullmatch(r"joint_step_(\d+)\.json", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return load_json(max(candidates)[1]) if candidates else {}


def expected_orchestra_rows(root: Path, num_steps: int) -> int | None:
    launch = load_env_file(root / "launch_config.env")
    try:
        return int(launch["BCP_TRAIN_BATCH_SIZE"]) * int(launch["BCP_ROLLOUT_N"]) * num_steps
    except (KeyError, TypeError, ValueError):
        pass
    joint = latest_joint_step(root)
    try:
        return int(joint["canonical_pair_count"]) * num_steps
    except (KeyError, TypeError, ValueError):
        return None


def summarize_step_signals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose whether each optimizer step contains useful within-group contrast."""
    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        try:
            step = int(row.get("optimizer_step"))
        except (TypeError, ValueError):
            continue
        by_step[step].append(row)

    summaries: list[dict[str, Any]] = []
    for step, step_rows in sorted(by_step.items()):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in step_rows:
            groups[str(row.get("grpo_group_uid") or "missing")].append(row)
        reward_variances: list[float] = []
        reward_different = mas_different = diagnosis_different = 0
        for group_rows in groups.values():
            scores = [finite(row.get("score")) for row in group_rows]
            clean_scores = [value for value in scores if value is not None]
            variance = population_variance(clean_scores)
            if variance is not None:
                reward_variances.append(variance)
            reward_different += len(set(clean_scores)) > 1
            mas_different += len({canonical(row.get("src_mas")) for row in group_rows}) > 1
            diagnoses = []
            for row in group_rows:
                edit = row.get("edit_operation") or {}
                diagnosis = edit.get("diagnosis") or edit.get("stage1_localization") or {}
                diagnoses.append(canonical({
                    "target_type": diagnosis.get("target_type"),
                    "target_id": diagnosis.get("target_id"),
                    "failure_type": diagnosis.get("failure_type"),
                }))
            diagnosis_different += len(set(diagnoses)) > 1

        transitions = Counter(str(row.get("editor_outcome") or "missing") for row in step_rows)
        orchestra_outcomes = Counter(str(row.get("outcome") or "missing") for row in step_rows)
        failure_types = Counter()
        operations = Counter()
        applied_rows = structural_rows = 0
        for row in step_rows:
            edit = row.get("edit_operation") or {}
            diagnosis = edit.get("diagnosis") or edit.get("stage1_localization") or {}
            translation = edit.get("translation") or {}
            failure_types[str(diagnosis.get("failure_type") or "missing")] += 1
            applied = translation.get("applied_operations") or []
            applied_rows += bool(applied)
            for operation in applied:
                operations[str(operation.get("op") or "missing")] += 1
            structural_rows += canonical(row.get("src_mas")) != canonical(row.get("updated_mas"))

        group_count = len(groups)
        summaries.append({
            "optimizer_step": step,
            "rows": len(step_rows),
            "groups": group_count,
            "group_size_distribution": counter_dict(Counter(len(value) for value in groups.values())),
            "reward_different_groups": reward_different,
            "reward_difference_rate": reward_different / group_count if group_count else None,
            "mas_different_groups": mas_different,
            "mas_difference_rate": mas_different / group_count if group_count else None,
            "diagnosis_different_groups": diagnosis_different,
            "diagnosis_difference_rate": diagnosis_different / group_count if group_count else None,
            "group_reward_variance": distribution(reward_variances),
            "source_score": distribution(row.get("src_score") for row in step_rows),
            "updated_score": distribution(row.get("upd_score") for row in step_rows),
            "score_delta": distribution(row.get("score_delta") for row in step_rows),
            "orchestra_reward": distribution(row.get("score") for row in step_rows),
            "refiner_reward": distribution(row.get("editor_score") for row in step_rows),
            "editor_outcomes": counter_dict(transitions),
            "orchestra_outcomes": counter_dict(orchestra_outcomes),
            "applied_rows": applied_rows,
            "structural_change_rows": structural_rows,
            "failure_type": counter_dict(failure_types),
            "failure_type_dominance": dominance(failure_types),
            "applied_operation_types": counter_dict(operations),
            "applied_operation_dominance": dominance(operations),
        })
    return summaries


def refiner_funnel_summary(
    strict_nested: Counter[str],
    *,
    raw_marginal_counts: dict[str, int],
) -> dict[str, Any]:
    """Report raw stage availability separately from the strict nested path.

    A strict funnel intentionally counts a patch only when every preceding
    grounding gate passed.  Without raw marginals, a grounding failure can
    misleadingly make thousands of actually applied patches appear as zero.
    """
    strict_keys = (
        "total_rows", "xml_valid", "diagnosis_grounded_proxy",
        "patch_applied", "structural_change", "score_improved", "score_damaged",
    )
    strict_counts = {key: int(strict_nested.get(key, 0)) for key in strict_keys}
    total = strict_counts["total_rows"]
    return {
        # Compatibility alias retained for downstream consumers of schema v4.
        "counts": strict_counts,
        "strict_nested_counts": strict_counts,
        "raw_marginal_counts": {key: int(value) for key, value in raw_marginal_counts.items()},
        "rates_over_total": {
            key: strict_counts[key] / total if total else None
            for key in strict_keys if key != "total_rows"
        },
        "stage_retention": {
            "diagnosis_given_xml": strict_counts["diagnosis_grounded_proxy"] / strict_counts["xml_valid"]
            if strict_counts["xml_valid"] else None,
            "patch_given_diagnosis": strict_counts["patch_applied"] / strict_counts["diagnosis_grounded_proxy"]
            if strict_counts["diagnosis_grounded_proxy"] else None,
            "change_given_patch": strict_counts["structural_change"] / strict_counts["patch_applied"]
            if strict_counts["patch_applied"] else None,
            "improvement_given_change": strict_counts["score_improved"] / strict_counts["structural_change"]
            if strict_counts["structural_change"] else None,
            "damage_given_change": strict_counts["score_damaged"] / strict_counts["structural_change"]
            if strict_counts["structural_change"] else None,
        },
        "interpretation": (
            "raw_marginal_counts are independent observed stage totals; "
            "strict_nested_counts require every preceding grounding gate and therefore diagnose where "
            "the causal-quality path collapses"
        ),
    }


def summarize_training(run: str, root: Path) -> dict[str, Any]:
    pattern = str(root / "round_1" / "orchestra" / "rollouts" / "canonical_rollouts" / "*.jsonl")
    rows = read_jsonl_glob(pattern)
    if not rows:
        raise RuntimeError(f"no canonical rollouts for run {run}: {pattern}")
    pair_ids = [str(row.get("pair_id") or "") for row in rows]
    if len(set(pair_ids)) != len(rows):
        raise RuntimeError(f"duplicate canonical pair ids in run {run}")

    actor_xml = Counter()
    refiner_xml = Counter()
    repairs = Counter()
    translation_status = Counter()
    requested_ops = Counter()
    canonical_ops = Counter()
    applied_ops = Counter()
    diagnosis_target_type = Counter()
    diagnosis_target_id = Counter()
    failure_type = Counter()
    source_status = Counter()
    outcomes = Counter()
    editor_outcomes = Counter()
    warning_types = Counter()
    placeholder_hits = Counter()
    retry_counts = Counter()
    group_rewards: dict[str, list[float]] = defaultdict(list)
    group_source_mas: dict[str, set[str]] = defaultdict(set)
    group_source_pairs: dict[str, set[str]] = defaultdict(set)

    requested_count = canonical_count = applied_count = noop_count = dropped_count = 0
    requested_rows = applied_rows = noop_rows = dropped_rows = 0
    semantic_preserved = structural_changes = updated_success = source_success = 0
    success_transitions = Counter()
    partial_translation = unrecoverable_translation = 0
    error_rows = violation_rows = repaired_actor = repaired_refiner = 0
    generation_seconds: list[float] = []
    source_elapsed: list[float] = []
    updated_elapsed: list[float] = []
    diagnosis_rows = target_resolved_rows = evidence_present_rows = 0
    evidence_location_resolved_rows = evidence_nonplaceholder_rows = 0
    diagnosis_patch_aligned_rows = 0
    refiner_funnel: Counter[str] = Counter()

    for row in rows:
        ax = classify_structured_output(row.get("actor_output"), "mas")
        rx = classify_structured_output(row.get("peer_output"), "refiner")
        actor_xml[ax["classification"]] += 1
        refiner_xml[rx["classification"]] += 1
        for repair in ax["repairs"]:
            repairs[f"actor:{repair}"] += 1
        for repair in rx["repairs"]:
            repairs[f"refiner:{repair}"] += 1
        repaired_actor += ax["classification"] == "repaired_xml"
        repaired_refiner += rx["classification"] == "repaired_xml"

        edit = row.get("edit_operation") or {}
        diagnosis = edit.get("diagnosis") or edit.get("stage1_localization") or {}
        translation = edit.get("translation") or {}
        diagnosis_rows += bool(diagnosis)
        target_resolved_rows += diagnosis_target_resolves(diagnosis, row.get("src_mas"))
        evidence = diagnosis.get("evidence") if isinstance(diagnosis, dict) else []
        evidence = evidence if isinstance(evidence, list) else []
        evidence_present_rows += bool(evidence)
        evidence_location_resolved_rows += bool(evidence) and all(
            isinstance(item, dict) and evidence_location_resolves(item.get("location"), row.get("src_mas"))
            for item in evidence
        )
        evidence_nonplaceholder_rows += bool(evidence) and all(
            isinstance(item, dict)
            and bool(str(item.get("observation") or "").strip())
            and not _contains_placeholder(item.get("observation"))
            for item in evidence
        )
        translation_status[str(translation.get("status") or "missing")] += 1
        req = translation.get("requested_operations") or []
        can = translation.get("canonical_operations") or []
        app = translation.get("applied_operations") or []
        diagnosis_patch_aligned_rows += diagnosis_patch_aligned(diagnosis, app)
        noop = translation.get("noop_operations") or []
        dropped = translation.get("dropped_operations") or []
        requested_count += len(req)
        canonical_count += len(can)
        applied_count += len(app)
        noop_count += len(noop)
        dropped_count += len(dropped)
        requested_rows += bool(req)
        applied_rows += bool(app)
        noop_rows += bool(noop)
        dropped_rows += bool(dropped)
        semantic_preserved += bool(translation.get("semantic_intent_preserved"))
        partial_translation += bool(app and dropped)
        unrecoverable_translation += bool(req and not can and not app and not noop)
        for operation in req:
            requested_ops[str(operation.get("op") or "missing")] += 1
        for operation in can:
            canonical_ops[str(operation.get("op") or "missing")] += 1
        for operation in app:
            applied_ops[str(operation.get("op") or "missing")] += 1
        for warning in translation.get("warnings") or []:
            warning_types[str(warning).split(":", 1)[0]] += 1

        diagnosis_target_type[str(diagnosis.get("target_type") or "missing")] += 1
        diagnosis_target_id[str(diagnosis.get("target_id") or "missing")] += 1
        failure_type[str(diagnosis.get("failure_type") or "missing")] += 1
        source_status[str(diagnosis.get("source_status") or "missing")] += 1
        outcomes[str(row.get("outcome") or "missing")] += 1
        editor_outcomes[str(row.get("editor_outcome") or "missing")] += 1
        peer_lower = str(row.get("peer_output") or "").lower()
        for phrase in PLACEHOLDER_PATTERNS:
            if phrase in peer_lower:
                placeholder_hits[phrase] += 1
        retry_counts[str((row.get("peer_retry") or {}).get("retry_count", "missing"))] += 1

        structural_changed = canonical(row.get("src_mas")) != canonical(row.get("updated_mas"))
        structural_changes += structural_changed
        xml_valid = rx["classification"] != "unrecoverable"
        target_resolved = diagnosis_target_resolves(diagnosis, row.get("src_mas"))
        evidence_items = diagnosis.get("evidence") if isinstance(diagnosis, dict) else []
        evidence_items = evidence_items if isinstance(evidence_items, list) else []
        evidence_grounded_proxy = bool(evidence_items) and all(
            isinstance(item, dict)
            and evidence_location_resolves(item.get("location"), row.get("src_mas"))
            and bool(str(item.get("observation") or "").strip())
            and not _contains_placeholder(item.get("observation"))
            for item in evidence_items
        )
        diagnosis_valid_proxy = target_resolved and evidence_grounded_proxy
        patch_applied = bool(app)
        delta_value = finite(row.get("score_delta"))
        improved = delta_value is not None and delta_value > 0
        damaged = delta_value is not None and delta_value < 0
        refiner_funnel["total_rows"] += 1
        if xml_valid:
            refiner_funnel["xml_valid"] += 1
            if diagnosis_valid_proxy:
                refiner_funnel["diagnosis_grounded_proxy"] += 1
                if patch_applied:
                    refiner_funnel["patch_applied"] += 1
                    if structural_changed:
                        refiner_funnel["structural_change"] += 1
                        if improved:
                            refiner_funnel["score_improved"] += 1
                        if damaged:
                            refiner_funnel["score_damaged"] += 1
        src_ok = bool((row.get("src_trajectory") or {}).get("success"))
        upd_ok = bool((row.get("updated_trajectory") or {}).get("success"))
        source_success += src_ok
        updated_success += upd_ok
        success_transitions[
            "fail_to_success" if not src_ok and upd_ok else
            "success_to_fail" if src_ok and not upd_ok else
            "both_success" if src_ok else "both_fail"
        ] += 1
        error_rows += bool(row.get("error"))
        violation_rows += bool(row.get("violations"))
        generation_seconds.append(row.get("actor_generate_seconds"))
        source_elapsed.append((row.get("src_trajectory") or {}).get("elapsed_sec"))
        updated_elapsed.append((row.get("updated_trajectory") or {}).get("elapsed_sec"))
        group_uid = str(row.get("grpo_group_uid") or "missing")
        group_source_mas[group_uid].add(canonical(row.get("src_mas")))
        group_source_pairs[group_uid].add(str(row.get("pair_id") or row.get("id") or "missing"))
        reward = finite(row.get("score"))
        if reward is not None:
            group_rewards[group_uid].append(reward)

    group_variances = {key: population_variance(values) for key, values in group_rewards.items()}
    group_sizes = Counter(len(values) for values in group_rewards.values())
    validations = {}
    for role in ("orchestra", "editor"):
        path = root / "round_1" / role / "artifact_validation.json"
        validations[role] = load_json(path) if path.is_file() else {}

    deltas = [finite(row.get("score_delta")) for row in rows]
    deltas_clean = [value for value in deltas if value is not None]
    training_metrics = {role: training_metric(root, role) for role in ("orchestra", "editor")}
    orchestra_steps = int(training_metrics["orchestra"].get("num_steps") or 0)
    expected_rows = expected_orchestra_rows(root, orchestra_steps)
    result = {
        "run": run,
        "root": str(root),
        "joint_step": latest_joint_step(root),
        "artifact_validation": validations,
        "rows": len(rows),
        "expected_rows": expected_rows,
        "row_count_valid": expected_rows is None or len(rows) == expected_rows,
        "unique_pairs": len(set(pair_ids)),
        "unique_tasks": len({str(row.get("dataset_sample_id")) for row in rows}),
        "unique_groups": len(group_rewards),
        "group_sizes": counter_dict(group_sizes),
        "editor_group_credit_assignment": {
            "task_groups": len(group_source_mas),
            "groups_with_multiple_source_mas": sum(len(values) > 1 for values in group_source_mas.values()),
            "groups_with_single_source_mas": sum(len(values) == 1 for values in group_source_mas.values()),
            "multiple_source_mas_rate": (
                sum(len(values) > 1 for values in group_source_mas.values()) / len(group_source_mas)
                if group_source_mas else None
            ),
            "unique_source_mas_per_group": distribution(len(values) for values in group_source_mas.values()),
            "unique_source_pairs_per_group": distribution(len(values) for values in group_source_pairs.values()),
            "inferred_semantics": "task_conditioned_source_pairs",
            "credit_assignment_caveat": (
                "each Editor GRPO group compares Refiner actions attached to different source MAS rollouts; "
                "reward ranking therefore mixes source context/difficulty with repair quality"
            ),
        },
        "group_reward_variance": {
            "mean": mean(group_variances.values()),
            "min": min((x for x in group_variances.values() if x is not None), default=None),
            "max": max((x for x in group_variances.values() if x is not None), default=None),
            "zero_variance_groups": sum(x == 0 for x in group_variances.values()),
            "nonzero_variance_groups": sum(x is not None and x > 0 for x in group_variances.values()),
            "difference_rate": (
                sum(x is not None and x > 0 for x in group_variances.values()) / len(group_variances)
                if group_variances else None
            ),
        },
        "reward_distributions": {
            "source_score": distribution(row.get("src_score") for row in rows),
            "updated_score": distribution(row.get("upd_score") for row in rows),
            "score_delta": distribution(row.get("score_delta") for row in rows),
            "orchestra_reward": distribution(row.get("score") for row in rows),
            "refiner_reward": distribution(row.get("editor_score") for row in rows),
        },
        "per_step_signals": summarize_step_signals(rows),
        "rewards": {
            "source_score_mean": mean(row.get("src_score") for row in rows),
            "updated_score_mean": mean(row.get("upd_score") for row in rows),
            "delta_mean": mean(deltas_clean),
            "delta_min": min(deltas_clean, default=None),
            "delta_max": max(deltas_clean, default=None),
            "orchestra_adversarial_reward_mean": mean(row.get("score") for row in rows),
            "refiner_reward_mean": mean(row.get("editor_score") for row in rows),
        },
        "transitions": counter_dict(editor_outcomes),
        "transition_dominance": dominance(editor_outcomes),
        "orchestra_outcomes": counter_dict(outcomes),
        "xml": {
            "actor": counter_dict(actor_xml),
            "refiner": counter_dict(refiner_xml),
            "repair_types": counter_dict(repairs),
            "repaired_actor": repaired_actor,
            "repaired_refiner": repaired_refiner,
        },
        "patch_translation": {
            "status": counter_dict(translation_status),
            "requested_operations": requested_count,
            "canonical_operations": canonical_count,
            "applied_operations": applied_count,
            "noop_operations": noop_count,
            "dropped_operations": dropped_count,
            "requested_rows": requested_rows,
            "applied_rows": applied_rows,
            "noop_rows": noop_rows,
            "dropped_rows": dropped_rows,
            "partial_rows": partial_translation,
            "untranslatable_rows": unrecoverable_translation,
            "semantic_intent_preserved_rows": semantic_preserved,
            "requested_operation_types": counter_dict(requested_ops),
            "canonical_operation_types": counter_dict(canonical_ops),
            "applied_operation_types": counter_dict(applied_ops),
            "warning_types": counter_dict(warning_types),
        },
        "diagnosis": {
            "target_type": counter_dict(diagnosis_target_type),
            "target_id": counter_dict(diagnosis_target_id),
            "failure_type": counter_dict(failure_type),
            "source_status": counter_dict(source_status),
            "placeholder_phrase_rows": counter_dict(placeholder_hits),
            "distinct_target_types": len(diagnosis_target_type),
            "distinct_failure_types": len(failure_type),
            "distinct_applied_operation_types": len(applied_ops),
            "failure_type_dominance": dominance(failure_type),
            "target_id_dominance": dominance(diagnosis_target_id),
            "applied_operation_dominance": dominance(applied_ops),
            "quality_proxy": {
                "diagnosis_rows": diagnosis_rows,
                "target_resolved_rows": target_resolved_rows,
                "target_resolved_rate": target_resolved_rows / diagnosis_rows if diagnosis_rows else None,
                "evidence_present_rows": evidence_present_rows,
                "evidence_present_rate": evidence_present_rows / diagnosis_rows if diagnosis_rows else None,
                "evidence_location_resolved_rows": evidence_location_resolved_rows,
                "evidence_location_resolved_rate": evidence_location_resolved_rows / diagnosis_rows if diagnosis_rows else None,
                "evidence_nonplaceholder_rows": evidence_nonplaceholder_rows,
                "evidence_nonplaceholder_rate": evidence_nonplaceholder_rows / diagnosis_rows if diagnosis_rows else None,
                "diagnosis_patch_aligned_rows": diagnosis_patch_aligned_rows,
                "diagnosis_patch_aligned_rate": diagnosis_patch_aligned_rows / diagnosis_rows if diagnosis_rows else None,
                "interpretation": "structural/lexical proxy only; not proof of causal correctness",
            },
        },
        "refiner_funnel": refiner_funnel_summary(
            refiner_funnel,
            raw_marginal_counts={
                "total_rows": len(rows),
                "xml_valid": int(refiner_funnel.get("xml_valid", 0)),
                "target_resolved_proxy": target_resolved_rows,
                "evidence_location_resolved_proxy": evidence_location_resolved_rows,
                "evidence_nonplaceholder_proxy": evidence_nonplaceholder_rows,
                "patch_applied": applied_rows,
                "structural_change": structural_changes,
                "score_improved": sum(value > 0 for value in deltas_clean),
                "score_damaged": sum(value < 0 for value in deltas_clean),
            },
        ),
        "execution": {
            "source_success_rows": source_success,
            "updated_success_rows": updated_success,
            "success_transitions": counter_dict(success_transitions),
            "structural_change_rows": structural_changes,
            "error_rows": error_rows,
            "violation_rows": violation_rows,
            "retry_counts": counter_dict(retry_counts),
            "actor_generation_seconds_mean": mean(generation_seconds),
            "source_elapsed_seconds_mean": mean(source_elapsed),
            "updated_elapsed_seconds_mean": mean(updated_elapsed),
        },
        "training_metrics": training_metrics,
    }
    return result


def accuracy(summary: dict[str, Any], stage: str) -> float | None:
    value = summary.get(stage)
    return finite(value.get("accuracy")) if isinstance(value, dict) else None


def eval_label(job: dict[str, Any]) -> str:
    kind = str(job.get("kind") or "")
    if kind == "baseline":
        return "Base/Base"
    orchestra_step = job.get("orchestra_step")
    editor_step = job.get("editor_step")
    if kind == "same_step" and orchestra_step is not None and editor_step is not None:
        return f"O{int(orchestra_step)}/E{int(editor_step)}"
    if kind == "orchestra_step" and orchestra_step is not None:
        return f"O{int(orchestra_step)}/Base"
    if kind == "editor_step" and orchestra_step is not None and editor_step is not None:
        return f"O{int(orchestra_step)}/E{int(editor_step)}"
    return kind or "unknown"


def eval_cell_sort_key(cell: dict[str, Any]) -> tuple[int, int, int]:
    if cell.get("kind") == "baseline":
        return (0, -1, -1)
    return (
        1,
        int(cell.get("orchestra_step") or -1),
        int(cell.get("editor_step") or -1),
    )


def summarize_evaluation(
    run: str, root: Path, draws: int, allow_incomplete: bool,
    *, expected_total: int | None = None, expected_steps: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = root / "matrix_plan.json"
    if not plan_path.is_file():
        if allow_incomplete:
            return {"run": run, "root": str(root), "complete": False, "reason": "matrix_plan_missing", "cells": []}, []
        raise RuntimeError(f"evaluation plan missing for run {run}: {plan_path}")
    plan = load_json(plan_path)
    pairing_mode = str(plan.get("pairing_mode") or "")
    if pairing_mode != "same-step":
        raise RuntimeError(f"run {run} expected pairing_mode=same-step, found {pairing_mode!r}")
    jobs = [job for job in plan.get("jobs") or [] if job.get("dataset") == "bcp"]
    baseline_jobs = [job for job in jobs if job.get("kind") == "baseline"]
    same_step_jobs = [job for job in jobs if job.get("kind") == "same_step"]
    invalid_jobs = [job for job in jobs if job.get("kind") not in {"baseline", "same_step"}]
    if len(baseline_jobs) != 1 or not same_step_jobs or invalid_jobs:
        raise RuntimeError(
            f"run {run} invalid same-step plan: baseline={len(baseline_jobs)} "
            f"same_step={len(same_step_jobs)} invalid={len(invalid_jobs)}"
        )
    mismatched = [
        job for job in same_step_jobs
        if job.get("orchestra_step") is None
        or job.get("editor_step") is None
        or int(job["orchestra_step"]) != int(job["editor_step"])
    ]
    if mismatched:
        raise RuntimeError(f"run {run} contains mismatched same-step jobs: {mismatched[:3]}")
    actual_steps = sorted(int(job["orchestra_step"]) for job in same_step_jobs)
    if expected_steps is not None and actual_steps != list(range(1, expected_steps + 1)):
        raise RuntimeError(
            f"run {run} expected same-step checkpoints 1..{expected_steps}, found {actual_steps}"
        )
    labels = [eval_label(job) for job in jobs]
    if len(set(labels)) != len(labels):
        raise RuntimeError(f"run {run} has duplicate evaluation labels: {labels}")

    journal = latest_jsonl(root / "matrix_results.jsonl", "job_key")
    cells: list[dict[str, Any]] = []
    records_by_label: dict[str, dict[str, dict[str, Any]]] = {}
    incomplete: list[str] = []
    for job in jobs:
        label = eval_label(job)
        output_dir = Path(job["output_dir"])
        summary_path = output_dir / "summary.json"
        results_path = output_dir / "results.jsonl"
        jr = journal.get(str(job.get("job_key"))) or {}
        base_cell = {
            "run": run,
            "label": label,
            "kind": job.get("kind"),
            "orchestra_round": job.get("orchestra_round"),
            "orchestra_step": job.get("orchestra_step"),
            "editor_round": job.get("editor_round"),
            "editor_step": job.get("editor_step"),
            "output_dir": str(output_dir),
            "orchestra_source": job.get("orchestra_source"),
            "editor_source": job.get("editor_source"),
        }
        if not summary_path.is_file() or not results_path.is_file():
            incomplete.append(label)
            cells.append({**base_cell, "status": jr.get("status", "pending")})
            continue
        summary = load_json(summary_path)
        records = latest_jsonl(results_path, "sample_key")
        if not records:
            records = latest_jsonl(results_path, "sample_id")
        summary_total = int(summary.get("num_total", -1))
        record_count = len(records)
        exact_total = expected_total is None or (summary_total == expected_total and record_count == expected_total)
        ok = (
            summary_total > 0
            and int(summary.get("num_fully_completed", -1)) == summary_total
            and int(summary.get("num_with_errors", 1)) == 0
            and bool(records)
            and exact_total
        )
        if not ok:
            incomplete.append(label)
        records_by_label[label] = records
        orchestra_accuracy = accuracy(summary, "orchestra_only")
        pipeline_accuracy = accuracy(summary, "orchestra_editor")
        cells.append({
            **base_cell,
            "status": "success" if ok else "incomplete",
            "num_total": summary.get("num_total"),
            "record_count": record_count,
            "expected_total": expected_total,
            "exact_total_valid": exact_total,
            "num_fully_completed": summary.get("num_fully_completed"),
            "num_with_errors": summary.get("num_with_errors"),
            "orchestra_only_accuracy": orchestra_accuracy,
            "orchestra_editor_accuracy": pipeline_accuracy,
            "editor_increment": (
                pipeline_accuracy - orchestra_accuracy
                if pipeline_accuracy is not None and orchestra_accuracy is not None else None
            ),
            "orchestra_only_correct": (summary.get("orchestra_only") or {}).get("correct"),
            "orchestra_editor_correct": (summary.get("orchestra_editor") or {}).get("correct"),
        })
    cells.sort(key=eval_cell_sort_key)
    if incomplete and not allow_incomplete:
        raise RuntimeError(f"run {run} evaluation incomplete: {', '.join(incomplete)}")

    comparisons: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    ordered_labels = [cell["label"] for cell in cells]
    baseline_label = "Base/Base"
    comparison_specs: list[tuple[str, str, str, str, str, bool]] = []
    for label in ordered_labels:
        comparison_specs.append((
            label, label, "orchestra_only", "orchestra_editor",
            f"{label} within-cell Editor increment", False,
        ))
    trained_labels = [label for label in ordered_labels if label != baseline_label]
    for label in trained_labels:
        comparison_specs.extend([
            (baseline_label, label, "orchestra_only", "orchestra_only", f"{label} Orchestra vs Base", True),
            (baseline_label, label, "orchestra_editor", "orchestra_editor", f"{label} pipeline vs Base", True),
        ])
    for before, after in zip(ordered_labels, ordered_labels[1:]):
        comparison_specs.append((
            before, after, "orchestra_editor", "orchestra_editor",
            f"{after} pipeline vs previous {before}", True,
        ))

    for index, (before, after, before_stage, after_stage, label, confounded) in enumerate(comparison_specs):
        if before not in records_by_label or after not in records_by_label:
            continue
        stats, rows = compare_records(
            records_by_label[before], records_by_label[after], before_stage, after_stage,
            label=label, run=run, seed=20260806 + index * 101 + (0 if run == "A" else 10000), draws=draws,
        )
        stats["stochastic_upstream_confounding"] = confounded
        comparisons.append(stats)
        pair_rows.extend(rows)
    return {
        "run": run,
        "root": str(root),
        "complete": not incomplete,
        "incomplete_cells": incomplete,
        "pairing_mode": pairing_mode,
        "num_jobs": len(jobs),
        "expected_steps": actual_steps,
        "required_steps": list(range(1, expected_steps + 1)) if expected_steps is not None else None,
        "expected_total_per_cell": expected_total,
        "cells": cells,
        "comparisons": comparisons,
    }, pair_rows


def evaluation_label_sort_key(label: str) -> tuple[int, int, int]:
    """Keep Base first and same-step labels in numeric rather than lexical order."""
    if label == "Base/Base":
        return (0, -1, -1)
    match = re.fullmatch(r"O(\d+)/E(\d+)", label)
    if match:
        return (1, int(match.group(1)), int(match.group(2)))
    return (2, 0, 0)


def evaluation_learning_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    """Bridge training checkpoints to held-out behavior without treating format success as gain."""
    cells = [cell for cell in evaluation.get("cells", []) if cell.get("status") == "success"]
    cells.sort(key=lambda cell: evaluation_label_sort_key(str(cell.get("label") or "")))
    baseline = next((cell for cell in cells if cell.get("label") == "Base/Base"), None)
    trained = [cell for cell in cells if cell.get("label") != "Base/Base"]
    baseline_pipeline = finite((baseline or {}).get("orchestra_editor_accuracy"))
    baseline_orchestra = finite((baseline or {}).get("orchestra_only_accuracy"))

    def metric(cell: dict[str, Any], key: str) -> float | None:
        return finite(cell.get(key))

    pipeline_values = [value for value in (metric(cell, "orchestra_editor_accuracy") for cell in trained) if value is not None]
    orchestra_values = [value for value in (metric(cell, "orchestra_only_accuracy") for cell in trained) if value is not None]
    editor_increments = [value for value in (metric(cell, "editor_increment") for cell in trained) if value is not None]
    def pipeline_for_max(cell: dict[str, Any]) -> float:
        value = metric(cell, "orchestra_editor_accuracy")
        return value if value is not None else -math.inf

    def pipeline_for_min(cell: dict[str, Any]) -> float:
        value = metric(cell, "orchestra_editor_accuracy")
        return value if value is not None else math.inf

    best = max(trained, key=pipeline_for_max, default=None)
    worst = min(trained, key=pipeline_for_min, default=None)
    final = trained[-1] if trained else None

    comparisons = {str(item.get("comparison")): item for item in evaluation.get("comparisons", [])}
    positive_ci = negative_ci = 0
    editor_positive_ci = editor_negative_ci = 0
    for cell in trained:
        label = str(cell.get("label"))
        comp = comparisons.get(f"{label} pipeline vs Base") or {}
        low, high = finite(comp.get("bootstrap_95ci_low")), finite(comp.get("bootstrap_95ci_high"))
        positive_ci += low is not None and low > 0
        negative_ci += high is not None and high < 0
        editor_comp = comparisons.get(f"{label} within-cell Editor increment") or {}
        editor_low = finite(editor_comp.get("bootstrap_95ci_low"))
        editor_high = finite(editor_comp.get("bootstrap_95ci_high"))
        editor_positive_ci += editor_low is not None and editor_low > 0
        editor_negative_ci += editor_high is not None and editor_high < 0

    def cell_summary(cell: dict[str, Any] | None) -> dict[str, Any] | None:
        if cell is None:
            return None
        pipeline = metric(cell, "orchestra_editor_accuracy")
        orchestra = metric(cell, "orchestra_only_accuracy")
        return {
            "label": cell.get("label"),
            "orchestra_only_accuracy": orchestra,
            "orchestra_editor_accuracy": pipeline,
            "pipeline_delta_vs_base": (
                pipeline - baseline_pipeline
                if pipeline is not None and baseline_pipeline is not None else None
            ),
            "editor_increment": metric(cell, "editor_increment"),
        }

    return {
        "run": evaluation.get("run"),
        "complete": evaluation.get("complete"),
        "trained_checkpoint_count": len(trained),
        "baseline_orchestra_accuracy": baseline_orchestra,
        "baseline_pipeline_accuracy": baseline_pipeline,
        "trained_orchestra_accuracy_mean": mean(orchestra_values),
        "trained_pipeline_accuracy_mean": mean(pipeline_values),
        "trained_pipeline_mean_delta_vs_base": (
            mean(pipeline_values) - baseline_pipeline
            if pipeline_values and baseline_pipeline is not None else None
        ),
        "checkpoints_pipeline_above_base": sum(
            metric(cell, "orchestra_editor_accuracy") is not None
            and baseline_pipeline is not None
            and metric(cell, "orchestra_editor_accuracy") > baseline_pipeline
            for cell in trained
        ),
        "checkpoints_pipeline_below_base": sum(
            metric(cell, "orchestra_editor_accuracy") is not None
            and baseline_pipeline is not None
            and metric(cell, "orchestra_editor_accuracy") < baseline_pipeline
            for cell in trained
        ),
        "pipeline_vs_base_positive_95ci": positive_ci,
        "pipeline_vs_base_negative_95ci": negative_ci,
        "editor_increment_mean": mean(editor_increments),
        "editor_increment_positive_cells": sum(value > 0 for value in editor_increments),
        "editor_increment_negative_cells": sum(value < 0 for value in editor_increments),
        "editor_increment_positive_95ci": editor_positive_ci,
        "editor_increment_negative_95ci": editor_negative_ci,
        "best_checkpoint": cell_summary(best),
        "worst_checkpoint": cell_summary(worst),
        "final_checkpoint": cell_summary(final),
        "caveat": "checkpoint-vs-base comparisons include stochastic upstream generation; within-cell Editor increments are the cleaner Refiner effect",
    }


def learning_rate_comparison(
    training_runs: list[dict[str, Any]], evaluation_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    evaluations = {item.get("run"): item for item in evaluation_runs}
    rows: list[dict[str, Any]] = []
    for training in training_runs:
        run = training.get("run")
        transitions = training.get("transitions") or {}
        total = int(training.get("rows") or 0)
        groups = int(training.get("unique_groups") or 0)
        zero_groups = int((training.get("group_reward_variance") or {}).get("zero_variance_groups") or 0)
        quality = ((training.get("diagnosis") or {}).get("quality_proxy") or {})
        orchestra_metrics = ((training.get("training_metrics") or {}).get("orchestra") or {}).get("selected") or {}
        editor_metrics = ((training.get("training_metrics") or {}).get("editor") or {}).get("selected") or {}
        rows.append({
            "run": run,
            "orchestra_lr": finite(orchestra_metrics.get("actor/lr")),
            "editor_lr": finite(editor_metrics.get("actor/lr")),
            "training_improved_rate": int(transitions.get("improved", 0)) / total if total else None,
            "training_damaged_rate": int(transitions.get("damaged", 0)) / total if total else None,
            "training_no_gain_rate": int(transitions.get("no_gain", 0)) / total if total else None,
            "zero_reward_variance_group_rate": zero_groups / groups if groups else None,
            "evidence_nonplaceholder_rate": finite(quality.get("evidence_nonplaceholder_rate")),
            "held_out": evaluation_learning_summary(evaluations.get(run) or {"run": run}),
        })
    return {
        "runs": rows,
        "interpretation_rule": (
            "A higher LR is useful only if training signal changes survive held-out same-step evaluation; "
            "XML/patch diversity or lower placeholder copying alone is not evidence of task improvement."
        ),
    }


def cross_run(eval_a: dict[str, Any], eval_b: dict[str, Any]) -> dict[str, Any]:
    cells_a = {cell["label"]: cell for cell in eval_a.get("cells", []) if cell.get("status") == "success"}
    cells_b = {cell["label"]: cell for cell in eval_b.get("cells", []) if cell.get("status") == "success"}
    rows = []
    for label in sorted(set(cells_a) & set(cells_b), key=evaluation_label_sort_key):
        a, b = cells_a[label], cells_b[label]
        orchestra_a, orchestra_b = finite(a.get("orchestra_only_accuracy")), finite(b.get("orchestra_only_accuracy"))
        pipeline_a, pipeline_b = finite(a.get("orchestra_editor_accuracy")), finite(b.get("orchestra_editor_accuracy"))
        rows.append({
            "label": label,
            "orchestra_only_accuracy_A": orchestra_a,
            "orchestra_only_accuracy_B": orchestra_b,
            "orchestra_only_signed_B_minus_A": orchestra_b - orchestra_a if orchestra_a is not None and orchestra_b is not None else None,
            "orchestra_only_abs_gap": abs(orchestra_a - orchestra_b) if orchestra_a is not None and orchestra_b is not None else None,
            "orchestra_editor_accuracy_A": pipeline_a,
            "orchestra_editor_accuracy_B": pipeline_b,
            "orchestra_editor_signed_B_minus_A": pipeline_b - pipeline_a if pipeline_a is not None and pipeline_b is not None else None,
            "orchestra_editor_abs_gap": abs(pipeline_a - pipeline_b) if pipeline_a is not None and pipeline_b is not None else None,
        })
    return {
        "cells": rows,
        "mean_orchestra_only_signed_B_minus_A": mean(row["orchestra_only_signed_B_minus_A"] for row in rows),
        "mean_orchestra_editor_signed_B_minus_A": mean(row["orchestra_editor_signed_B_minus_A"] for row in rows),
        "mean_orchestra_only_abs_gap": mean(row["orchestra_only_abs_gap"] for row in rows),
        "mean_orchestra_editor_abs_gap": mean(row["orchestra_editor_abs_gap"] for row in rows),
    }


# Backward-compatible alias for older callers; the current experiment compares LRs, not seeds.
cross_seed = cross_run

def pct(value: Any, denominator: Any) -> str:
    value_f, denominator_f = finite(value), finite(denominator)
    return "n/a" if value_f is None or not denominator_f else f"{100 * value_f / denominator_f:.2f}%"


def fmt(value: Any, digits: int = 6) -> str:
    value_f = finite(value)
    return "n/a" if value_f is None else f"{value_f:.{digits}f}"


def audit_validity(payload: Any) -> str:
    """Render heterogeneous audit schemas without assuming one validator implementation."""
    if not isinstance(payload, dict):
        return "unknown"
    for key in ("valid", "ok", "passed", "success", "complete"):
        if key in payload:
            return str(bool(payload.get(key))).lower()
    status = payload.get("status")
    if status is not None:
        return str(status)
    integrity = payload.get("integrity")
    if isinstance(integrity, dict):
        for key in ("valid", "ok", "passed", "success", "complete"):
            if key in integrity:
                return str(bool(integrity.get(key))).lower()
    errors = payload.get("errors")
    if isinstance(errors, list):
        return "true" if not errors else f"false ({len(errors)} errors)"
    return "present"


def root_cause_findings(report: dict[str, Any]) -> list[str]:
    """Build findings from this run's evidence instead of historical assumptions."""
    findings: list[str] = []
    runs = report["training_runs"]
    total = sum(run["rows"] for run in runs)
    xml_bad = sum(
        run["xml"][role].get("unrecoverable", 0)
        for run in runs for role in ("actor", "refiner")
    )
    no_gain = sum(run["transitions"].get("no_gain", 0) for run in runs)
    applied_rows = sum(run["patch_translation"].get("applied_rows", 0) for run in runs)
    structural = sum(run["execution"].get("structural_change_rows", 0) for run in runs)
    groups = sum(run.get("unique_groups", 0) for run in runs)
    zero_groups = sum(run["group_reward_variance"].get("zero_variance_groups", 0) for run in runs)
    findings.append(
        f"**结构化链路漏斗。** actor/refiner 合计不可恢复 XML={xml_bad}/{2 * total} "
        f"({pct(xml_bad, 2 * total)})；patch applied rows={applied_rows}/{total} ({pct(applied_rows, total)})；"
        f"structural-change rows={structural}/{total} ({pct(structural, total)})。"
    )
    findings.append(
        f"**终局改进信号密度。** no-gain={no_gain}/{total} ({pct(no_gain, total)})；"
        f"零方差 GRPO groups={zero_groups}/{groups} ({pct(zero_groups, groups)})。"
        "零方差比例越高，单纯增加学习率越难产生有效相对优势。"
    )
    for run in runs:
        diag = run["diagnosis"]
        failure = diag["failure_type_dominance"]
        operation = diag["applied_operation_dominance"]
        findings.append(
            f"**Run {run['run']} 诊断/动作集中度。** failure top={failure['top']} "
            f"({pct(failure['top_count'], failure['total'])})；operation top={operation['top']} "
            f"({pct(operation['top_count'], operation['total'])})。"
            "高集中度意味着 group 表面文本不同但诊断机制可能仍同质。"
        )
        group_credit = run.get("editor_group_credit_assignment") or {}
        findings.append(
            f"**Run {run['run']} Editor credit assignment。** multi-source groups="
            f"{group_credit.get('groups_with_multiple_source_mas')}/{group_credit.get('task_groups')} "
            f"({fmt(group_credit.get('multiple_source_mas_rate'), 4)})；"
            f"unique source MAS/group={group_credit.get('unique_source_mas_per_group')}。"
            "当前每组排序混合了不同 source MAS 的上下文/难度与 Refiner 修复质量，"
            "因此 group 内 reward 有差异也不等于定位/patch 的因果 credit 干净。"
        )
        funnel = run.get("refiner_funnel") or {}
        findings.append(
            f"**Run {run['run']} Refiner 漏斗。** raw marginals={funnel.get('raw_marginal_counts')}；"
            f"strict nested={funnel.get('strict_nested_counts', funnel.get('counts'))}；"
            f"stage retention={funnel.get('stage_retention')}。主要掉点应依据最低 retention 阶段定位，"
            "raw patch 数表示代码实际执行量，strict patch 数则要求前置 diagnosis grounding 同时通过；"
            "两者不能混为一谈，也不能把 XML 成功率当成优化成功率。"
        )
        quality = diag.get("quality_proxy") or {}
        findings.append(
            f"**Run {run['run']} 故障定位 grounding。** evidence non-placeholder="
            f"{quality.get('evidence_nonplaceholder_rows')}/{quality.get('diagnosis_rows')} "
            f"({fmt(quality.get('evidence_nonplaceholder_rate'), 4)})；"
            f"target resolved={fmt(quality.get('target_resolved_rate'), 4)}；"
            f"diagnosis-patch aligned={fmt(quality.get('diagnosis_patch_aligned_rate'), 4)}。"
            "这些仍是结构代理，但能直接揭示 Refiner 是否只复制模板占位内容。"
        )
        for role in ("orchestra", "editor"):
            trends = run["training_metrics"][role].get("trends") or {}
            reward = trends.get("critic/rewards/mean") or {}
            kl = trends.get("actor/ppo_kl") or trends.get("actor/kl_loss") or {}
            grad = trends.get("actor/grad_norm") or {}
            findings.append(
                f"**Run {run['run']} {role} 更新轨迹。** reward mean first→last="
                f"{fmt(reward.get('first'))}→{fmt(reward.get('last'))}；KL first→last="
                f"{fmt(kl.get('first'))}→{fmt(kl.get('last'))}；"
                f"gradient norm range={fmt(grad.get('min'))}–{fmt(grad.get('max'))}。"
            )
    findings.append(
        "**因果解释边界。** 同 cell 的 Orchestra-only→Orchestra+Editor 是最干净的 Refiner 增量；"
        "跨 checkpoint/cross-run 对比仍受生成随机性影响，必须与 bootstrap CI、McNemar 和两档 LR 的一致趋势联合解释。"
    )
    return findings


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MAS-Loop 双 8 卡双学习率训练与 same-step 配对评测 eReport",
        "",
        "## 1. 结论摘要",
        "",
        "本报告严格区分：**XML 格式有效**、**patch 可翻译/可应用**、**MAS 确实发生结构或语义变化**、以及 **任务得分真实提升**。前三者成功不等于最后一项成功。",
        "",
    ]
    for train in report["training_runs"]:
        rows = train["rows"]
        trans = train["transitions"]
        patch = train["patch_translation"]
        diag = train["diagnosis"]
        lines.extend([
            f"- Run {train['run']}: XML actor/refiner 原生或可修复输出分别为 "
            f"{rows - train['xml']['actor'].get('unrecoverable', 0)}/{rows}、"
            f"{rows - train['xml']['refiner'].get('unrecoverable', 0)}/{rows}；"
            f"patch applied={patch['applied_operations']}、dropped={patch['dropped_operations']}；"
            f"improved={trans.get('improved', 0)}、damaged={trans.get('damaged', 0)}、no_gain={trans.get('no_gain', 0)}。",
            f"  诊断 failure_type={diag['failure_type']}，applied operation={patch['applied_operation_types']}。",
        ])
    complete = all(item.get("complete") for item in report["evaluation_runs"])
    planned_cells = sum(len(item.get("cells", [])) for item in report["evaluation_runs"])
    successful_cells = sum(
        cell.get("status") == "success"
        for item in report["evaluation_runs"] for cell in item.get("cells", [])
    )
    lines.append(
        f"- same-step cells 完整状态：**{'全部完成' if complete else '尚未全部完成'}**；"
        f"成功 {successful_cells}/{planned_cells}。"
    )
    lines.extend(["", "## 2. 当前训练流程与主要超参数", ""])
    joint = report["training_runs"][0]["joint_step"]
    lines.extend([
        f"- 两套独立 8 卡训练：A seed={report['training_runs'][0]['joint_step'].get('data_seed')} / GPUs={report['training_runs'][0]['joint_step'].get('gpu_namespace')}；"
        f"B seed={report['training_runs'][1]['joint_step'].get('data_seed')} / GPUs={report['training_runs'][1]['joint_step'].get('gpu_namespace')}。两者 seed 相同，主要变量是学习率。",
        f"- 每个 task 的 Orchestra GRPO group size={joint.get('task_group_size')}；每轮 source/updated executions="
        f"{joint.get('source_executions')}/{joint.get('updated_executions')}。",
        "- Orchestra token target 始终是 source `<mas>`；updated MAS 仅进入 adversarial reward，不替换 Orchestra rollout token。",
        f"- Reward: Refiner = updated_score - source_score；Orchestra = source_score - "
        f"{joint.get('orchestra_adversarial_penalty')} * max(delta, 0)。",
        "- Refiner 学习 `<refiner>` 中的三阶段诊断、因果建模、修复计划和 `<patch>`；代码解析 patch 并编辑 source MAS。",
        f"- 结构化协议={joint.get('structured_output_protocol')} / {joint.get('xml_protocol_version')}；patch compiler={joint.get('patch_compiler_version')}。",
    ])
    for train in report["training_runs"]:
        lines.extend(["", f"### Run {train['run']} optimizer metrics", ""])
        orchestra_selected = train["training_metrics"]["orchestra"].get("selected") or {}
        editor_selected = train["training_metrics"]["editor"].get("selected") or {}
        step_config = train.get("joint_step") or {}
        lines.append(
            f"- LR: Orchestra={fmt(orchestra_selected.get('actor/lr'), 8)}，"
            f"Editor={fmt(editor_selected.get('actor/lr'), 8)}；"
            f"task_group_size={step_config.get('task_group_size')}，"
            f"canonical_pair_count={step_config.get('canonical_pair_count')}，"
            f"total_steps={step_config.get('total_steps')}。"
        )
        lines.append(
            f"- 已记录 Orchestra/Editor optimizer steps="
            f"{train['training_metrics']['orchestra'].get('num_steps', 0)}/"
            f"{train['training_metrics']['editor'].get('num_steps', 0)}；"
            f"canonical Orchestra rows={train['rows']}/{train.get('expected_rows')}。"
        )
        lines.append("| role | lr | loss | entropy | grad norm | PPO KL | reward mean | time/step(s) | generation(s) | throughput |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for role in ("orchestra", "editor"):
            m = train["training_metrics"][role].get("selected") or {}
            lines.append(
                f"| {role} | {fmt(m.get('actor/lr'), 8)} | {fmt(m.get('actor/loss'))} | {fmt(m.get('actor/entropy'))} | "
                f"{fmt(m.get('actor/grad_norm'))} | {fmt(m.get('actor/ppo_kl'))} | {fmt(m.get('critic/rewards/mean'))} | "
                f"{fmt(m.get('perf/time_per_step'), 2)} | {fmt(m.get('timing_s/gen'), 2)} | {fmt(m.get('perf/throughput'), 2)} |"
            )
    lines.extend(["", "## 3. 训练中间信号", ""])
    for train in report["training_runs"]:
        reward, execution, patch, diag = train["rewards"], train["execution"], train["patch_translation"], train["diagnosis"]
        lines.extend([
            f"### Run {train['run']}", "",
            f"- source/upd/delta/adversarial reward mean = {fmt(reward['source_score_mean'])} / {fmt(reward['updated_score_mean'])} / "
            f"{fmt(reward['delta_mean'])} / {fmt(reward['orchestra_adversarial_reward_mean'])}。",
            f"- transitions={train['transitions']}；Orchestra outcomes={train['orchestra_outcomes']}。",
            f"- group reward variance mean={fmt(train['group_reward_variance']['mean'])}；zero-variance groups="
            f"{train['group_reward_variance']['zero_variance_groups']}/{train['unique_groups']}；"
            f"reward-different group rate={fmt(train['group_reward_variance'].get('difference_rate'))}。",
            f"- Editor group source semantics={train.get('editor_group_credit_assignment')}。"
            "这里按真实 source MAS identity 推断，不采用历史 parquet 中可能误标的 same-source 标签。",
            f"- score delta distribution p10/p50/p90="
            f"{fmt(train['reward_distributions']['score_delta'].get('p10'))}/"
            f"{fmt(train['reward_distributions']['score_delta'].get('p50'))}/"
            f"{fmt(train['reward_distributions']['score_delta'].get('p90'))}；"
            f"Refiner reward p10/p50/p90="
            f"{fmt(train['reward_distributions']['refiner_reward'].get('p10'))}/"
            f"{fmt(train['reward_distributions']['refiner_reward'].get('p50'))}/"
            f"{fmt(train['reward_distributions']['refiner_reward'].get('p90'))}。",
            f"- XML actor={train['xml']['actor']}；Refiner={train['xml']['refiner']}；repairs={train['xml']['repair_types']}。",
            f"- patch requested/canonical/applied/noop/dropped={patch['requested_operations']}/{patch['canonical_operations']}/"
            f"{patch['applied_operations']}/{patch['noop_operations']}/{patch['dropped_operations']}；status={patch['status']}。",
            f"- patch row funnel requested/applied/noop/dropped={patch['requested_rows']}/{patch['applied_rows']}/"
            f"{patch['noop_rows']}/{patch['dropped_rows']}。",
            f"- Refiner raw stage marginals={train.get('refiner_funnel', {}).get('raw_marginal_counts')}；"
            f"strict nested XML→grounded diagnosis proxy→patch applied→structural change→score improved="
            f"{train.get('refiner_funnel', {}).get('strict_nested_counts', train.get('refiner_funnel', {}).get('counts'))}；stage retention="
            f"{train.get('refiner_funnel', {}).get('stage_retention')}。",
            f"- structural change={execution['structural_change_rows']}/{train['rows']}；source/updated execution success="
            f"{execution['source_success_rows']}/{execution['updated_success_rows']}；errors={execution['error_rows']}；violations={execution['violation_rows']}。",
            f"- execution success transitions={execution['success_transitions']}。",
            f"- diagnosis target_type={diag['target_type']}；failure_type={diag['failure_type']}；operation={patch['applied_operation_types']}。",
            f"- diagnosis dominance failure={diag['failure_type_dominance']}；target={diag['target_id_dominance']}；"
            f"operation={diag['applied_operation_dominance']}。",
            f"- diagnosis quality proxy={diag.get('quality_proxy')}。该指标只验证 target/location/patch 的结构一致性和非占位文本，不代表因果诊断一定正确。",
            f"- placeholder hits={diag['placeholder_phrase_rows']}。",
            "- 这里的 source/updated 是执行结果对比；当前 canonical rollout **没有**显式 "
            "`selected_variant`/`selected_source` 字段，不能解读为模型 token 级选择了 old/new MAS。",
            "",
            "| step | rows | groups | reward-diff | MAS-diff | diagnosis-diff | improved | damaged | no_gain | patch-applied | structural-change | delta mean |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for step_signal in train.get("per_step_signals", []):
            outcomes = step_signal.get("editor_outcomes") or {}
            lines.append(
                f"| {step_signal.get('optimizer_step')} | {step_signal.get('rows')} | {step_signal.get('groups')} | "
                f"{step_signal.get('reward_different_groups')} ({fmt(step_signal.get('reward_difference_rate'), 4)}) | "
                f"{step_signal.get('mas_different_groups')} ({fmt(step_signal.get('mas_difference_rate'), 4)}) | "
                f"{step_signal.get('diagnosis_different_groups')} ({fmt(step_signal.get('diagnosis_difference_rate'), 4)}) | "
                f"{outcomes.get('improved', 0)} | {outcomes.get('damaged', 0)} | {outcomes.get('no_gain', 0)} | "
                f"{step_signal.get('applied_rows')} | {step_signal.get('structural_change_rows')} | "
                f"{fmt((step_signal.get('score_delta') or {}).get('mean'))} |"
            )
        lines.extend(["", "#### Audit artifacts", ""])
        validations = train.get("artifact_validation") or {}
        for role in ("orchestra", "editor"):
            payload = validations.get(role) or {}
            lines.append(f"- `{role}_artifact_validation`: {audit_validity(payload)}")
        artifacts = train.get("audit_artifacts") or {}
        if artifacts:
            for name, artifact in sorted(artifacts.items()):
                lines.append(
                    f"- `{name}`: {audit_validity((artifact or {}).get('content'))}；"
                    f"path=`{(artifact or {}).get('path')}`"
                )
        else:
            lines.append("- 详细 post-training audit 尚未生成；最终报告生成前必须补齐并重新读取。")
    lines.extend(["", "## 4. Base 与逐 checkpoint same-step 配对评测", ""])
    for evaluation in report["evaluation_runs"]:
        lines.extend([f"### Run {evaluation['run']}", ""])
        lines.append("| cell | status | Orchestra-only | Orchestra+Editor | within-cell Editor Δ | completed |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for cell in evaluation.get("cells", []):
            lines.append(
                f"| {cell.get('label')} | {cell.get('status')} | {fmt(cell.get('orchestra_only_accuracy'))} | "
                f"{fmt(cell.get('orchestra_editor_accuracy'))} | {fmt(cell.get('editor_increment'))} | "
                f"{cell.get('num_fully_completed', 'n/a')}/{cell.get('num_total', 'n/a')} |"
            )
        if evaluation.get("comparisons"):
            lines.extend(["", "| paired comparison | n | W→C | C→W | tie-correct | tie-wrong | Δ | 95% CI | McNemar p |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
            for comp in evaluation["comparisons"]:
                lines.append(
                    f"| {comp['comparison']} | {comp['paired_n']} | {comp['wrong_to_correct']} | {comp['correct_to_wrong']} | "
                    f"{comp['both_correct']} | {comp['both_wrong']} | {fmt(comp['paired_delta_mean'])} | "
                    f"[{fmt(comp['bootstrap_95ci_low'])}, {fmt(comp['bootstrap_95ci_high'])}] | {fmt(comp['mcnemar_exact_p'])} |"
                )
        lines.append("")
    cross = report.get("cross_run") or {}
    lines.extend(["### 双 LR 跨运行同 cell 对比", ""])
    lines.append(
        f"- mean absolute gap: Orchestra-only={fmt(cross.get('mean_orchestra_only_abs_gap'))}；"
        f"Orchestra+Editor={fmt(cross.get('mean_orchestra_editor_abs_gap'))}。"
    )
    lines.extend([
        "| cell | A Orchestra | B Orchestra | abs gap | A pipeline | B pipeline | abs gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for cell in cross.get("cells", []):
        lines.append(
            f"| {cell.get('label')} | {fmt(cell.get('orchestra_only_accuracy_A'))} | "
            f"{fmt(cell.get('orchestra_only_accuracy_B'))} | {fmt(cell.get('orchestra_only_abs_gap'))} | "
            f"{fmt(cell.get('orchestra_editor_accuracy_A'))} | "
            f"{fmt(cell.get('orchestra_editor_accuracy_B'))} | "
            f"{fmt(cell.get('orchestra_editor_abs_gap'))} |"
        )
    lines.extend(["", "## 5. 训练信号是否转化为 held-out 收益", ""])
    lines.append(
        "本节是双学习率判断的核心：训练内 XML、文本或 patch 多样性只能说明探索发生了；"
        "只有同一套 168 个 held-out 样本上的 Base→checkpoint 变化，以及同 cell 的 Orchestra-only→Orchestra+Editor 增量，才能说明泛化收益。"
    )
    lines.extend([
        "",
        "| run | Orchestra LR | Editor LR | train improved | train damaged | train no-gain | zero-var groups | non-placeholder evidence | held-out mean Δ vs Base | checkpoints > Base | significant +/− | Editor mean Δ | Editor significant +/− | best | final |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in (report.get("learning_rate_comparison") or {}).get("runs", []):
        held = row.get("held_out") or {}
        best = held.get("best_checkpoint") or {}
        final = held.get("final_checkpoint") or {}
        lines.append(
            f"| {row.get('run')} | {fmt(row.get('orchestra_lr'), 8)} | {fmt(row.get('editor_lr'), 8)} | "
            f"{fmt(row.get('training_improved_rate'), 4)} | {fmt(row.get('training_damaged_rate'), 4)} | "
            f"{fmt(row.get('training_no_gain_rate'), 4)} | {fmt(row.get('zero_reward_variance_group_rate'), 4)} | "
            f"{fmt(row.get('evidence_nonplaceholder_rate'), 4)} | "
            f"{fmt(held.get('trained_pipeline_mean_delta_vs_base'), 4)} | "
            f"{held.get('checkpoints_pipeline_above_base', 0)}/{held.get('trained_checkpoint_count', 0)} | "
            f"{held.get('pipeline_vs_base_positive_95ci', 0)}/{held.get('pipeline_vs_base_negative_95ci', 0)} | "
            f"{fmt(held.get('editor_increment_mean'), 4)} | "
            f"{held.get('editor_increment_positive_95ci', 0)}/{held.get('editor_increment_negative_95ci', 0)} | "
            f"{best.get('label', 'n/a')} ({fmt(best.get('pipeline_delta_vs_base'), 4)}) | "
            f"{final.get('label', 'n/a')} ({fmt(final.get('pipeline_delta_vs_base'), 4)}) |"
        )
    lines.extend([
        "",
        "- `significant +/−` 使用 paired bootstrap 95% CI；checkpoint-vs-Base 仍含 rollout 随机性，不能仅凭单个最好 checkpoint 宣称 LR 胜出。",
        "- 同 cell Editor 增量共享同一个 Orchestra 输出，是更干净的 Refiner 边际效果；最终 LR 结论优先看其跨 step 一致性，再看 pipeline-vs-Base。",
        "- 如果更高 LR 只降低模板复制或增加诊断文本差异，却没有提高 held-out Editor 增量，则应判定为‘探索增加但 credit assignment 未改善’。",
    ])
    lines.extend(["", "## 6. 根因分析", ""])
    for index, finding in enumerate(root_cause_findings(report), 1):
        lines.append(f"{index}. {finding}")
    lines.extend(["", "## 7. 优先改进方案", ""])
    lines.extend([
        "### P0：让 Refiner 学会‘定位证据’，而不是只学 XML 模板",
        "",
        "- 保留当前三阶段 XML，不增加更多硬筛选阶段；把 stage1 的训练目标变成可核验的 trajectory grounding：必须引用具体 `agent_id + turn/round + output/evidence location + observed failure`。",
        "- 增加轻量、可代码计算的 grounded rewards，并与终局 delta 联合：target 节点存在、evidence location 可解析、诊断与 patch agent 一致、patch 导致真实语义变化、非占位文本、以及 damage penalty。终局 delta 仍是最高权重。",
        "- 针对 source 成功样本允许 `<accept>`，避免强迫模型把所有轨迹都诊断成 `actionable_failure`；accept 正确性由 source score、judge confidence 和轨迹异常共同监督。",
        "- Prompt 中提供失败类型的可操作定义和对照例，但不预填默认 `information_loss`，否则会形成 label prior。",
        "",
        "### P1：提高 group 内有效差异和 credit assignment",
        "",
        "- 组装 Refiner group 时按 target node、failure type、operation family、improved/damaged/no-gain 分层，避免 32 个近同质 `rewrite_role(agent_0)`。保留多操作 patch 能力。",
        "- 对三阶段分别记录辅助分数，但最终仍训练同一个 Refiner response：`R = w_outcome*delta + w_ground*grounding + w_consistency*diagnosis_patch_consistency + w_change*semantic_change - w_damage*damage`。",
        "- 对 no-gain 样本区分：有效诊断但环境未改善、语义 no-op、错误 target、patch 与诊断不一致；不要全部压成同一个 0。",
        "",
        "### P2：训练与监控",
        "",
        "- 先修好 Refiner reward 密度和 group 多样性，再做 Editor 专属 LR/step sweep；建议双 8 卡分别验证 `lr/steps` 与 `grounded reward/group stratification`，而不是同时改很多变量。",
        "- 保存 Refiner behavior token logprobs，恢复 replay probability correlation、KL 和 importance-ratio 监控。",
        "- 每轮固定输出四层漏斗：XML valid → patch translated/applied → structural/semantic change → score improved，并按 failure/action/target 分桶。",
        "- 继续使用 XML-first；XML 已明显解决结构化训练与编译可靠性问题，不建议退回 JSON。",
    ])
    lines.extend(["", "## 8. 审计说明", ""])
    lines.extend([
        "- Orchestra GRPO 响应 token 来自 source `<mas>`；updated MAS 仅用于 adversarial score。",
        "- Refiner replay 响应是完整 `<refiner>` 三阶段 reasoning + patch XML，不是更新后的完整 MAS。",
        "- canonical rollout 当前不含显式 old/new selection 字段；报告只陈述 source/updated outcome comparison，不虚构 selected variant。",
        "- 评测模式固定为 `same-step`：每个 run 只评 Base/Base 与 O1/E1…ON/EN，不做 O×E 两两 Cartesian。",
        f"- experiment manifests={list((report.get('experiment_manifests') or {}).keys())}。",
        f"- 报告 schema: `{SCHEMA}`；完整机器可读信号见 `eReport.json` 与 `signal_summary.json`，样本级配对见 `paired_results.csv`。",
    ])
    return "\n".join(lines) + "\n"


def load_audit_artifacts(run_root: Path, evaluation_root: Path) -> dict[str, Any]:
    """Collect post-training audits without making the report depend on optional files."""
    domain_root = evaluation_root.parent
    candidates = {
        "strict_training_audit": domain_root / "training_audit.json",
        "training_semantics_audit": domain_root / "training_semantics_audit.json",
        "training_rollout_audit": domain_root / "training_rollout_audit.json",
        "run_training_semantics_audit": run_root / "training_semantics_audit.json",
        "run_training_rollout_audit": run_root / "training_rollout_audit.json",
        "evaluation_artifact_validation": evaluation_root / "artifact_validation.json",
    }
    artifacts: dict[str, Any] = {}
    for name, path in candidates.items():
        if path.is_file():
            artifacts[name] = {"path": str(path), "content": load_json(path)}
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-root", required=True)
    parser.add_argument("--run-b-root", required=True)
    parser.add_argument("--eval-a-root", required=True)
    parser.add_argument("--eval-b-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--expected-eval-total", type=int, default=168)
    parser.add_argument("--expected-eval-steps", type=int, default=11)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be >= 100")
    roots = {"A": Path(args.run_a_root).resolve(), "B": Path(args.run_b_root).resolve()}
    eval_roots = {"A": Path(args.eval_a_root).resolve(), "B": Path(args.eval_b_root).resolve()}
    output = Path(args.output_root).resolve()

    training = [summarize_training(run, roots[run]) for run in ("A", "B")]
    for item in training:
        run = item["run"]
        item["audit_artifacts"] = load_audit_artifacts(roots[run], eval_roots[run])
    experiment_manifests = {}
    for name in ("lifecycle_restart_manifest.json", "launch_manifest.json"):
        path = output.parent / name
        if path.is_file():
            experiment_manifests[name] = {"path": str(path), "content": load_json(path)}
    evaluations: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for run in ("A", "B"):
        summary, rows = summarize_evaluation(
            run, eval_roots[run], args.bootstrap_draws, args.allow_incomplete,
            expected_total=args.expected_eval_total, expected_steps=args.expected_eval_steps,
        )
        evaluations.append(summary)
        pairs.extend(rows)
    report = {
        "schema_version": SCHEMA,
        "inputs": {
            "run_a_root": str(roots["A"]), "run_b_root": str(roots["B"]),
            "eval_a_root": str(eval_roots["A"]), "eval_b_root": str(eval_roots["B"]),
            "bootstrap_draws": args.bootstrap_draws,
        },
        "training_runs": training,
        "experiment_manifests": experiment_manifests,
        "evaluation_runs": evaluations,
        "learning_rate_comparison": learning_rate_comparison(training, evaluations),
        "cross_run": cross_run(evaluations[0], evaluations[1]),
        "integrity": {
            "expected_training_rows_per_run": {item["run"]: item.get("expected_rows") for item in training},
            "training_row_counts_valid": all(item.get("row_count_valid") for item in training),
            "expected_eval_cells_total": sum(item.get("num_jobs", 0) for item in evaluations),
            "planned_eval_cells_total": sum(len(item.get("cells", [])) for item in evaluations),
            "successful_eval_cells_total": sum(cell.get("status") == "success" for item in evaluations for cell in item.get("cells", [])),
            "expected_eval_total_per_cell": args.expected_eval_total,
            "expected_same_step_count_per_run": args.expected_eval_steps,
            "all_eval_cells_exact_total": all(
                cell.get("exact_total_valid") is True
                for item in evaluations for cell in item.get("cells", [])
            ),
            "all_evaluations_complete": all(item.get("complete") for item in evaluations),
            "fingerprint_sha256": "",
        },
    }
    fingerprint_payload = canonical({"training": training, "evaluation": evaluations})
    report["integrity"]["fingerprint_sha256"] = hashlib.sha256(fingerprint_payload.encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "eReport.json", report)
    atomic_json(output / "signal_summary.json", {
        "schema_version": SCHEMA,
        "training_runs": training,
        "experiment_manifests": experiment_manifests,
        "evaluation_runs": evaluations,
        "learning_rate_comparison": report["learning_rate_comparison"],
        "cross_run": report["cross_run"],
        "integrity": report["integrity"],
    })
    write_csv(output / "paired_results.csv", pairs)
    atomic_text(output / "eReport.md", markdown(report))
    print(json.dumps({
        "output_root": str(output),
        "training_rows": {item["run"]: item["rows"] for item in training},
        "successful_eval_cells": report["integrity"]["successful_eval_cells_total"],
        "all_evaluations_complete": report["integrity"]["all_evaluations_complete"],
        "fingerprint_sha256": report["integrity"]["fingerprint_sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
