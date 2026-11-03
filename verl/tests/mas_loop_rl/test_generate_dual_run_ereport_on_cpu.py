from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from mas_loop_rl.generate_dual_run_ereport import (
    cross_run,
    eval_label,
    evaluation_learning_summary,
    expected_orchestra_rows,
    refiner_funnel_summary,
    summarize_evaluation,
    training_metric,
)


def test_refiner_funnel_separates_raw_marginals_from_strict_nested_path() -> None:
    result = refiner_funnel_summary(
        Counter({
            "total_rows": 100,
            "xml_valid": 100,
            "diagnosis_grounded_proxy": 0,
            "patch_applied": 0,
        }),
        raw_marginal_counts={
            "total_rows": 100,
            "xml_valid": 100,
            "target_resolved_proxy": 100,
            "evidence_nonplaceholder_proxy": 0,
            "patch_applied": 97,
            "structural_change": 95,
            "score_improved": 4,
            "score_damaged": 5,
        },
    )

    assert result["raw_marginal_counts"]["patch_applied"] == 97
    assert result["strict_nested_counts"]["patch_applied"] == 0
    assert result["counts"] == result["strict_nested_counts"]
    assert result["stage_retention"]["diagnosis_given_xml"] == 0.0


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _evaluation_cell(root: Path, label: str, offset: int) -> Path:
    output = root / "bcp" / label.replace("/", "_")
    results = [
        {
            "sample_key": "sample-0",
            "sample_id": "0",
            "task": "q0",
            "gold_answer": "a0",
            "orchestra_only": {"score": offset % 2, "prediction": "p0"},
            "orchestra_editor": {"score": (offset + 1) % 2, "prediction": "e0"},
        },
        {
            "sample_key": "sample-1",
            "sample_id": "1",
            "task": "q1",
            "gold_answer": "a1",
            "orchestra_only": {"score": 1, "prediction": "p1"},
            "orchestra_editor": {"score": 1, "prediction": "e1"},
        },
    ]
    _write_jsonl(output / "results.jsonl", results)
    orch_correct = sum(row["orchestra_only"]["score"] for row in results)
    editor_correct = sum(row["orchestra_editor"]["score"] for row in results)
    _write_json(output / "summary.json", {
        "num_total": 2,
        "num_fully_completed": 2,
        "num_with_errors": 0,
        "orchestra_only": {"correct": orch_correct, "accuracy": orch_correct / 2},
        "orchestra_editor": {"correct": editor_correct, "accuracy": editor_correct / 2},
    })
    return output


def test_eval_label_supports_dynamic_same_step() -> None:
    assert eval_label({"kind": "baseline"}) == "Base/Base"
    assert eval_label({"kind": "same_step", "orchestra_step": 11, "editor_step": 11}) == "O11/E11"


def test_summarize_evaluation_accepts_base_plus_eleven_same_step_cells(tmp_path: Path) -> None:
    jobs = []
    labels = ["Base/Base"] + [f"O{step}/E{step}" for step in range(1, 12)]
    for index, label in enumerate(labels):
        output = _evaluation_cell(tmp_path, label, index)
        if index == 0:
            job = {
                "dataset": "bcp", "kind": "baseline", "job_key": "base",
                "orchestra_step": None, "editor_step": None,
                "orchestra_round": None, "editor_round": None,
                "orchestra_source": "base", "editor_source": "base",
                "output_dir": str(output),
            }
        else:
            job = {
                "dataset": "bcp", "kind": "same_step", "job_key": f"step-{index}",
                "orchestra_step": index, "editor_step": index,
                "orchestra_round": 1, "editor_round": 1,
                "orchestra_source": f"orchestra-{index}", "editor_source": f"editor-{index}",
                "output_dir": str(output),
            }
        jobs.append(job)
    _write_json(tmp_path / "matrix_plan.json", {
        "pairing_mode": "same-step", "jobs": jobs,
    })
    _write_jsonl(tmp_path / "matrix_results.jsonl", [
        {"job_key": job["job_key"], "status": "success"} for job in jobs
    ])

    summary, pairs = summarize_evaluation("A", tmp_path, draws=100, allow_incomplete=False)

    assert summary["complete"] is True
    assert summary["pairing_mode"] == "same-step"
    assert summary["num_jobs"] == 12
    assert summary["expected_steps"] == list(range(1, 12))
    assert [cell["label"] for cell in summary["cells"]] == labels
    assert len(summary["comparisons"]) == 45
    assert len(pairs) == 90


def test_summarize_evaluation_rejects_trajectory_plan(tmp_path: Path) -> None:
    _write_json(tmp_path / "matrix_plan.json", {"pairing_mode": "training-trajectory", "jobs": []})
    with pytest.raises(RuntimeError, match="pairing_mode=same-step"):
        summarize_evaluation("A", tmp_path, draws=100, allow_incomplete=False)


def test_training_rows_and_metric_series_are_dynamic(tmp_path: Path) -> None:
    (tmp_path / "launch_config.env").write_text(
        "BCP_TRAIN_BATCH_SIZE=64\nBCP_ROLLOUT_N=32\n", encoding="utf-8"
    )
    metric_path = tmp_path / "round_1" / "orchestra" / "metrics" / "training_metrics.jsonl"
    _write_jsonl(metric_path, [
        {"global_step": 1, "metrics": {"actor/lr": 5e-7}},
        {"global_step": 2, "metrics": {"actor/lr": 5e-7}},
    ])

    metrics = training_metric(tmp_path, "orchestra")

    assert metrics["num_steps"] == 2
    assert [row["global_step"] for row in metrics["series"]] == [1, 2]
    assert expected_orchestra_rows(tmp_path, metrics["num_steps"]) == 4096


def test_summarize_evaluation_rejects_wrong_exact_cell_size(tmp_path: Path) -> None:
    jobs = []
    for index, label in enumerate(["Base/Base"] + [f"O{step}/E{step}" for step in range(1, 12)]):
        output = _evaluation_cell(tmp_path, label, index)
        jobs.append({
            "dataset": "bcp",
            "kind": "baseline" if index == 0 else "same_step",
            "job_key": "base" if index == 0 else f"step-{index}",
            "orchestra_step": None if index == 0 else index,
            "editor_step": None if index == 0 else index,
            "orchestra_round": None if index == 0 else 1,
            "editor_round": None if index == 0 else 1,
            "orchestra_source": "base" if index == 0 else f"orchestra-{index}",
            "editor_source": "base" if index == 0 else f"editor-{index}",
            "output_dir": str(output),
        })
    _write_json(tmp_path / "matrix_plan.json", {"pairing_mode": "same-step", "jobs": jobs})
    _write_jsonl(tmp_path / "matrix_results.jsonl", [
        {"job_key": job["job_key"], "status": "success"} for job in jobs
    ])

    with pytest.raises(RuntimeError, match="evaluation incomplete"):
        summarize_evaluation(
            "A", tmp_path, draws=100, allow_incomplete=False,
            expected_total=168, expected_steps=11,
        )


def test_diagnosis_structural_quality_proxies() -> None:
    from mas_loop_rl.generate_dual_run_ereport import (
        diagnosis_patch_aligned,
        diagnosis_target_resolves,
        evidence_location_resolves,
    )

    mas = {
        "agents": [{"id": "a"}, {"id": "b"}],
        "edges": [{"from": "a", "to": "b"}],
        "aggregation": "b",
    }
    agent_diag = {"target_type": "agent", "target_id": "a"}
    edge_diag = {"target_type": "edge", "target_id": "a->b"}
    assert diagnosis_target_resolves(agent_diag, mas)
    assert diagnosis_target_resolves(edge_diag, mas)
    assert not diagnosis_target_resolves({"target_type": "agent", "target_id": "missing"}, mas)
    assert evidence_location_resolves("agent:a.output", mas)
    assert evidence_location_resolves("edge:a->b", mas)
    assert not evidence_location_resolves("agent:missing.output", mas)
    assert diagnosis_patch_aligned(agent_diag, [{"op": "rewrite_role", "agent_id": "a"}])
    assert diagnosis_patch_aligned(edge_diag, [{"op": "add_edge", "from": "a", "to": "b"}])
    assert not diagnosis_patch_aligned(agent_diag, [{"op": "rewrite_role", "agent_id": "b"}])


def test_evaluation_learning_summary_separates_training_cells_from_base() -> None:
    evaluation = {
        "run": "A",
        "complete": True,
        "cells": [
            {"label": "Base/Base", "status": "success", "orchestra_only_accuracy": 0.50, "orchestra_editor_accuracy": 0.50, "editor_increment": 0.0},
            {"label": "O1/E1", "status": "success", "orchestra_only_accuracy": 0.50, "orchestra_editor_accuracy": 0.60, "editor_increment": 0.10},
            {"label": "O2/E2", "status": "success", "orchestra_only_accuracy": 0.45, "orchestra_editor_accuracy": 0.40, "editor_increment": -0.05},
        ],
        "comparisons": [
            {"comparison": "O1/E1 pipeline vs Base", "bootstrap_95ci_low": 0.01, "bootstrap_95ci_high": 0.19},
            {"comparison": "O2/E2 pipeline vs Base", "bootstrap_95ci_low": -0.20, "bootstrap_95ci_high": -0.01},
            {"comparison": "O1/E1 within-cell Editor increment", "bootstrap_95ci_low": 0.01, "bootstrap_95ci_high": 0.19},
            {"comparison": "O2/E2 within-cell Editor increment", "bootstrap_95ci_low": -0.11, "bootstrap_95ci_high": 0.01},
        ],
    }

    result = evaluation_learning_summary(evaluation)

    assert result["trained_checkpoint_count"] == 2
    assert result["trained_pipeline_mean_delta_vs_base"] == pytest.approx(0.0)
    assert result["checkpoints_pipeline_above_base"] == 1
    assert result["checkpoints_pipeline_below_base"] == 1
    assert result["pipeline_vs_base_positive_95ci"] == 1
    assert result["pipeline_vs_base_negative_95ci"] == 1
    assert result["editor_increment_positive_95ci"] == 1
    assert result["best_checkpoint"]["label"] == "O1/E1"
    assert result["final_checkpoint"]["label"] == "O2/E2"


def test_cross_run_uses_numeric_same_step_order_and_signed_lr_gap() -> None:
    def evaluation(run: str, values: dict[str, float]) -> dict:
        return {
            "run": run,
            "cells": [
                {
                    "label": label, "status": "success",
                    "orchestra_only_accuracy": value,
                    "orchestra_editor_accuracy": value,
                }
                for label, value in values.items()
            ],
        }

    result = cross_run(
        evaluation("A", {"O10/E10": 0.4, "Base/Base": 0.5, "O2/E2": 0.6}),
        evaluation("B", {"O10/E10": 0.5, "Base/Base": 0.5, "O2/E2": 0.55}),
    )

    assert [row["label"] for row in result["cells"]] == ["Base/Base", "O2/E2", "O10/E10"]
    assert result["cells"][1]["orchestra_editor_signed_B_minus_A"] == pytest.approx(-0.05)
    assert result["cells"][2]["orchestra_editor_signed_B_minus_A"] == pytest.approx(0.10)
