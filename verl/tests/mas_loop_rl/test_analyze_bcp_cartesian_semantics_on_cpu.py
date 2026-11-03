import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[3] / "datasets" / "analyze_bcp_cartesian_mas_eval.py"
SPEC = importlib.util.spec_from_file_location("analyze_bcp_cartesian_mas_eval", SCRIPT)
assert SPEC and SPEC.loader
ANALYZE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ANALYZE
SPEC.loader.exec_module(ANALYZE)


def _write_audit(evaluation_root: Path, value: object) -> Path:
    evaluation_root.mkdir(parents=True)
    path = evaluation_root.parent / "training_semantics_audit.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_load_training_semantics_audit_returns_valid_payload(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "run" / "evaluation"
    audit = {
        "schema": "mas_loop_bcp_training_semantics_audit_v1",
        "valid": True,
        "orchestra": {"groups": 768},
    }
    _write_audit(evaluation_root, audit)

    assert ANALYZE.load_training_semantics_audit(evaluation_root) == audit


def test_load_training_semantics_audit_accepts_current_v3_payload(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "run" / "evaluation"
    audit = {
        "schema": "mas_loop_bcp_training_semantics_audit_v3",
        "valid": True,
        "orchestra": {"groups": 768},
    }
    _write_audit(evaluation_root, audit)

    assert ANALYZE.load_training_semantics_audit(evaluation_root) == audit


def test_load_training_semantics_audit_allows_missing_legacy_artifact(tmp_path: Path) -> None:
    evaluation_root = tmp_path / "run" / "evaluation"
    evaluation_root.mkdir(parents=True)

    assert ANALYZE.load_training_semantics_audit(evaluation_root) is None


@pytest.mark.parametrize(
    "audit",
    [
        [],
        {"schema": "wrong", "valid": True},
        {"schema": "mas_loop_bcp_training_semantics_audit_v1", "valid": False},
    ],
)
def test_load_training_semantics_audit_rejects_invalid_artifact(tmp_path: Path, audit: object) -> None:
    evaluation_root = tmp_path / "run" / "evaluation"
    _write_audit(evaluation_root, audit)

    with pytest.raises(RuntimeError, match="training semantics audit"):
        ANALYZE.load_training_semantics_audit(evaluation_root)


def _task_case(
    *,
    cell: str,
    source_score: int,
    edited_score: int,
    accept: bool,
    source_mas: dict,
    final_mas: dict,
) -> dict:
    return {
        "cell": cell,
        "sample_key": "task-1",
        "sample_id": "id-1",
        "question": "Which structure works?",
        "gold_answer": "answer",
        "source_score": source_score,
        "edited_score": edited_score,
        "transition": ANALYZE.transition(source_score, edited_score),
        "accept": accept,
        "operations": ["accept"] if accept else ["add_verifier"],
        "source_mas_features": ANALYZE.mas_features(source_mas),
        "final_mas_features": ANALYZE.mas_features(final_mas),
        "source_mas": source_mas,
        "final_mas": final_mas,
    }


def test_mas_features_exposes_depth_width_and_role_distribution() -> None:
    mas = {
        "agents": [
            {"id": "planner", "role": "Decompose and plan the investigation."},
            {"id": "web_a", "role": "Research primary sources."},
            {"id": "web_b", "role": "Browse for independent evidence."},
            {"id": "final", "role": "Synthesize the final answer."},
        ],
        "edges": [
            {"from": "planner", "to": "web_a"},
            {"from": "planner", "to": "web_b"},
            {"from": "web_a", "to": "final"},
            {"from": "web_b", "to": "final"},
        ],
        "aggregation": "final",
    }

    features = ANALYZE.mas_features(mas)

    assert features["roots"] == 1
    assert features["leaves"] == 1
    assert features["graph_depth"] == 3
    assert features["graph_width"] == 2
    assert features["has_cycle"] == 0
    assert features["invalid_edges"] == 0
    assert features["role_planner"] == 1
    assert features["role_researcher"] == 2
    assert features["role_synthesizer"] == 1
    assert features["role_signature"] == "planner:1,researcher:2,synthesizer:1"


def test_mas_features_marks_cycles_invalid_and_duplicate_edges_explicitly() -> None:
    mas = {
        "agents": [
            {"id": "a", "role": "solve"},
            {"id": "b", "role": "verify"},
        ],
        "edges": [
            {"from": "a", "to": "b"},
            {"from": "a", "to": "b"},
            {"from": "b", "to": "a"},
            {"from": "b", "to": "missing"},
        ],
        "aggregation": "b",
    }

    features = ANALYZE.mas_features(mas)

    assert features["has_cycle"] == 1
    assert features["graph_depth"] == -1
    assert features["graph_width"] == -1
    assert features["roots"] == 0
    assert features["leaves"] == 0
    assert features["invalid_edges"] == 1
    assert features["duplicate_edges"] == 1
    assert features["role_solver"] == 1
    assert features["role_verifier"] == 1


def test_build_task_structure_rows_preserves_per_task_mas_variants() -> None:
    source = {
        "agents": [{"id": "researcher", "role": "research", "budget": 1}],
        "edges": [],
        "aggregation": "researcher",
        "topology": "single",
        "max_iterations": 1,
    }
    verified = {
        "agents": [
            {"id": "researcher", "role": "research", "budget": 1},
            {"id": "verifier", "role": "verify", "budget": 1},
        ],
        "edges": [{"from": "researcher", "to": "verifier"}],
        "aggregation": "verifier",
        "topology": "chain",
        "max_iterations": 1,
    }
    rows = ANALYZE.build_task_structure_rows(
        [
            _task_case(
                cell="O1xE1",
                source_score=0,
                edited_score=1,
                accept=False,
                source_mas=source,
                final_mas=verified,
            ),
            _task_case(
                cell="O1xE2",
                source_score=1,
                edited_score=0,
                accept=True,
                source_mas=source,
                final_mas=source,
            ),
        ],
        expected_cells=2,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["cells"] == 2
    assert row["help"] == 1
    assert row["harm"] == 1
    assert row["accept_rate"] == 0.5
    assert row["source_mas_unique"] == 1
    assert row["final_mas_unique"] == 2
    assert row["operation_counts"] == {"accept": 1, "add_verifier": 1}
    assert row["delta_feature_directions"]["agents"] == {"increase": 1, "same": 1}
    assert sum(item["count"] for item in row["final_mas_variants"]) == 2
    assert any(item["mas"] == verified for item in row["final_mas_variants"])


def test_build_task_structure_rows_requires_every_cartesian_cell() -> None:
    mas = {"agents": [], "edges": []}
    case = _task_case(
        cell="O1xE1",
        source_score=0,
        edited_score=0,
        accept=True,
        source_mas=mas,
        final_mas=mas,
    )

    with pytest.raises(RuntimeError, match="appears in 1 cells, expected 2"):
        ANALYZE.build_task_structure_rows([case], expected_cells=2)


def test_validated_patch_errors_treats_model_error_as_data() -> None:
    errors = ["early_stop_rule:early_stop_rule requires max_iterations in [1, 8]"]

    assert ANALYZE.validated_patch_errors({"patch_errors": errors}, context="fixture") == errors
    assert ANALYZE.patch_error_type(errors[0]) == "early_stop_rule"


@pytest.mark.parametrize("value", ["not-a-list", [1], [""]])
def test_validated_patch_errors_rejects_malformed_artifacts(value: object) -> None:
    with pytest.raises(RuntimeError, match="malformed patch_errors"):
        ANALYZE.validated_patch_errors({"patch_errors": value}, context="fixture")


def test_paired_accuracies_respects_before_and_after_semantics() -> None:
    stats = ANALYZE.outcome_stats({"help": 3, "harm": 2, "both_correct": 1, "both_wrong": 4})

    source_accuracy, edited_accuracy = ANALYZE.paired_accuracies(stats)

    assert source_accuracy == pytest.approx(0.3)
    assert edited_accuracy == pytest.approx(0.4)
    assert edited_accuracy - source_accuracy == pytest.approx((3 - 2) / 10)


def test_paired_accuracies_rejects_empty_stats() -> None:
    with pytest.raises(RuntimeError, match="positive n"):
        ANALYZE.paired_accuracies({"n": 0})


def test_main_writes_per_task_mas_artifacts_and_correct_paired_accuracy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "evaluation"
    baseline_dir = root / "baseline"
    cell_dir = root / "o1_e1"
    baseline_dir.mkdir(parents=True)
    cell_dir.mkdir(parents=True)
    source_mas = {
        "agents": [{"id": "a", "role": "research", "budget": 1}],
        "edges": [],
        "aggregation": "a",
        "topology": "single",
        "max_iterations": 1,
    }
    final_mas = {
        "agents": [
            {"id": "a", "role": "research", "budget": 1},
            {"id": "v", "role": "verifier", "budget": 1},
        ],
        "edges": [{"from": "a", "to": "v"}],
        "aggregation": "v",
        "topology": "chain",
        "max_iterations": 1,
    }

    def stage(score: int, mas: dict, *, updated_mas: dict | None = None, operation: str | None = None) -> dict:
        value = {
            "score": score,
            "semantic_judge": {"score": score, "reasoning": "fixture evidence"},
            "prediction": "fixture",
            "mas": mas,
        }
        if updated_mas is not None:
            value["updated_mas"] = updated_mas
        if operation is not None:
            value["patch"] = {"operations": [{"op": operation}]}
        return value

    baseline_row = {
        "sample_key": "task-1",
        "sample_id": "id-1",
        "task": "question | pipe",
        "gold_answer": "answer",
        "orchestra_only": stage(0, source_mas),
        "orchestra_editor": stage(0, source_mas, updated_mas=source_mas, operation="accept"),
    }
    cell_row = {
        **{key: baseline_row[key] for key in ("sample_key", "sample_id", "task", "gold_answer")},
        "orchestra_only": stage(0, source_mas),
        "orchestra_editor": {
            **stage(1, source_mas, updated_mas=final_mas, operation="add_verifier"),
            "patch_errors": ["early_stop_rule:invalid max_iterations"],
        },
    }
    for output_dir, row in ((baseline_dir, baseline_row), (cell_dir, cell_row)):
        (output_dir / "summary.json").write_text(
            json.dumps({"num_total": 1, "num_fully_completed": 1, "num_with_errors": 0}),
            encoding="utf-8",
        )
        (output_dir / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (root / "matrix_plan.json").write_text(
        json.dumps(
            {
                "pairing_mode": "cartesian",
                "repeat_count": 1,
                "num_jobs": 2,
                "checkpoint_counts": {"bcp": {"orchestra": 1, "editor": 1}},
                "jobs": [
                    {
                        "dataset": "bcp",
                        "kind": "baseline",
                        "job_key": "base",
                        "output_dir": str(baseline_dir),
                    },
                    {
                        "dataset": "bcp",
                        "kind": "cartesian",
                        "job_key": "o1e1",
                        "output_dir": str(cell_dir),
                        "orchestra_step": 1,
                        "editor_step": 1,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--evaluation-root",
            str(root),
            "--expected-total",
            "1",
            "--expected-orchestra",
            "1",
            "--expected-editor",
            "1",
        ],
    )

    assert ANALYZE.main() == 0

    comparison = root / "comparison"
    analysis = json.loads((comparison / "cartesian_analysis.json").read_text(encoding="utf-8"))
    task_row = json.loads((comparison / "task_mas_structure_summary.jsonl").read_text(encoding="utf-8"))
    case_row = json.loads((comparison / "case_transitions.jsonl").read_text(encoding="utf-8"))
    assert analysis["best_cell"]["source_accuracy"] == 0.0
    assert analysis["best_cell"]["edited_accuracy"] == 1.0
    assert analysis["best_cell"]["editor_delta"] == 1.0
    assert analysis["accept"]["selectivity_gap_correct_minus_wrong"] is None
    assert analysis["audit"]["model_patch_error_rows"] == 1
    assert analysis["audit"]["infrastructure_errors"] == 0
    assert analysis["patch_errors"]["rows_total"] == 1
    assert analysis["patch_errors"]["types"][0]["patch_error_type"] == "early_stop_rule"
    assert analysis["task_structure_count"] == 1
    assert task_row["help"] == 1
    assert task_row["source_mas_unique"] == 1
    assert task_row["final_mas_unique"] == 1
    assert case_row["source_mas"]["topology"] == "single"
    assert case_row["final_mas"]["topology"] == "chain"
    assert case_row["source_mas_features"]["graph_depth"] == 1
    assert case_row["final_mas_features"]["graph_depth"] == 2
    assert case_row["final_mas_features"]["role_signature"] == "researcher:1,verifier:1"
    assert case_row["patch_error_types"] == ["early_stop_rule"]
    assert (comparison / "patch_error_metrics.csv").is_file()
    assert r"question \| pipe" in (comparison / "task_mas_structure_report.md").read_text(encoding="utf-8")
    assert "source depth/width" in (comparison / "task_mas_structure_report.md").read_text(encoding="utf-8")
    assert "## 逐 task MAS 结构" in (comparison / "cartesian_report.md").read_text(encoding="utf-8")
