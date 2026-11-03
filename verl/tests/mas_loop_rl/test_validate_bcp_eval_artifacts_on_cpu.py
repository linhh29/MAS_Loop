from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load_validator():
    path = ROOT / "datasets" / "validate_bcp_eval_artifacts.py"
    spec = importlib.util.spec_from_file_location("validate_bcp_eval_artifacts", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def judge(candidate: str) -> dict:
    return {
        "score": 1.0, "model": "judge", "max_tokens": 1024,
        "raw_judgement": '{"correct":"yes"}', "error": "", "parse_error": False,
        "candidate": candidate,
        "prompt": [{"role": "system", "content": "system"}, {"role": "user", "content": "official"}],
        "canonical_browsecomp_plus_prompt": True,
    }


def make_job(out: Path, total: int = 2) -> None:
    args = {
        "orchestra_max_tokens": 4096, "editor_max_tokens": 8192,
        "subagent_max_tokens": 8192, "bcp_judge_max_tokens": 1024,
        "batch_size": 256, "workers": 256, "max_samples": 0,
        "actor_guided_json": False,
        "orchestra_temperature": 0.6, "editor_temperature": 0.7,
        "bcp_judge_temperature": 0.0, "no_resume": True,
    }
    (out / "logs").mkdir(parents=True)
    (out / "run_config.json").write_text(json.dumps({
        "dataset": "bcp", "num_selected_samples": total, "args": args,
        "runtime": {
            "physical_gpus": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
            "tensor_parallel_size": 16, "data_parallel_size": 1, "world_size": 16,
            "vllm_sleep_level": 1, "exclusive_vllm": True,
            "structured_retries": 5,
            "editor_validation_profile": "explicit_accept_or_edits_strict_v1",
            "editor_retry_profile": "xml_refiner_source_aware_dynamic_validation_feedback_v4",
        },
        "retrieval": {
            "profile": "controller", "backend": "bm25", "bm25_backend": "controller_parquet",
            "bm25_k1": 1.2, "bm25_b": 0.75, "bm25_max_doc_chars": 9000,
            "topk": 5, "max_rounds": 5, "evidence_policy": "accumulated",
            "doc_max_tokens": 900, "doc_truncation_unit": "whitespace_words",
            "max_final_snippets": 15, "planner_temperature": 1.0,
            "planner_max_tokens": 256,
            "initial_query_max_tokens": 8192,
            "agent_generation_max_tokens": 8192,
            "initial_query_message_mode": "single_user_no_system",
            "controller_output_constraint": "response_format_json_object",
            "agent_message_mode": "single_user_no_system",
            "execution_prompt_profile": "controller_sub_agent_exact_v1",
            "subagent_temperature": 1.0,
            "disk_query_cache_disabled": True,
        },
        "models": {
            "judge": {"model_path": "/models/Qwen3-30B-A3B"},
            "subagent": {"model_path": "/models/Qwen3-30B-A3B"},
        },
    }), encoding="utf-8")
    stage_summary = {"completed": total, "scored": total, "correct": total, "accuracy": 1.0}
    (out / "summary.json").write_text(json.dumps({
        "num_total": total, "num_records": total, "num_fully_completed": total,
        "num_with_errors": 0, "orchestra_only": stage_summary,
        "orchestra_editor": stage_summary,
    }), encoding="utf-8")
    results = []
    judges = []
    for i in range(total):
        key = f"sample-{i}"
        def stage(candidate: str):
            raw = (
                '<mas topology="dag" aggregation="solver"><agents><agent id="solver" budget="1" '
                'execution_mode="llm_only"><role>Solve and answer.</role><tools /></agent></agents>'
                '<edges /></mas>'
                if candidate == "source" else
                '<refiner><stage1_localization source_status="actionable_failure" target_type="agent" '
                'target_id="solver" failure_type="information_loss"><evidence><item>missing fact</item>'
                '</evidence></stage1_localization><stage2_causal_model><causal_chain><step>fact omitted</step>'
                '</causal_chain><causal_explanation>omission hurts answer</causal_explanation>'
                '<counterfactual>retain fact</counterfactual><downstream_impact>correct answer</downstream_impact>'
                '</stage2_causal_model><stage3_repair_plan confidence="0.8"><repair_goal>retain fact</repair_goal>'
                '<expected_effect>better answer</expected_effect><risk>low</risk></stage3_repair_plan>'
                '<patch><operation op="rewrite_role" agent_id="solver"><role>Use all facts.</role>'
                '</operation></patch></refiner>'
            )
            return {"prompt": [], "raw_output": raw, "trajectory": {}, "prediction": "a", "score": 1,
                    "semantic_judge": judge(candidate)}
        results.append({"sample_key": key, "task": "q", "gold_answer": "a", "source_row": {},
                        "errors": [], "orchestra_only": stage("source"), "orchestra_editor": stage("updated")})
        for candidate in ("source", "updated"):
            standalone = judge(candidate)
            standalone.pop("prompt")
            standalone.pop("canonical_browsecomp_plus_prompt")
            judges.append({"pair_id": key, "candidate": candidate, "judge_prompt": "official",
                           "response": "a", "correct_answer": "a", "judge": standalone})
    write_jsonl(out / "results.jsonl", results)
    write_jsonl(out / "judge_results" / "judge.jsonl", judges)
    write_jsonl(
        out / "runtime_profile_shards" / "profiles.jsonl",
        [
            {
                "task_sha256": f"task-{i}", "phase": "updated",
                "retrieval_sec": 0.0, "bm25_sec": 0.0,
                "generation_sec": 0.0, "total_sec": 0.0,
            }
            for i in range(total)
        ],
    )
    for relative in load_validator().REQUIRED_LOGS:
        path = out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("log\n", encoding="utf-8")


def test_validate_complete_matrix(tmp_path: Path) -> None:
    validator = load_validator()
    out = tmp_path / "job"
    make_job(out)
    plan = {
        "physical_gpus": validator.GPU_SET, "pairing_mode": "cartesian", "repeat_count": 1,
        "num_jobs": 1,
        "checkpoint_counts": {"bcp": {"orchestra": 0, "editor": 0}},
        "jobs": [{"dataset": "bcp", "kind": "baseline", "job_key": "baseline", "output_dir": str(out)}],
    }
    (tmp_path / "matrix_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    result = validator.validate_matrix(tmp_path, expected_total=2)
    assert result["valid"] is True
    assert result["jobs"][0]["expected_judge_candidates"] == 4
    assert (tmp_path / "evaluation_audit.json").is_file()
    assert (tmp_path / "evaluation_artifact_audit.json").is_file()


def test_validate_matrix_enforces_explicit_checkpoint_and_job_gates(tmp_path: Path) -> None:
    validator = load_validator()
    out = tmp_path / "job"
    make_job(out)
    plan = {
        "physical_gpus": validator.GPU_SET,
        "pairing_mode": "same-step",
        "repeat_count": 1,
        "num_jobs": 1,
        "checkpoint_counts": {"bcp": {"orchestra": 0, "editor": 0}},
        "jobs": [
            {
                "dataset": "bcp",
                "kind": "baseline",
                "job_key": "baseline",
                "output_dir": str(out),
            }
        ],
    }
    (tmp_path / "matrix_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(RuntimeError, match="checkpoint count mismatch"):
        validator.validate_matrix(
            tmp_path,
            expected_total=2,
            expected_checkpoints_per_role=11,
            expected_jobs=12,
        )


def test_validate_matrix_enforces_explicit_job_count_gate(tmp_path: Path) -> None:
    validator = load_validator()
    out = tmp_path / "job"
    make_job(out)
    plan = {
        "physical_gpus": validator.GPU_SET,
        "pairing_mode": "same-step",
        "repeat_count": 1,
        "num_jobs": 1,
        "checkpoint_counts": {"bcp": {"orchestra": 0, "editor": 0}},
        "jobs": [
            {
                "dataset": "bcp",
                "kind": "baseline",
                "job_key": "baseline",
                "output_dir": str(out),
            }
        ],
    }
    (tmp_path / "matrix_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(RuntimeError, match="matrix job count mismatch"):
        validator.validate_matrix(tmp_path, expected_total=2, expected_jobs=12)


def test_validate_job_reports_dynamic_eight_gpu_runtime_contract(tmp_path: Path) -> None:
    validator = load_validator()
    out = tmp_path / "job"
    make_job(out)
    physical_gpus = "8,9,10,11,12,13,14,15"
    config = json.loads((out / "run_config.json").read_text())
    config["runtime"]["physical_gpus"] = physical_gpus
    config["runtime"]["tensor_parallel_size"] = 8
    config["runtime"]["world_size"] = 8
    (out / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    result = validator.validate_job(out, expected_total=2, physical_gpus=physical_gpus)

    assert result["runtime_contract"]["physical_gpus"] == physical_gpus
    assert result["runtime_contract"]["tensor_parallel_size"] == 8
    assert result["runtime_contract"]["world_size"] == 8


def test_validator_fails_when_standalone_judge_evidence_is_missing(tmp_path: Path) -> None:
    validator = load_validator()
    out = tmp_path / "job"
    make_job(out)
    judge_file = out / "judge_results" / "judge.jsonl"
    judge_file.write_text("\n".join(judge_file.read_text().splitlines()[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lack standalone Judge evidence"):
        validator.validate_job(out, expected_total=2)


def test_validator_fails_when_output_token_contract_changes(tmp_path: Path) -> None:
    validator = load_validator()
    out = tmp_path / "job"
    make_job(out)
    config = json.loads((out / "run_config.json").read_text())
    config["args"]["orchestra_max_tokens"] = 1024
    (out / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="token contract mismatch"):
        validator.validate_job(out, expected_total=2)


def test_structured_raw_output_accepts_refiner_xml(tmp_path: Path) -> None:
    validator = load_validator()
    block = {
        "raw_output": (
            '<refiner><stage1_localization source_status="actionable_failure" target_type="agent" '
            'target_id="solver" failure_type="inaccurate_answer"><evidence><item>wrong result</item></evidence>'
            '</stage1_localization><stage2_causal_model><causal_chain><step>bad reasoning</step></causal_chain>'
            '<causal_explanation>bad reasoning</causal_explanation><counterfactual>verify</counterfactual>'
            '<downstream_impact>correct result</downstream_impact></stage2_causal_model>'
            '<stage3_repair_plan confidence="0.8"><repair_goal>verify</repair_goal>'
            '<expected_effect>correct answer</expected_effect><risk>low</risk></stage3_repair_plan>'
            '<patch><operation op="rewrite_role" agent_id="solver"><role>Verify result.</role>'
            '</operation></patch></refiner>'
        )
    }
    validator._validate_structured_raw_output(
        block,
        output_dir=tmp_path,
        sample_key="sample",
        stage="orchestra_editor",
    )


def test_structured_raw_output_rejects_text_without_mas_xml(tmp_path: Path) -> None:
    validator = load_validator()
    with pytest.raises(RuntimeError, match="no recoverable <mas> XML"):
        validator._validate_structured_raw_output(
            {"raw_output": "not structured output"},
            output_dir=tmp_path,
            sample_key="sample",
            stage="orchestra_only",
        )


@pytest.mark.parametrize("field", ["orchestra_temperature", "editor_temperature"])
def test_validator_fails_when_sampling_temperature_changes(tmp_path: Path, field: str) -> None:
    validator = load_validator()
    out = tmp_path / "job"
    make_job(out)
    config = json.loads((out / "run_config.json").read_text())
    config["args"][field] = 0.5
    (out / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match=rf"argument contract mismatch {field}"):
        validator.validate_job(out, expected_total=2)


@pytest.mark.parametrize("field", ["seed", "random_seed", "sampling_seed"])
def test_validator_fails_when_explicit_sampling_seed_is_present(tmp_path: Path, field: str) -> None:
    validator = load_validator()
    out = tmp_path / "job"
    make_job(out)
    config = json.loads((out / "run_config.json").read_text())
    config["args"][field] = 1234
    (out / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not set an explicit sampling seed"):
        validator.validate_job(out, expected_total=2)


def test_validator_fails_when_retrieval_contract_changes(tmp_path: Path) -> None:
    validator = load_validator()
    out = tmp_path / "job"
    make_job(out)
    config = json.loads((out / "run_config.json").read_text())
    config["retrieval"]["doc_max_tokens"] = 512
    (out / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="retrieval contract mismatch doc_max_tokens"):
        validator.validate_job(out, expected_total=2)


def test_validate_same_step_matrix(tmp_path: Path) -> None:
    validator = load_validator()
    baseline = tmp_path / "baseline"
    step1 = tmp_path / "step1"
    step2 = tmp_path / "step2"
    for out in (baseline, step1, step2):
        make_job(out)
    jobs = [
        {"dataset": "bcp", "kind": "baseline", "job_key": "baseline", "output_dir": str(baseline)},
        {
            "dataset": "bcp", "kind": "same_step", "job_key": "step1", "output_dir": str(step1),
            "orchestra_round": 1, "orchestra_step": 1, "editor_round": 1, "editor_step": 1,
        },
        {
            "dataset": "bcp", "kind": "same_step", "job_key": "step2", "output_dir": str(step2),
            "orchestra_round": 1, "orchestra_step": 2, "editor_round": 1, "editor_step": 2,
        },
    ]
    plan = {
        "physical_gpus": validator.GPU_SET, "pairing_mode": "same-step", "repeat_count": 1,
        "num_jobs": len(jobs),
        "checkpoint_counts": {"bcp": {"orchestra": 2, "editor": 2}},
        "jobs": jobs,
    }
    (tmp_path / "matrix_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    result = validator.validate_matrix(tmp_path, expected_total=2)
    assert result["valid"] is True
    assert result["pairing_mode"] == "same-step"
    assert result["num_jobs"] == 3


def test_validate_same_step_matrix_rejects_mismatched_pair(tmp_path: Path) -> None:
    validator = load_validator()
    baseline = tmp_path / "baseline"
    step = tmp_path / "step"
    for out in (baseline, step):
        make_job(out)
    jobs = [
        {"dataset": "bcp", "kind": "baseline", "job_key": "baseline", "output_dir": str(baseline)},
        {
            "dataset": "bcp", "kind": "same_step", "job_key": "bad", "output_dir": str(step),
            "orchestra_round": 1, "orchestra_step": 1, "editor_round": 1, "editor_step": 2,
        },
    ]
    plan = {
        "physical_gpus": validator.GPU_SET, "pairing_mode": "same-step", "repeat_count": 1,
        "num_jobs": len(jobs),
        "checkpoint_counts": {"bcp": {"orchestra": 1, "editor": 1}},
        "jobs": jobs,
    }
    (tmp_path / "matrix_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(RuntimeError, match="mismatched Orchestra/Editor pair"):
        validator.validate_matrix(tmp_path, expected_total=2)
