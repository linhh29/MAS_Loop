from __future__ import annotations

import errno
import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_script(name: str):
    path = ROOT / "datasets" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _CapturingActorClient:
    def __init__(self, parsed):
        self.parsed = parsed
        self.calls = []

    def chat_validated(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return "{}", self.parsed, {"attempts": 1}


@pytest.mark.parametrize("enabled", [False, True])
def test_eval_actor_guided_json_is_opt_in(enabled: bool):
    evaluator = load_script("mas_loop_two_stage_eval.py")
    args = argparse.Namespace(
        dataset="bcp",
        max_agents=5,
        orchestra_temperature=0.2,
        orchestra_max_tokens=8192,
        editor_temperature=0.2,
        editor_max_tokens=8192,
        actor_guided_json=enabled,
    )
    row = {"task": "question", "gold_answer": "answer", "index": 0, "sample_id": "0"}

    valid_mas = {
        "agents": [{"id": "solver", "role": "Solve the task."}],
        "edges": [],
        "aggregation": "solver",
    }
    orchestra = _CapturingActorClient(valid_mas)
    evaluator._generate_orchestra(orchestra, row, args)
    expected_orchestra = evaluator.MAS_GUIDED_JSON_SCHEMA if enabled else None
    assert orchestra.calls[0][1]["guided_json"] == expected_orchestra
    assert orchestra.calls[0][1]["retry_instruction"] == evaluator.ORCHESTRA_RETRY_INSTRUCTION
    assert "XML <mas>" in evaluator.ORCHESTRA_RETRY_INSTRUCTION
    assert "not JSON" in evaluator.ORCHESTRA_RETRY_INSTRUCTION

    editor = _CapturingActorClient({"operations": []})
    evaluator._generate_editor(
        editor,
        row,
        {"mas": valid_mas},
        {"trajectory": {}},
        args,
    )
    expected_editor = evaluator.PATCH_GUIDED_JSON_SCHEMA if enabled else None
    assert editor.calls[0][1]["guided_json"] == expected_editor
    editor_validator = editor.calls[0][1]["validator"]
    compiled = editor_validator(
        '<refiner><stage1_localization source_status="actionable_failure" target_type="agent" '
        'target_id="solver" failure_type="missing_execution_trace"><evidence>'
        '<item agent_id="solver" execution_index="0" field="execution_status" '
        'location="node_cards[0].execution_status">The solver execution status is missing.</item></evidence>'
        '</stage1_localization><stage2_causal_model><causal_chain><step>loss</step></causal_chain>'
        '<causal_explanation>loss</causal_explanation><counterfactual>retain</counterfactual>'
        '<downstream_impact>answer</downstream_impact></stage2_causal_model>'
        '<stage3_repair_plan confidence="0.8"><repair_goal>retain evidence</repair_goal>'
        '<expected_effect>better answer</expected_effect><risk>low</risk></stage3_repair_plan>'
        '<patch><operation op="rewrite_role" agent_id="solver"><role>Use all evidence.</role>'
        '</operation></patch></refiner>'
    )
    assert compiled["operations"][0]["agent_id"] == "solver"
    retry_instruction = editor.calls[0][1]["retry_instruction"]
    assert "omit accept completely" in retry_instruction
    assert "Never mix accept" in retry_instruction
    assert "Allowed operation op attributes are exactly" in retry_instruction
    assert "<refiner>" in retry_instruction
    assert "Return no JSON" in retry_instruction
    assert "Do not use change_role; use rewrite_role" in retry_instruction
    assert "change_status is not a valid operation" in retry_instruction
    assert "validation error appended below" in retry_instruction
    for operation in evaluator.VALID_PATCH_OPS:
        assert operation in retry_instruction
    assert evaluator.EDITOR_RETRY_PROFILE == "xml_refiner_source_aware_dynamic_validation_feedback_v4"


def test_bcp_orchestra_prompt_uses_neutral_mixed_routing_graph():
    from mas_loop_rl.prompts import orchestra_prompt

    prompt = orchestra_prompt("Find the answer.", {"evaluation_dataset": "bcp", "max_agents": 8})
    system = prompt[0]["content"]
    text = prompt[-1]["content"]
    assert "XML <mas>" in system
    assert "Use 2-8 agents" not in system
    assert "Use 3-6 agents" in text
    assert "3-6 clear self-contained sub-tasks" in text
    assert "task-specific role" in text
    assert "Generic or reusable placeholder roles are invalid" in text
    assert "specialized role and instructions" not in text
    assert 'tools=["bm25"]' in text
    assert 'tools=[]' in text
    assert "final synthesizer/verifier role and required final-answer format" not in text
    assert "NEW facts/documents" in text
    assert "aggregation agent should normally be llm_only" in text


def test_bcp_eval_accepts_native_mas_xml_before_overlimit_repair():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    raw = (
        '<mas topology="dag" aggregation="agent_2" max_cost="3" max_iterations="3">'
        '<agents><agent id="agent_0" budget="1" execution_mode="multi_turn_search">'
        '<role>Find evidence</role><tools><tool>bm25</tool></tools></agent>'
        '<agent id="agent_1" budget="1" execution_mode="llm_only"><role>Reconcile</role><tools />'
        '</agent><agent id="agent_2" budget="1" execution_mode="llm_only">'
        '<role>Return answer</role><tools /></agent></agents><edges>'
        '<edge from="agent_0" to="agent_1" type="data_flow" />'
        '<edge from="agent_1" to="agent_2" type="data_flow" /></edges></mas>'
    )
    mas = evaluator._normalize_eval_mas(raw, "bcp")
    assert [agent["id"] for agent in mas["agents"]] == ["agent_0", "agent_1", "agent_2"]
    assert mas["aggregation"] == "agent_2"
    assert mas["agents"][0]["execution_mode"] == "multi_turn_search"


def test_bcp_overlimit_mas_preserves_declared_llm_only_aggregation():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    agents = [
        {
            "id": f"search_{i}",
            "role": f"Search clue {i}",
            "tools": ["bm25"],
            "budget": 4,
            "execution_mode": "multi_turn_search",
        }
        for i in range(9)
    ]
    agents.append({
        "id": "final",
        "role": "Return one concise answer.",
        "tools": ["bm25"],
        "budget": 1,
        "execution_mode": "llm_only",
    })
    mas = evaluator._normalize_eval_mas({
        "agents": agents,
        "edges": [{"from": f"search_{i}", "to": f"search_{i+1}", "type": "data_flow"} for i in range(9)],
        "aggregation": "final",
        "topology": "dag",
        "max_cost": 10,
        "max_iterations": 3,
    }, "bcp")

    assert len(mas["agents"]) == 8
    assert mas["aggregation"] == "final"
    final = next(agent for agent in mas["agents"] if agent["id"] == "final")
    assert final["execution_mode"] == "llm_only"
    assert final["tools"] == []
    retained = {agent["id"] for agent in mas["agents"] if agent["id"] != "final"}
    assert retained
    assert all(any(edge["from"] == agent_id and edge["to"] == "final" for edge in mas["edges"])
               for agent_id in retained)


def test_bcp_overlimit_mas_synthesizes_final_when_all_authored_agents_search():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    agents = [
        {
            "id": f"search_{i}", "role": f"Search clue {i}", "tools": ["bm25"],
            "budget": 4, "execution_mode": "multi_turn_search",
        }
        for i in range(10)
    ]
    mas = evaluator._normalize_eval_mas({
        "agents": agents, "edges": [], "aggregation": "search_9",
        "topology": "sequential", "max_cost": 10, "max_iterations": 3,
    }, "bcp")

    assert len(mas["agents"]) == 8
    assert mas["aggregation"].startswith("final_synthesizer")
    assert next(agent for agent in mas["agents"] if agent["id"] == mas["aggregation"])["execution_mode"] == "llm_only"
    assert any(agent["execution_mode"] == "multi_turn_search" for agent in mas["agents"])



def test_bcp_all_llm_only_graph_repairs_retrieval_role():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    mas = evaluator._normalize_eval_mas({
        "agents": [
            {"id": "reasoner", "role": "Reconcile constraints", "tools": ["bm25"], "execution_mode": "llm_only"},
            {"id": "retriever", "role": "Search the corpus for evidence", "tools": [], "execution_mode": "llm_only"},
            {"id": "final", "role": "Return the answer", "tools": ["bm25"], "execution_mode": "llm_only"},
        ],
        "edges": [{"from": "retriever", "to": "final", "type": "data_flow"}],
        "aggregation": "final",
    }, "bcp")

    by_id = {agent["id"]: agent for agent in mas["agents"]}
    assert by_id["retriever"]["execution_mode"] == "multi_turn_search"
    assert by_id["retriever"]["tools"] == ["bm25"]
    assert by_id["reasoner"]["execution_mode"] == "llm_only"
    assert by_id["reasoner"]["tools"] == []
    assert by_id["final"]["execution_mode"] == "llm_only"
    assert by_id["final"]["tools"] == []


def test_bcp_search_aggregation_is_repaired_to_mixed_routing():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    mas = evaluator._normalize_eval_mas({
        "agents": [
            {"id": "worker", "role": "Analyze the task", "tools": [], "execution_mode": "llm_only"},
            {"id": "final", "role": "Search and answer", "tools": ["bm25"], "execution_mode": "multi_turn_search"},
        ],
        "edges": [{"from": "worker", "to": "final", "type": "data_flow"}],
        "aggregation": "final",
    }, "bcp")

    by_id = {agent["id"]: agent for agent in mas["agents"]}
    assert by_id["worker"]["execution_mode"] == "multi_turn_search"
    assert by_id["worker"]["tools"] == ["bm25"]
    assert by_id["final"]["execution_mode"] == "llm_only"
    assert by_id["final"]["tools"] == []


def test_bcp_valid_mixed_routing_preserves_graph_semantics():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    authored_edges = [
        {"from": "search", "to": "reason", "type": "data_flow"},
        {"from": "reason", "to": "final", "type": "summarize"},
    ]
    mas = evaluator._normalize_eval_mas({
        "agents": [
            {"id": "search", "role": "Find evidence", "tools": ["bm25"], "execution_mode": "multi_turn_search"},
            {"id": "reason", "role": "Reconcile evidence", "tools": [], "execution_mode": "llm_only"},
            {"id": "final", "role": "Return the answer", "tools": [], "execution_mode": "llm_only"},
        ],
        "edges": authored_edges,
        "aggregation": "final",
        "topology": "dag",
    }, "bcp")

    assert mas["edges"] == authored_edges
    assert [agent["execution_mode"] for agent in mas["agents"]] == [
        "multi_turn_search", "llm_only", "llm_only",
    ]


def test_bcp_one_agent_graph_gets_bounded_final_synthesizer():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    mas = evaluator._normalize_eval_mas({
        "agents": [{"id": "solo", "role": "Solve", "tools": [], "execution_mode": "llm_only"}],
        "edges": [],
        "aggregation": "solo",
    }, "bcp")

    assert len(mas["agents"]) == 2
    by_id = {agent["id"]: agent for agent in mas["agents"]}
    assert by_id["solo"]["execution_mode"] == "multi_turn_search"
    assert by_id["solo"]["tools"] == ["bm25"]
    assert mas["aggregation"].startswith("final_synthesizer")
    assert by_id[mas["aggregation"]]["execution_mode"] == "llm_only"
    assert any(edge["from"] == "solo" and edge["to"] == mas["aggregation"] for edge in mas["edges"])


def test_non_reasoning_non_bcp_normalization_does_not_apply_routing_repair():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    mas = evaluator._normalize_eval_mas({
        "agents": [{"id": "solver", "role": "Solve", "tools": ["custom"], "execution_mode": "llm_only"}],
        "edges": [],
        "aggregation": "solver",
    }, "alfworld")

    assert len(mas["agents"]) == 1
    assert mas["agents"][0]["execution_mode"] == "llm_only"
    assert mas["agents"][0]["tools"] == ["custom"]

def test_eval_launcher_handles_disabled_actor_guided_json_with_nounset():
    launcher = (ROOT / "datasets" / "run_mas_loop_two_stage_eval.sh").read_text(encoding="utf-8")
    guarded_expansion = '${actor_guided_args[@]+"${actor_guided_args[@]}"}'
    assert launcher.count(guarded_expansion) == 2

    completed = subprocess.run(
        [
            "bash",
            "-uc",
            'actor_guided_args=(); args=(before ${actor_guided_args[@]+"${actor_guided_args[@]}"} after); '
            '[[ "${#args[@]}" == 2 && "${args[0]}" == before && "${args[1]}" == after ]]',
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_eval_launchers_reject_destructive_level_two_sleep():
    common = (ROOT / "datasets" / "run_mas_loop_two_stage_eval.sh").read_text(encoding="utf-8")
    bcp = (ROOT / "datasets" / "run_bcp_mas_loop_eval.sh").read_text(encoding="utf-8")
    aime = (ROOT / "datasets" / "run_deepscaler_aime_mas_loop_eval.sh").read_text(encoding="utf-8")
    evaluator = (ROOT / "datasets" / "mas_loop_two_stage_eval.py").read_text(encoding="utf-8")

    assert "${BCP_EVAL_VLLM_SLEEP_LEVEL:-1}" in bcp
    assert "${AIME_EVAL_VLLM_SLEEP_LEVEL:-1}" in aime
    assert "${MAS_LOOP_VLLM_SLEEP_LEVEL:-1}" in common
    assert '[[ "${MAS_LOOP_VLLM_SLEEP_LEVEL}" != "1" ]]' in common
    assert "level 2 discards model weights" in common
    reset_pos = common.index('reset_prefix_cache" >/dev/null')
    sleep_pos = common.index('/sleep?level=${MAS_LOOP_VLLM_SLEEP_LEVEL}')
    assert reset_pos < sleep_pos
    assert 'os.getenv("MAS_LOOP_VLLM_SLEEP_LEVEL", "1")' in evaluator


def test_training_trajectory_covers_each_saved_role_step_once():
    matrix = load_script("run_all_mas_loop_checkpoint_matrix_eval.py")
    baseline = "/models/base"
    checkpoints = [
        matrix.Checkpoint("bcp", "orchestra", 1, 1, "/o1"),
        matrix.Checkpoint("bcp", "orchestra", 1, 2, "/o2"),
        matrix.Checkpoint("bcp", "editor", 1, 1, "/e1"),
        matrix.Checkpoint("bcp", "editor", 1, 2, "/e2"),
    ]
    jobs = matrix.build_jobs("bcp", baseline, checkpoints, "training-trajectory")
    assert [job.kind for job in jobs] == ["baseline", "orchestra_step", "orchestra_step", "editor_step", "editor_step"]
    assert [job.orchestra_step for job in jobs[1:3]] == [1, 2]
    assert all(job.editor_source == baseline for job in jobs[1:3])
    assert [job.editor_step for job in jobs[3:]] == [1, 2]
    assert all(job.orchestra_step == 2 for job in jobs[3:])


def test_checkpoint_discovery_ignores_recovery_quarantine(tmp_path: Path):
    matrix = load_script("run_all_mas_loop_checkpoint_matrix_eval.py")
    run_root = tmp_path / "run"

    def write_checkpoint(path: Path) -> None:
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")

    write_checkpoint(run_root / "round_1" / "orchestra" / "global_step_6" / "actor" / "huggingface")
    write_checkpoint(run_root / "round_1" / "editor" / "global_step_6" / "actor" / "huggingface")
    write_checkpoint(
        run_root
        / "recovery_quarantine"
        / "incomplete_joint_step_6"
        / "orchestra"
        / "global_step_6"
        / "actor"
        / "huggingface"
    )

    checkpoints = matrix.load_checkpoints([run_root], "bcp")
    assert [(item.role, item.train_round, item.global_step) for item in checkpoints] == [
        ("editor", 1, 6),
        ("orchestra", 1, 6),
    ]


def test_bcp_matrix_pins_independent_eight_gpu_domain(tmp_path: Path):
    matrix = load_script("run_all_mas_loop_checkpoint_matrix_eval.py")
    job = matrix.EvalJob(
        dataset="bcp",
        kind="baseline",
        orchestra_source_type="local",
        orchestra_source="/models/base",
        orchestra_round=None,
        orchestra_step=None,
        editor_source_type="local",
        editor_source="/models/base",
        editor_round=None,
        editor_step=None,
    )
    gpu_set = "8,9,10,11,12,13,14,15"
    _, env = matrix.env_for_job(job, tmp_path, "/models/subagent", gpu_set, 8810)
    assert env["CUDA_VISIBLE_DEVICES"] == gpu_set
    assert env["BCP_EVAL_ALL_GPUS"] == gpu_set
    assert env["BCP_EVAL_N_GPUS"] == "8"
    assert env["BCP_EVAL_TENSOR_PARALLEL_SIZE"] == "8"
    assert env["BCP_EVAL_DATA_PARALLEL_SIZE"] == "1"
    assert env["BCP_EVAL_ORCHESTRA_PORT"] == "8810"
    assert env["BCP_EVAL_EDITOR_PORT"] == "8811"
    assert env["BCP_EVAL_SUBAGENT_PORT"] == "8812"
    assert env["BCP_EVAL_JUDGE_PORT"] == "8813"


def test_bcp_matrix_pins_tp16_and_all_subagent_concurrency_layers(tmp_path: Path):
    matrix = load_script("run_all_mas_loop_checkpoint_matrix_eval.py")
    job = matrix.EvalJob(
        dataset="bcp",
        kind="baseline",
        orchestra_source_type="local",
        orchestra_source="/models/base",
        orchestra_round=None,
        orchestra_step=None,
        editor_source_type="local",
        editor_source="/models/base",
        editor_round=None,
        editor_step=None,
    )
    _, env = matrix.env_for_job(job, tmp_path, "/models/subagent")

    gpu_set = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    assert env["CUDA_VISIBLE_DEVICES"] == gpu_set
    assert env["BCP_EVAL_ALL_GPUS"] == gpu_set
    assert env["BCP_EVAL_N_GPUS"] == "16"
    assert env["BCP_EVAL_TENSOR_PARALLEL_SIZE"] == "16"
    assert env["BCP_EVAL_DATA_PARALLEL_SIZE"] == "1"
    assert env["MAS_LOOP_STRUCTURED_OUTPUT_PROTOCOL"] == "xml"
    assert env["MAS_LOOP_ACTOR_GUIDED_JSON"] == "0"
    assert env["MAS_LOOP_EDITOR_GUIDED_JSON"] == "0"
    assert env["MAS_LOOP_JUDGE_GUIDED_JSON"] == "0"
    for name in (
        "BCP_EVAL_BATCH_SIZE",
        "BCP_EVAL_WORKERS",
        "BCP_EVAL_MAX_CONCURRENCY",
        "BCP_EVAL_DISPATCH_WORKERS_PER_PROCESS",
        "BCP_EVAL_SUBAGENT_MAX_ACTIVE",
        "BCP_EVAL_SUBAGENT_MAX_NUM_SEQS",
    ):
        assert env[name] == ("256" if name == "BCP_EVAL_BATCH_SIZE" else "128")

    expected_gate_exact = {
        "BCP_EVAL_BM25_BACKEND": "controller_parquet",
        "BCP_EVAL_BM25_K1": "1.2",
        "BCP_EVAL_BM25_B": "0.75",
        "BCP_EVAL_BM25_MAX_DOC_CHARS": "9000",
        "BCP_EVAL_RETRIEVAL_PROFILE": "controller",
        "BCP_EVAL_SEARCH_ROUNDS": "5",
        "BCP_EVAL_BM25_TOPK": "5",
        "BCP_EVAL_DOC_MAX_TOKENS": "900",
        "BCP_EVAL_MAX_FINAL_SNIPPETS": "15",
        "BCP_EVAL_SEARCH_PLANNER_MAX_TOKENS": "256",
        "BCP_EVAL_INITIAL_QUERY_MAX_TOKENS": "8192",
        "BCP_EVAL_ORCHESTRA_MAX_TOKENS": "4096",
        "BCP_EVAL_EDITOR_MAX_TOKENS": "8192",
        "BCP_EVAL_SUBAGENT_MAX_TOKENS": "8192",
        "BCP_EVAL_BCP_JUDGE_MAX_TOKENS": "1024",
        "BCP_EVAL_ORCHESTRA_TEMPERATURE": "0.6",
        "BCP_EVAL_EDITOR_TEMPERATURE": "0.7",
        "BCP_EVAL_SUBAGENT_TEMPERATURE": "1.0",
        "BCP_EVAL_JUDGE_TEMPERATURE": "0.0",
        "BCP_EVAL_JUDGE_GUIDED_JSON": "0",
        "BCP_EVAL_SCORE_MODE": "judge",
        "BCP_EVAL_NO_RESUME": "1",
        "MAS_LOOP_STRUCTURED_RETRIES": "5",
    }
    assert {name: env[name] for name in expected_gate_exact} == expected_gate_exact


def make_phase(root: Path, role: str, steps: int = 2) -> Path:
    phase = root / "round_1" / role
    metrics = phase / "metrics" / "training_metrics.jsonl"
    metrics.parent.mkdir(parents=True)
    with metrics.open("w", encoding="utf-8") as handle:
        for step in range(1, steps + 1):
            handle.write(json.dumps({"global_step": step, "metrics": {"training/global_step": step}}) + "\n")
    canonical = phase / "rollouts" / "canonical_rollouts"
    canonical.mkdir(parents=True)
    (canonical / f"{role}.jsonl").write_text(
        "".join(json.dumps({"pair_id": f"{role}-{i}"}) + "\n" for i in range(steps)),
        encoding="utf-8",
    )
    for step in range(1, steps + 1):
        folder = phase / f"global_step_{step}"
        hf = folder / "actor" / "huggingface"
        hf.mkdir(parents=True)
        (hf / "config.json").write_text("{}\n", encoding="utf-8")
        (hf / "model.safetensors").write_bytes(b"model")
        (folder / ".mas_loop_checkpoint_complete.json").write_text(
            json.dumps({"global_step": step, "protocol": "mas-loop-checkpoint-complete-v1"}) + "\n",
            encoding="utf-8",
        )
    return phase


def test_every_optimizer_step_checkpoint_audit(tmp_path: Path):
    supervisor = load_script("run_bcp_posttrain_eval_supervisor.py")
    make_phase(tmp_path, "orchestra")
    make_phase(tmp_path, "editor")
    audit = supervisor.verify_every_step_saved(tmp_path)
    assert audit["valid"] is True
    assert audit["roles"]["orchestra"][0]["checkpoint_steps"] == [1, 2]
    assert audit["roles"]["editor"][0]["metric_steps"] == [1, 2]


def test_every_optimizer_step_checkpoint_audit_fails_closed(tmp_path: Path):
    supervisor = load_script("run_bcp_posttrain_eval_supervisor.py")
    make_phase(tmp_path, "orchestra")
    editor = make_phase(tmp_path, "editor")
    marker = editor / "global_step_2" / ".mas_loop_checkpoint_complete.json"
    marker.unlink()
    with pytest.raises(RuntimeError, match="completion marker"):
        supervisor.verify_every_step_saved(tmp_path)


def test_analysis_writes_paired_step_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    analysis = load_script("analyze_bcp_checkpoint_eval.py")
    base_dir = tmp_path / "bcp" / "base"
    step_dir = tmp_path / "bcp" / "step"
    base_dir.mkdir(parents=True)
    step_dir.mkdir(parents=True)
    plan = {
        "jobs": [
            {"dataset": "bcp", "kind": "baseline", "job_key": "base", "output_dir": str(base_dir),
             "orchestra_round": None, "orchestra_step": None, "editor_round": None, "editor_step": None},
            {"dataset": "bcp", "kind": "orchestra_step", "job_key": "step", "output_dir": str(step_dir),
             "orchestra_round": 1, "orchestra_step": 1, "editor_round": None, "editor_step": None},
        ]
    }
    (tmp_path / "matrix_plan.json").write_text(json.dumps(plan), encoding="utf-8")

    def write_eval(path: Path, scores: list[int]):
        rows = []
        for i, score in enumerate(scores):
            stage = {
                "score": score, "exact_match_score": score, "prediction": f"p{i}",
                "semantic_judge": {"score": score, "error": "", "parse_error": ""},
            }
            rows.append({"sample_key": f"s{i}", "sample_id": str(i), "task": f"q{i}", "gold_answer": f"a{i}",
                         "orchestra_only": stage, "orchestra_editor": dict(stage)})
        (path / "results.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
        correct = sum(scores)
        summary = {"num_total": len(scores), "num_fully_completed": len(scores), "num_with_errors": 0,
                   "orchestra_only": {"accuracy": correct / len(scores), "correct": correct, "scored": len(scores)},
                   "orchestra_editor": {"accuracy": correct / len(scores), "correct": correct, "scored": len(scores)}}
        (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    write_eval(base_dir, [0, 1])
    write_eval(step_dir, [1, 1])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            "--evaluation-root",
            str(tmp_path),
            "--bootstrap-draws",
            "100",
            "--expected-total",
            "2",
        ],
    )
    assert analysis.main() == 0
    comparison = tmp_path / "comparison"
    assert (comparison / "analysis.md").is_file()
    assert (comparison / "per_step_metrics.jsonl").is_file()
    significance = json.loads((comparison / "significance.json").read_text(encoding="utf-8"))
    assert significance["comparisons"][-1]["wins"] == 1
    assert significance["comparisons"][-1]["losses"] == 0



def test_analysis_reports_editor_upstream_trajectory_variability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    analysis = load_script("analyze_bcp_checkpoint_eval.py")
    job_specs = [
        ("base", "baseline", None, None),
        ("orchestra-2", "orchestra_step", 2, None),
        ("editor-1", "editor_step", 2, 1),
        ("editor-2", "editor_step", 2, 2),
    ]
    jobs = []
    paths = {}
    for key, kind, orchestra_step, editor_step in job_specs:
        output_dir = tmp_path / "bcp" / key
        output_dir.mkdir(parents=True)
        paths[key] = output_dir
        jobs.append(
            {
                "dataset": "bcp",
                "kind": kind,
                "job_key": key,
                "output_dir": str(output_dir),
                "orchestra_round": 1 if orchestra_step is not None else None,
                "orchestra_step": orchestra_step,
                "editor_round": 1 if editor_step is not None else None,
                "editor_step": editor_step,
            }
        )
    (tmp_path / "matrix_plan.json").write_text(json.dumps({"jobs": jobs}), encoding="utf-8")

    def write_eval(path: Path, orchestra_scores: list[int], pipeline_scores: list[int]):
        rows = []
        for index, (orchestra_score, pipeline_score) in enumerate(zip(orchestra_scores, pipeline_scores)):
            def stage(score: int, suffix: str):
                return {
                    "score": score,
                    "exact_match_score": score,
                    "prediction": f"p{index}-{suffix}",
                    "semantic_judge": {"score": score, "error": "", "parse_error": ""},
                }

            rows.append(
                {
                    "sample_key": f"s{index}",
                    "sample_id": str(index),
                    "task": f"q{index}",
                    "gold_answer": f"a{index}",
                    "orchestra_only": stage(orchestra_score, "o"),
                    "orchestra_editor": stage(pipeline_score, "e"),
                }
            )
        (path / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        total = len(rows)
        (path / "summary.json").write_text(
            json.dumps(
                {
                    "num_total": total,
                    "num_fully_completed": total,
                    "num_with_errors": 0,
                    "orchestra_only": {
                        "accuracy": sum(orchestra_scores) / total,
                        "correct": sum(orchestra_scores),
                        "scored": total,
                    },
                    "orchestra_editor": {
                        "accuracy": sum(pipeline_scores) / total,
                        "correct": sum(pipeline_scores),
                        "scored": total,
                    },
                }
            ),
            encoding="utf-8",
        )

    write_eval(paths["base"], [0, 0], [0, 0])
    write_eval(paths["orchestra-2"], [1, 0], [1, 0])
    write_eval(paths["editor-1"], [1, 0], [1, 1])
    write_eval(paths["editor-2"], [1, 1], [1, 0])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze",
            "--evaluation-root",
            str(tmp_path),
            "--bootstrap-draws",
            "100",
            "--expected-total",
            "2",
        ],
    )

    assert analysis.main() == 0
    comparison = tmp_path / "comparison"
    variability = json.loads(
        (comparison / "editor_orchestra_reference_variability.json").read_text(encoding="utf-8")
    )
    assert variability["job_count"] == 2
    assert variability["accuracy_min"] == 0.5
    assert variability["accuracy_max"] == 1.0
    assert variability["accuracy_range"] == 0.5
    assert variability["accuracy_mean"] == 0.75
    assert variability["accuracy_population_stddev"] == 0.25
    assert variability["all_orchestra_only_accuracies_equal"] is False
    assert len(variability["pairwise_comparisons"]) == 1
    assert variability["pairwise_comparisons"][0]["paired_disagreements"] == 1

    role_rows = [
        json.loads(line)
        for line in (comparison / "role_reference_comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    editor_rows = [
        row for row in role_rows if row["comparison"] == "trained_editor_vs_base_editor_on_final_orchestra"
    ]
    assert editor_rows
    assert all(row["interpretation"] == "end_to_end_pipeline_comparison" for row in editor_rows)
    assert all(row["causal_editor_effect"] is False for row in editor_rows)
    assert all(row["confounded_by_independently_regenerated_upstream_trajectory"] for row in editor_rows)

    contribution_rows = [
        json.loads(line)
        for line in (comparison / "editor_contribution.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["upstream_trajectory_held_fixed"] for row in contribution_rows)
    report = (comparison / "analysis.md").read_text(encoding="utf-8")
    assert "跨 job 比较的随机轨迹混杂" in report
    assert "不能解释为纯 Editor 因果效应" in report

def test_rollout_pair_count_and_storage_contract(tmp_path: Path):
    supervisor = load_script("run_bcp_posttrain_eval_supervisor.py")
    (tmp_path / "launch_config.env").write_text(
        "BCP_TRAIN_BATCH_SIZE=1\nBCP_ROLLOUT_N=1\nBCP_EDITOR_TRAIN_BATCH_SIZE=1\n",
        encoding="utf-8",
    )
    make_phase(tmp_path, "orchestra", steps=2)
    make_phase(tmp_path, "editor", steps=2)
    audit = supervisor.verify_every_step_saved(tmp_path)
    assert audit["roles"]["orchestra"][0]["expected_canonical_pairs"] == 2
    assert audit["roles"]["orchestra"][0]["all_rollout_pairs_saved"] is True
    (tmp_path / "work").mkdir()
    with pytest.raises(RuntimeError, match="forbidden standalone work"):
        supervisor.verify_every_step_saved(tmp_path)


def test_rollout_pair_count_fails_if_one_pair_is_missing(tmp_path: Path):
    supervisor = load_script("run_bcp_posttrain_eval_supervisor.py")
    (tmp_path / "launch_config.env").write_text(
        "BCP_TRAIN_BATCH_SIZE=1\nBCP_ROLLOUT_N=2\nBCP_EDITOR_TRAIN_BATCH_SIZE=1\n",
        encoding="utf-8",
    )
    make_phase(tmp_path, "orchestra", steps=2)
    make_phase(tmp_path, "editor", steps=2)
    with pytest.raises(RuntimeError, match="not every orchestra rollout Pair"):
        supervisor.verify_every_step_saved(tmp_path)


def replace_with_editor_replay(phase: Path, rows: list[dict]) -> Path:
    canonical = phase / "rollouts" / "canonical_rollouts"
    for shard in canonical.glob("*.jsonl"):
        shard.unlink()
    canonical.rmdir()
    replay = phase / "rollouts" / "editor_replay_consumed"
    replay.mkdir(parents=True)
    (replay / "editor-replay.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return replay


def test_editor_replay_consumed_accepts_complete_padding_group(tmp_path: Path):
    supervisor = load_script("run_bcp_posttrain_eval_supervisor.py")
    (tmp_path / "launch_config.env").write_text(
        "BCP_TRAIN_BATCH_SIZE=1\nBCP_ROLLOUT_N=8\n"
        "BCP_EDITOR_TRAIN_BATCH_SIZE=16\nBCP_EDITOR_GROUP_SIZE=8\n",
        encoding="utf-8",
    )
    orchestra = make_phase(tmp_path, "orchestra", steps=1)
    canonical = orchestra / "rollouts" / "canonical_rollouts" / "orchestra.jsonl"
    canonical.write_text(
        "".join(json.dumps({"pair_id": f"o-{i}"}) + "\n" for i in range(8)),
        encoding="utf-8",
    )
    editor = make_phase(tmp_path, "editor", steps=1)
    base_rows = [
        {"pair_id": f"e-{i}", "editor_group_uid": "editor-group-0", "rollout_sample_index": i}
        for i in range(8)
    ]
    padding_rows = [
        {"pair_id": f"e-{i}", "editor_group_uid": "editor-group-0:padding-0", "rollout_sample_index": i}
        for i in range(8)
    ]
    replace_with_editor_replay(editor, base_rows + padding_rows)

    audit = supervisor.verify_every_step_saved(tmp_path)
    editor_audit = audit["roles"]["editor"][0]["rollout_audit"]
    assert editor_audit["source_kind"] == "editor_replay_consumed"
    assert editor_audit["rows"] == 16
    assert editor_audit["unique_pair_ids"] == 8
    assert editor_audit["padding_rows"] == 8
    assert editor_audit["padding_groups"] == 1


def test_editor_replay_consumed_rejects_non_padding_duplicate(tmp_path: Path):
    supervisor = load_script("run_bcp_posttrain_eval_supervisor.py")
    make_phase(tmp_path, "orchestra", steps=1)
    editor = make_phase(tmp_path, "editor", steps=1)
    replace_with_editor_replay(
        editor,
        [
            {"pair_id": "duplicate", "editor_group_uid": "group-a", "rollout_sample_index": 0},
            {"pair_id": "duplicate", "editor_group_uid": "group-b", "rollout_sample_index": 1},
        ],
    )
    with pytest.raises(RuntimeError, match="duplicate non-padding editor pair_id"):
        supervisor.verify_every_step_saved(tmp_path)


def test_editor_replay_consumed_rejects_incomplete_padding_group(tmp_path: Path):
    supervisor = load_script("run_bcp_posttrain_eval_supervisor.py")
    make_phase(tmp_path, "orchestra", steps=1)
    editor = make_phase(tmp_path, "editor", steps=1)
    replace_with_editor_replay(
        editor,
        [
            {"pair_id": "e-0", "editor_group_uid": "group-a", "rollout_sample_index": 0},
            {"pair_id": "e-0", "editor_group_uid": "group-a:padding-0", "rollout_sample_index": 0},
        ],
    )
    with pytest.raises(RuntimeError, match="incomplete Editor padding group"):
        supervisor.verify_every_step_saved(tmp_path)


def test_transient_log_open_retries_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    supervisor = load_script("run_bcp_posttrain_eval_supervisor.py")
    log = tmp_path / "supervisor.log"
    original_open = Path.open
    calls = 0

    def flaky_open(path: Path, *args, **kwargs):
        nonlocal calls
        if path == log and calls < 2:
            calls += 1
            raise OSError(errno.EIO, "transient test EIO", str(path))
        calls += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", flaky_open)
    with supervisor._open_append_with_retry(log, attempts=3, initial_delay_sec=0) as handle:
        handle.write("recovered\n")
    assert calls == 3
    assert log.read_text(encoding="utf-8") == "recovered\n"


def test_bcp_launcher_uses_balanced_subagent_capacity_and_agent_limit():
    launcher = (ROOT / "datasets" / "run_bcp_mas_loop_eval.sh").read_text(encoding="utf-8")
    assert "BCP_EVAL_SUBAGENT_MAX_ACTIVE:-128" in launcher
    assert "BCP_EVAL_SUBAGENT_MAX_NUM_SEQS:-128" in launcher
    assert "BCP_EVAL_MAX_CONCURRENCY:-128" in launcher
    assert "BCP_EVAL_AGENT_LOOP_WORKER_PROCESSES:-1" in launcher
    assert "BCP_EVAL_DISPATCH_WORKERS_PER_PROCESS:-128" in launcher
    assert "BCP_EVAL_SUBAGENT_MAX_BATCHED_TOKENS:-65536" in launcher
    assert "BCP_EVAL_ORCHESTRA_GPU_MEMORY_UTILIZATION:-0.82" in launcher
    assert "BCP_EVAL_ORCHESTRA_MAX_BATCHED_TOKENS:-16384" in launcher
    assert "BCP_EVAL_EDITOR_GPU_MEMORY_UTILIZATION:-0.82" in launcher
    assert "BCP_EVAL_EDITOR_MAX_BATCHED_TOKENS:-16384" in launcher
    assert "BCP_EVAL_MAX_AGENTS:-8" in launcher


def test_two_stage_launcher_defaults_to_schema_agent_limit():
    launcher = (ROOT / "datasets" / "run_mas_loop_two_stage_eval.sh").read_text(encoding="utf-8")
    assert launcher.count('--max-agents "${EVAL_MAX_AGENTS:-8}"') == 2


def test_two_stage_direct_entry_defaults_to_balanced_sample_concurrency():
    launcher = (ROOT / "datasets" / "run_mas_loop_two_stage_eval.sh").read_text(encoding="utf-8")
    evaluator = (ROOT / "datasets" / "mas_loop_two_stage_eval.py").read_text(encoding="utf-8")
    assert launcher.count('--batch-size "${EVAL_BATCH_SIZE:-256}" --workers "${EVAL_WORKERS:-128}"') == 2
    assert 'p.add_argument("--batch-size", type=int, default=256)' in evaluator
    assert 'p.add_argument("--workers", type=int, default=256)' in evaluator


def test_editor_action_reports_unsupported_operation_name_for_retry_feedback():
    from mas_loop_rl.schema import normalize_editor_action

    with pytest.raises(ValueError, match=r"unsupported_operation:change_status") as exc_info:
        normalize_editor_action({"patch": [{"op": "change_status", "status": "success"}]})
    assert "not translatable" in str(exc_info.value)


def test_domain_same_step_supervisor_exposes_repo_packages_to_parent_process(monkeypatch):
    supervisor = load_script("run_bcp_domain_same_step_posttrain.py")
    expected = [str(ROOT / "verl" / ".mas_loop_deps"), str(ROOT / "verl"), str(ROOT)]
    stripped = [entry for entry in sys.path if entry not in expected]
    monkeypatch.setattr(sys, "path", stripped)

    roots = supervisor.ensure_repo_pythonpath(ROOT)

    assert roots == expected
    assert sys.path[:3] == expected
    import mas_loop_rl.backfill_bcp_judges as backfill  # noqa: F401


@pytest.mark.parametrize("physical_gpus", ["0,1,2,3,4,5,6,7", "8,9,10,11,12,13,14,15"])
def test_judge_backfill_noop_records_selected_gpu_domain(tmp_path, physical_gpus: str):
    helpers = load_script("run_bcp_posttrain_eval_supervisor.py")
    run_root = tmp_path / "run"
    run_root.mkdir()
    control = tmp_path / "control"

    audit = helpers.backfill_missing_qwen_judges(
        repo=ROOT,
        run_root=run_root,
        control=control,
        env={"CUDA_VISIBLE_DEVICES": physical_gpus},
        log=tmp_path / "supervisor.log",
        judge_model=tmp_path / "unused-model",
        judge_port=9999,
        judge_workers=1,
        physical_gpus=physical_gpus,
    )

    assert audit["state"] == "no_op"
    assert audit["physical_gpus"] == physical_gpus
    assert json.loads((control / "judge_backfill_audit.json").read_text())["physical_gpus"] == physical_gpus


def test_hotpotqa_official_style_answer_em_and_f1():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    exact, extracted, details = evaluator.score_hotpotqa("The Eiffel Tower", "<think>x</think>The Eiffel Tower")
    assert exact == 1
    assert extracted == "The Eiffel Tower"
    assert details["f1"] == 1.0

    exact, _, details = evaluator.score_hotpotqa("Eiffel Tower Paris", "Eiffel Tower")
    assert exact == 0
    assert details["f1"] == pytest.approx(0.8)


def test_eval_resume_allows_only_nondecreasing_generation_limit():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    assert evaluator._nondecreasing_generation_limit(768, 768)
    assert evaluator._nondecreasing_generation_limit(768, 2048)
    assert not evaluator._nondecreasing_generation_limit(2048, 768)
    assert not evaluator._nondecreasing_generation_limit(None, 2048)


def test_hotpotqa_accuracy_artifacts_include_answer_f1(tmp_path: Path):
    evaluator = load_script("mas_loop_two_stage_eval.py")
    summary = {
        "num_total": 2,
        "num_fully_completed": 2,
        "num_with_errors": 0,
        "orchestra_only": {"completed": 2, "scored": 2, "correct": 1, "accuracy": 0.5, "answer_f1": 0.75},
        "orchestra_editor": {"completed": 2, "scored": 2, "correct": 2, "accuracy": 1.0, "answer_f1": 0.9},
        "delta_accuracy": 0.5,
    }
    evaluator._write_accuracy_artifacts(tmp_path, "hotpotqa", summary)
    compact = json.loads((tmp_path / "accuracy.json").read_text(encoding="utf-8"))
    assert compact["orchestra_only"]["answer_f1"] == 0.75
    assert "answer_f1" in (tmp_path / "accuracy.csv").read_text(encoding="utf-8").splitlines()[0]


def test_hotpotqa_and_bcp_use_same_checkpoint_root_with_same_step_only():
    matrix = load_script("run_all_mas_loop_checkpoint_matrix_eval.py")
    baseline = "/models/base"
    checkpoint_paths = {"o1": "/run/o1", "o2": "/run/o2", "e1": "/run/e1", "e2": "/run/e2"}

    def checkpoints(dataset: str):
        return [
            matrix.Checkpoint(dataset, "orchestra", 1, 1, checkpoint_paths["o1"], "/run"),
            matrix.Checkpoint(dataset, "orchestra", 1, 2, checkpoint_paths["o2"], "/run"),
            matrix.Checkpoint(dataset, "editor", 1, 1, checkpoint_paths["e1"], "/run"),
            matrix.Checkpoint(dataset, "editor", 1, 2, checkpoint_paths["e2"], "/run"),
        ]

    hotpot_jobs = matrix.build_jobs("hotpotqa", baseline, checkpoints("hotpotqa"), "same-step")
    bcp_jobs = matrix.build_jobs("bcp", baseline, checkpoints("bcp"), "same-step")
    for jobs in (hotpot_jobs, bcp_jobs):
        assert [job.kind for job in jobs] == ["baseline", "same_step", "same_step"]
        assert [(job.orchestra_step, job.editor_step) for job in jobs[1:]] == [(1, 1), (2, 2)]
        assert not any(job.kind == "cartesian" for job in jobs)
    assert [job.orchestra_source for job in hotpot_jobs] == [job.orchestra_source for job in bcp_jobs]
    assert [job.editor_source for job in hotpot_jobs] == [job.editor_source for job in bcp_jobs]


def test_hotpotqa_matrix_uses_independent_domain_and_enables_semantic_judge(tmp_path: Path):
    matrix = load_script("run_all_mas_loop_checkpoint_matrix_eval.py")
    job = matrix.EvalJob(
        dataset="hotpotqa", kind="baseline",
        orchestra_source_type="local", orchestra_source="/models/base",
        orchestra_round=None, orchestra_step=None,
        editor_source_type="local", editor_source="/models/base",
        editor_round=None, editor_step=None,
    )
    launcher, env = matrix.env_for_job(
        job, tmp_path / "hotpotqa", "/models/subagent", "0,1,2,3,4,5,6,7", 8710,
    )
    assert launcher.name == "run_hotpotqa_mas_loop_eval.sh"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert env["HOTPOTQA_EVAL_ORCHESTRA_PORT"] == "8710"
    assert env["HOTPOTQA_EVAL_SUBAGENT_PORT"] == "8712"
    assert env["HOTPOTQA_EVAL_BM25_BACKEND"] == "lucene_pyserini"
    common = (ROOT / "datasets" / "run_mas_loop_two_stage_eval.sh").read_text(encoding="utf-8")
    assert '( "${EVAL_DATASET}" == "bcp" || "${EVAL_DATASET}" == "hotpotqa" )' in common
    assert '"${EVAL_BCP_SCORE_MODE:-both}" != "exact"' in common


def test_hotpotqa_eval_wrapper_enforces_training_token_and_retrieval_contract():
    wrapper = (ROOT / "datasets" / "run_hotpotqa_mas_loop_eval.sh").read_text(encoding="utf-8")

    for fixed_export in (
        "export EVAL_ORCHESTRA_MAX_TOKENS=8192",
        "export EVAL_EDITOR_MAX_TOKENS=8192",
        "export EVAL_SUBAGENT_MAX_TOKENS=8192",
        "export EVAL_BCP_JUDGE_MAX_TOKENS=4096",
        "export MAS_LOOP_BCP_RETRIEVAL_PROFILE=legacy",
        "export MAS_LOOP_BCP_SEARCH_ROUNDS=1",
        "export MAS_LOOP_BCP_SEARCH_PLANNER_MAX_TOKENS=64",
    ):
        assert fixed_export in wrapper

    for stale_default in (
        "HOTPOTQA_EVAL_EDITOR_MAX_TOKENS:-8192",
        "HOTPOTQA_EVAL_RETRIEVAL_PROFILE:-controller",
        "HOTPOTQA_EVAL_SEARCH_ROUNDS:-5",
        "HOTPOTQA_EVAL_SEARCH_PLANNER_MAX_TOKENS:-256",
    ):
        assert stale_default not in wrapper
    assert "ignoring stale ${token_name}" in wrapper
    assert "ignoring stale ${retrieval_name}" in wrapper


def test_hotpotqa_training_launcher_forces_fixed_batch_and_vllm_output_tokens():
    launcher = (ROOT / "verl" / "scripts" / "run_hotpotqa_mas_loop_full.sh").read_text(encoding="utf-8")
    assert "TRAIN_BATCH=512" in launcher
    assert "ROLLOUT_N=16" in launcher
    assert 'DS_TRAIN_BATCH_SIZE="${TRAIN_BATCH}"' in launcher
    assert 'DS_EDITOR_TRAIN_BATCH_SIZE="${EDITOR_BATCH}"' in launcher
    assert "DS_MAX_AGENT_TOKENS=8192 DS_EDITOR_MAX_TOKENS=8192 DS_JUDGE_MAX_TOKENS=4096" in launcher
    assert "HP_MAX_AGENT_TOKENS:8192" in launcher
    assert "HP_EDITOR_MAX_TOKENS:8192" in launcher
    assert "HP_JUDGE_MAX_TOKENS:4096" in launcher


def test_hotpotqa_training_launcher_exposes_independent_lrs_and_two_gpu_domains():
    launcher = (ROOT / "verl" / "scripts" / "run_hotpotqa_mas_loop_full.sh").read_text(encoding="utf-8")
    assert 'HP_REFINER_LR=${HP_REFINER_LR:-${HP_EDITOR_LR:-1e-6}}' in launcher
    assert 'HP_ORCH_LR=${HP_ORCH_LR:-5e-7}' in launcher
    assert 'DS_ORCH_LR="${HP_ORCH_LR}" DS_EDITOR_LR="${HP_REFINER_LR}"' in launcher
    assert '0,1,2,3,4,5,6,7) DEFAULT_NAMESPACE=87; DOMAIN_TAG=gpu0-7' in launcher
    assert '8,9,10,11,12,13,14,15) DEFAULT_NAMESPACE=88; DOMAIN_TAG=gpu8-15' in launcher
    assert 'runs/hotpotqa/${TS}/${DOMAIN_TAG}' in launcher
    for isolated_name in ("HP_PORT_NAMESPACE", "HP_RAY_TEMP_DIR", "HP_CONTROL_DIR", "HP_RUN_ROOT", "HP_CKPT_ROOT"):
        assert isolated_name in launcher


def _minimal_eval_row():
    return {
        "task": "Were Alice and Bob born in the same country?",
        "gold_answer": "no",
        "index": 0,
        "sample_id": "hp-0",
        "source_row": {},
    }


def _minimal_exec(final_output: str):
    return {"trajectory": {"final_output": final_output}, "success": True}


def test_hotpotqa_build_record_uses_semantic_primary_and_keeps_official_diagnostics():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    row = _minimal_eval_row()
    judge = {"score": 1.0, "correct": "yes", "prompt_version": "hotpotqa_semantic_judge_v1"}
    record = evaluator._build_record(
        row,
        {"mas": {"agents": [], "edges": [], "aggregation": ""}},
        _minimal_exec("No, they were born in different countries."),
        None,
        None,
        judge,
        None,
        [],
        evaluator.score_hotpotqa,
        {"dataset": "hotpotqa", "models": {}, "semantic_score_mode": "both"},
    )
    stage = record["orchestra_only"]
    assert stage["score"] == 1
    assert stage["semantic_judge"] is judge
    assert stage["exact_match_score"] == 0
    assert isinstance(stage["hotpotqa_f1"], float)


def test_hotpotqa_missing_judge_never_falls_back_to_em():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    row = _minimal_eval_row()
    record = evaluator._build_record(
        row,
        {"mas": {"agents": [], "edges": [], "aggregation": ""}},
        _minimal_exec("no"),
        None,
        None,
        None,
        None,
        [{"stage": "orchestra_semantic_judge", "message": "judge unavailable"}],
        evaluator.score_hotpotqa,
        {"dataset": "hotpotqa", "models": {}, "semantic_score_mode": "both"},
    )
    stage = record["orchestra_only"]
    assert stage["exact_match_score"] == 1
    assert stage["score"] is None
    assert stage["correct"] is None


def test_aime_build_record_remains_rule_scored():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    row = {**_minimal_eval_row(), "task": "What is 2+2?", "gold_answer": "4"}
    record = evaluator._build_record(
        row,
        {"mas": {"agents": [], "edges": [], "aggregation": ""}},
        _minimal_exec("\\boxed{4}"),
        None,
        None,
        None,
        None,
        [],
        evaluator.score_aime,
        {"dataset": "aime", "models": {}, "semantic_score_mode": "both"},
    )
    assert record["orchestra_only"]["score"] == 1


class _FailIfExecuted:
    def execute(self, *args, **kwargs):
        raise AssertionError("accepted editor action must not re-execute the stochastic MAS")


def test_editor_accept_reuses_source_execution_without_resampling():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    source_execution = {
        "trajectory": {"final_output": r"\\boxed{B}", "trace": ["original"]},
        "execution_elapsed_sec": 1.25,
    }
    editor = {
        "patch": {
            "operations": [{"op": "accept"}],
            "translation": {"status": "accepted", "applied_operations": []},
        },
        "updated_mas": {"agents": []},
    }
    row = {"task": "question", "gold_answer": "B", "index": 0, "sample_id": "0"}

    result = evaluator._execute_editor_result(_FailIfExecuted(), row, editor, source_execution)

    assert result["trajectory"] is source_execution["trajectory"]
    assert result["execution_elapsed_sec"] == 0.0
    assert result["execution_reused"] is True
    assert result["execution_source_stage"] == "orchestra_only"
    assert result["reuse_reason"] == "editor_no_applied_operations"


def test_editor_compiler_noop_reuses_source_execution():
    evaluator = load_script("mas_loop_two_stage_eval.py")
    editor = {
        "patch": {
            "operations": [{"op": "rewrite_role", "agent_id": "missing", "role": "verify"}],
            "translation": {"status": "no_change", "applied_operations": []},
        }
    }
    assert evaluator._editor_reuses_source_execution(editor) is True


def test_editor_applied_operation_still_executes_updated_mas():
    evaluator = load_script("mas_loop_two_stage_eval.py")

    class CapturingExecutor:
        def __init__(self):
            self.calls = []

        def execute(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"final_output": r"\\boxed{C}"}

    executor = CapturingExecutor()
    editor = {
        "patch": {
            "translation": {
                "status": "applied",
                "applied_operations": [{"op": "rewrite_role", "agent_id": "solver"}],
            }
        },
        "updated_mas": {"agents": [{"id": "solver", "role": "verify"}]},
    }
    source_execution = {"trajectory": {"final_output": r"\\boxed{B}"}}
    row = {"task": "question", "gold_answer": "C", "index": 1, "sample_id": "1"}

    result = evaluator._execute_editor_result(executor, row, editor, source_execution)

    assert len(executor.calls) == 1
    assert result["trajectory"]["final_output"] == r"\\boxed{C}"
    assert result["execution_reused"] is False


@pytest.mark.parametrize("dataset", ["aime", "gpqa", "gpqa_diamond", "gpqa-diamond"])
def test_reasoning_eval_normalization_forcibly_disables_retrieval(dataset):
    evaluator = load_script("mas_loop_two_stage_eval.py")
    mas = evaluator._normalize_eval_mas({
        "agents": [
            {
                "id": "legacy_searcher",
                "role": "Search for an answer.",
                "tools": ["bm25"],
                "budget": 1,
                "execution_mode": "multi_turn_search",
            },
            {
                "id": "final",
                "role": "Return the answer.",
                "tools": ["bm25"],
                "budget": 1,
                "execution_mode": "llm_only",
            },
        ],
        "edges": [{"from": "legacy_searcher", "to": "final", "type": "data_flow"}],
        "aggregation": "final",
    }, dataset)

    assert all(agent["execution_mode"] == "llm_only" for agent in mas["agents"])
    assert all(agent["tools"] == [] for agent in mas["agents"])


@pytest.mark.parametrize("dataset", ["aime", "gpqa"])
def test_reasoning_eval_prompts_use_parallel_llm_only_graph_without_bm25(dataset):
    from mas_loop_rl.prompts import orchestra_prompt

    prompt = orchestra_prompt("Solve this benchmark item.", {"evaluation_dataset": dataset, "max_agents": 5})
    text = "\n".join(message["content"] for message in prompt)
    example = text.split("Example structure:\n", 1)[1]

    expected_agents = 5 if dataset == "aime" else 3
    assert f"Use exactly {expected_agents} complementary llm_only agents" in text
    assert 'execution_mode="multi_turn_search"' not in example
    assert "<tool>bm25</tool>" not in example
    assert "\x08" not in text
    assert r"\boxed" in example
    assert example.count('execution_mode="llm_only"') == expected_agents
    assert '<edge from="agent_0" to="agent_1"' not in example
    assert '<edge from="agent_1" to="agent_0"' not in example
    if dataset == "aime":
        assert 'aggregation="agent_4"' in example
        for source in range(3):
            for target in range(3):
                if source != target:
                    assert f'<edge from="agent_{source}" to="agent_{target}"' not in example
        assert "after checking" in text.lower()
        assert "conditioned sample space" in text.lower()
    else:
        assert 'aggregation="agent_2"' in example
        assert '<edge from="agent_0" to="agent_2" type="data_flow" />' in example
        assert '<edge from="agent_1" to="agent_2" type="data_flow" />' in example
        assert example.count('<edge ') == 2
        assert "aorchestra" in text.lower()
        assert "main reviewer" in text.lower()
        assert "do not add an audit" in text.lower()
        assert "preserve" in text.lower()


def test_reasoning_editor_preserve_policy_builds_zero_cost_accept(monkeypatch):
    evaluator = load_script("mas_loop_two_stage_eval.py")
    monkeypatch.setenv("MAS_LOOP_REASONING_EDITOR_POLICY", "preserve")
    source = {"mas": {"agents": [], "edges": [], "aggregation": ""}}

    assert evaluator._reasoning_editor_preserve_enabled("gpqa")
    editor = evaluator._build_reasoning_preserve_editor(source)

    assert editor["updated_mas"] is source["mas"]
    assert editor["generation_skipped"] is True
    assert editor["generation_elapsed_sec"] == 0.0
    assert editor["patch"]["translation"]["applied_operations"] == []
    assert evaluator._editor_reuses_source_execution(editor)


def test_reasoning_editor_preserve_policy_reuses_source(monkeypatch):
    evaluator = load_script("mas_loop_two_stage_eval.py")
    monkeypatch.setenv("MAS_LOOP_REASONING_EDITOR_POLICY", "preserve")

    class FailingExecutor:
        def execute(self, *args, **kwargs):
            raise AssertionError("reasoning preserve policy must not re-execute the updated MAS")

    source = {"trajectory": {"final_output": r"\boxed{117}"}}
    editor = {
        "patch": {
            "translation": {
                "status": "applied",
                "applied_operations": [
                    {"op": "rewrite_role", "agent_id": "final", "role": "retry"}
                ],
            }
        },
        "updated_mas": {"agents": [{"id": "final", "role": "retry"}]},
    }
    result = evaluator._execute_editor_result(
        FailingExecutor(), {"dataset": "aime", "task": "q", "gold_answer": "117"}, editor, source
    )

    assert result["trajectory"] is source["trajectory"]
    assert result["execution_reused"] is True
    assert result["reuse_reason"] == "reasoning_dataset_editor_preserve_policy"
