from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _text(relative: str) -> str:
    return (REPO_ROOT / relative).read_text()


def test_bcp_training_subagent_has_8192_completion_budget() -> None:
    text = _text("verl/scripts/run_bcp_mas_loop_full.sh")
    assert "BCP_MAX_AGENT_TOKENS=${BCP_MAX_AGENT_TOKENS:-8192}" in text
    assert 'MAS_LOOP_MAX_AGENT_TOKENS="${BCP_MAX_AGENT_TOKENS}"' in text
    assert "BCP_SUB_AGENT_MAX_MODEL_LEN=${BCP_SUB_AGENT_MAX_MODEL_LEN:-23192}" in text


def test_deepscaler_training_subagent_has_8192_completion_budget() -> None:
    text = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    assert "DS_MAX_AGENT_TOKENS=${DS_MAX_AGENT_TOKENS:-8192}" in text
    assert 'MAS_LOOP_MAX_AGENT_TOKENS="${DS_MAX_AGENT_TOKENS}"' in text
    assert "DS_SUB_AGENT_MAX_MODEL_LEN=${DS_SUB_AGENT_MAX_MODEL_LEN:-23192}" in text


def test_generic_trainer_fallback_is_8192() -> None:
    text = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")
    assert "export MAS_LOOP_MAX_AGENT_TOKENS=${MAS_LOOP_MAX_AGENT_TOKENS:-8192}" in text


def test_checkpoint_evaluations_use_same_subagent_budget() -> None:
    bcp = _text("datasets/run_bcp_mas_loop_eval.sh")
    ds = _text("datasets/run_deepscaler_aime_mas_loop_eval.sh")
    common = _text("datasets/run_mas_loop_two_stage_eval.sh")
    assert "EVAL_SUBAGENT_MAX_TOKENS=${BCP_EVAL_SUBAGENT_MAX_TOKENS:-8192}" in bcp
    assert "EVAL_SUBAGENT_MAX_MODEL_LEN=${BCP_EVAL_SUBAGENT_MAX_MODEL_LEN:-32768}" in bcp
    assert "EVAL_SUBAGENT_MAX_TOKENS=${AIME_EVAL_SUBAGENT_MAX_TOKENS:-8192}" in ds
    assert "EVAL_SUBAGENT_MAX_MODEL_LEN=${AIME_EVAL_SUBAGENT_MAX_MODEL_LEN:-23192}" in ds
    assert "SUBAGENT_MAX_MODEL_LEN=${EVAL_SUBAGENT_MAX_MODEL_LEN:-23192}" in common
    assert common.count("EVAL_SUBAGENT_MAX_TOKENS:-8192") == 2


def test_checkpoint_evaluation_vllm_v1_disables_concurrent_partial_prefill() -> None:
    common = _text("datasets/run_mas_loop_two_stage_eval.sh")
    assert "export VLLM_SERVER_DEV_MODE=1" in common
    for role in ("ORCHESTRA", "EDITOR", "SUBAGENT"):
        assert f"EVAL_{role}_PARTIAL_PREFILLS=${{EVAL_{role}_PARTIAL_PREFILLS:-1}}" in common
        assert f"EVAL_{role}_LONG_PARTIAL_PREFILLS=${{EVAL_{role}_LONG_PARTIAL_PREFILLS:-1}}" in common
        assert f"(( EVAL_{role}_PARTIAL_PREFILLS > 1 ))" in common
    assert "vLLM V1 does not support concurrent partial prefill" in common


def test_checkpoint_evaluation_avoids_empty_array_expansion_under_nounset() -> None:
    common = _text("datasets/run_mas_loop_two_stage_eval.sh")
    assert "eval_cmd=(" in common
    assert '"${eval_cmd[@]}" 2>&1 | tee' in common
    assert '"${extra_args[@]}"' not in common


def test_checkpoint_evaluation_supports_validated_tp_dp_topology() -> None:
    common = _text("datasets/run_mas_loop_two_stage_eval.sh")
    bcp = _text("datasets/run_bcp_mas_loop_eval.sh")
    evaluator = _text("datasets/mas_loop_two_stage_eval.py")

    assert "EVAL_TENSOR_PARALLEL_SIZE=${EVAL_TENSOR_PARALLEL_SIZE:-${N_GPUS}}" in common
    assert "EVAL_DATA_PARALLEL_SIZE=${EVAL_DATA_PARALLEL_SIZE:-1}" in common
    assert "EVAL_TENSOR_PARALLEL_SIZE * EVAL_DATA_PARALLEL_SIZE != N_GPUS" in common
    assert '--tensor-parallel-size "${EVAL_TENSOR_PARALLEL_SIZE}"' in common
    assert '--data-parallel-size "${EVAL_DATA_PARALLEL_SIZE}"' in common
    assert '--data-parallel-size-local "${EVAL_DATA_PARALLEL_SIZE}"' in common
    assert "BCP_EVAL_TENSOR_PARALLEL_SIZE:-16" in bcp
    assert "BCP_EVAL_DATA_PARALLEL_SIZE:-1" in bcp
    assert '"tensor_parallel_size": int(os.getenv("EVAL_TENSOR_PARALLEL_SIZE"' in evaluator
    assert '"data_parallel_size": int(os.getenv("EVAL_DATA_PARALLEL_SIZE", "1"))' in evaluator
    assert '"world_size": int(os.getenv("EVAL_N_GPUS", "8"))' in evaluator
