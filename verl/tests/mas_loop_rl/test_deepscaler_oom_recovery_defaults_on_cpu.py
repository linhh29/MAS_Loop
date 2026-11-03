from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_deepscaler_defaults_reserve_sampler_headroom() -> None:
    text = (REPO_ROOT / "verl/scripts/run_deepscaler_mas_loop_full.sh").read_text()
    assert "DS_ROLLOUT_GPU_MEM_UTIL=${DS_ROLLOUT_GPU_MEM_UTIL:-0.80}" in text
    assert "DS_ROLLOUT_MAX_NUM_SEQS=${DS_ROLLOUT_MAX_NUM_SEQS:-128}" in text
    assert "DS_ORCHESTRA_VLLM_MAX_ACTIVE=${DS_ORCHESTRA_VLLM_MAX_ACTIVE:-128}" in text
    assert "DS_RAY_SERVER_MAX_CONCURRENCY=${DS_RAY_SERVER_MAX_CONCURRENCY:-1024}" in text
    assert "DS_RAY_GENERATE_MAX_OUTSTANDING=${DS_RAY_GENERATE_MAX_OUTSTANDING:-128}" in text
    assert "DS_ORCHESTRA_VLLM_MAX_ACTIVE <= DS_ROLLOUT_MAX_NUM_SEQS" in text


def test_failure_cleanup_is_bounded_and_ray_session_scoped() -> None:
    text = (REPO_ROOT / "verl/scripts/run_mas_loop_adversarial_grpo.sh").read_text()
    assert "terminate_pid_bounded" in text
    assert 'terminate_pid_bounded "${COORDINATOR_PID}" phase-coordinator 15' in text
    assert "ray_temp_owned_pids" in text
    assert '"/proc/${pid}/fd"/*' in text
    assert '"${sid}" == "${owner_sid}"' in text
    assert "ray stop --force" in text  # present only in the explicit prohibition comment


def test_watchdog_detects_integrated_engine_death() -> None:
    text = (REPO_ROOT / "verl/scripts/run_mas_loop_adversarial_grpo.sh").read_text()
    assert "EngineCore fatal error|EngineDeadError|CUDA out of memory" in text
    assert 'start_service_watchdog "$trainer_pid" "$round_log" "$phase_ray_dir"' in text
