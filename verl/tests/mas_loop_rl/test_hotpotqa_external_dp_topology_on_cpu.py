from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _text(relative: str) -> str:
    return (REPO_ROOT / relative).read_text()


def test_generic_external_servers_support_role_specific_tp_dp() -> None:
    text = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")
    for role in ("PEER", "SUB_AGENT", "JUDGE"):
        assert f"{role}_TENSOR_PARALLEL_SIZE=" in text
        assert f"{role}_DATA_PARALLEL_SIZE=" in text
    assert 'parallel_args=(--tensor-parallel-size "${tensor_parallel_size}")' in text
    assert '--data-parallel-size "${data_parallel_size}"' in text
    assert '--data-parallel-size-local "${data_parallel_size}"' in text
    assert "tp * dp != N_GPUS" in text


def test_hotpotqa_uses_eight_real_external_replicas_by_default() -> None:
    text = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    for role in ("PEER", "SUBAGENT", "JUDGE"):
        assert f"HP_{role}_TENSOR_PARALLEL_SIZE=${{HP_{role}_TENSOR_PARALLEL_SIZE:-1}}" in text
        assert f"HP_{role}_DATA_PARALLEL_SIZE=${{HP_{role}_DATA_PARALLEL_SIZE:-8}}" in text
    assert 'PEER_TENSOR_PARALLEL_SIZE="${HP_PEER_TENSOR_PARALLEL_SIZE}"' in text
    assert 'SUB_AGENT_TENSOR_PARALLEL_SIZE="${HP_SUBAGENT_TENSOR_PARALLEL_SIZE}"' in text
    assert 'JUDGE_TENSOR_PARALLEL_SIZE="${HP_JUDGE_TENSOR_PARALLEL_SIZE}"' in text
    assert "cached, deduplicated, or shared" in text


def test_hotpotqa_fixed_trajectory_and_token_contract_is_unchanged() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")
    assert "TRAIN_BATCH=512" in hotpot
    assert "ROLLOUT_N=16" in hotpot
    assert "EDITOR_BATCH=$((TRAIN_BATCH * ROLLOUT_N))" in hotpot
    assert "MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-8192}" in generic
    assert "export MAS_LOOP_JUDGE_MAX_TOKENS=${MAS_LOOP_JUDGE_MAX_TOKENS:-4096}" in generic
    assert "export MAS_LOOP_MAX_AGENT_TOKENS=${MAS_LOOP_MAX_AGENT_TOKENS:-8192}" in generic


def test_hotpotqa_exposes_2048_resident_trajectory_ceiling() -> None:
    text = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    assert "HP_AGENT_LOOP_WORKER_PROCESSES=${HP_AGENT_LOOP_WORKER_PROCESSES:-4}" in text
    assert "HP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER=${HP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER:-512}" in text
    assert 'MAS_LOOP_AGENT_LOOP_WORKER_PROCESSES="${HP_AGENT_LOOP_WORKER_PROCESSES}"' in text
    assert 'MAS_LOOP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER="${HP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER}"' in text
    assert "HP_RAY_GENERATE_MAX_OUTSTANDING=${HP_RAY_GENERATE_MAX_OUTSTANDING:-1024}" in text
    assert 'DS_RAY_GENERATE_MAX_OUTSTANDING="${HP_RAY_GENERATE_MAX_OUTSTANDING}"' in text
    assert "EDITOR_RAY_CAP=${HP_RAY_GENERATE_MAX_OUTSTANDING}" in text
    assert '"ray_generate:${EDITOR_RAY_CAP}"' in text


def test_hotpotqa_editor_defaults_fill_dp8_scheduler_without_result_reuse() -> None:
    text = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    assert "HP_REFINER_VLLM_MAX_ACTIVE=${HP_REFINER_VLLM_MAX_ACTIVE:-1024}" in text
    assert "HP_REFINER_MAX_NUM_SEQS=${HP_REFINER_MAX_NUM_SEQS:-128}" in text
    assert "HP_REFINER_CANDIDATE_CONCURRENCY=${HP_REFINER_CANDIDATE_CONCURRENCY:-1024}" in text
    assert "HP_REFINER_HTTP_POOL_MAXSIZE=${HP_REFINER_HTTP_POOL_MAXSIZE:-1024}" in text
    assert 'DS_EDITOR_VLLM_MAX_ACTIVE="${HP_REFINER_VLLM_MAX_ACTIVE}"' in text
    assert 'DS_PEER_MAX_NUM_SEQS="${HP_REFINER_MAX_NUM_SEQS}"' in text
    assert 'DS_EDITOR_CANDIDATE_CONCURRENCY="${HP_REFINER_CANDIDATE_CONCURRENCY}"' in text
    assert 'DS_EDITOR_HTTP_POOL_MAXSIZE="${HP_REFINER_HTTP_POOL_MAXSIZE}"' in text
    assert "no cache, deduplication, or result" in text


def test_hotpotqa_vllm_v1_disables_concurrent_editor_partial_prefill() -> None:
    text = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    assert "HP_REFINER_MAX_NUM_PARTIAL_PREFILLS=${HP_REFINER_MAX_NUM_PARTIAL_PREFILLS:-1}" in text
    assert "HP_REFINER_MAX_LONG_PARTIAL_PREFILLS=${HP_REFINER_MAX_LONG_PARTIAL_PREFILLS:-1}" in text
    assert "vLLM 0.10.0 V1 does not support concurrent partial prefill" in text
    assert "HP_REFINER_MAX_NUM_PARTIAL_PREFILLS != 1" in text
    assert "HP_REFINER_MAX_LONG_PARTIAL_PREFILLS != 1" in text


def test_hotpotqa_propagates_validated_fsdp_offload_controls() -> None:
    text = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    assert "HP_ACTOR_PARAM_OFFLOAD=${HP_ACTOR_PARAM_OFFLOAD:-${ACTOR_PARAM_OFFLOAD:-True}}" in text
    assert "HP_ACTOR_OPTIMIZER_OFFLOAD=${HP_ACTOR_OPTIMIZER_OFFLOAD:-${ACTOR_OPTIMIZER_OFFLOAD:-True}}" in text
    assert "HP_REF_PARAM_OFFLOAD=${HP_REF_PARAM_OFFLOAD:-${REF_PARAM_OFFLOAD:-True}}" in text
    assert 'DS_ACTOR_PARAM_OFFLOAD="${HP_ACTOR_PARAM_OFFLOAD}"' in text
    assert 'DS_ACTOR_OPTIMIZER_OFFLOAD="${HP_ACTOR_OPTIMIZER_OFFLOAD}"' in text
    assert 'DS_REF_PARAM_OFFLOAD="${HP_REF_PARAM_OFFLOAD}"' in text


def test_adapter_validates_editor_admission_against_external_dp_capacity() -> None:
    adapter = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    assert "DS_PEER_DATA_PARALLEL_SIZE=${PEER_DATA_PARALLEL_SIZE:-1}" in adapter
    assert "DS_EDITOR_SCHEDULER_CAPACITY=$((DS_PEER_DATA_PARALLEL_SIZE * DS_PEER_MAX_NUM_SEQS))" in adapter
    assert "DS_EDITOR_VLLM_MAX_ACTIVE <= DS_EDITOR_SCHEDULER_CAPACITY" in adapter
    assert "DS_EDITOR_VLLM_MAX_ACTIVE <= DS_ROLLOUT_MAX_NUM_SEQS" not in adapter


def test_hotpotqa_uses_real_dag_parallelism_with_prefix_cache_disabled() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    adapter = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")
    client = _text("verl/mas_loop_rl/local_client.py")
    assert "MAS_LOOP_DEPENDENCY_PROMPT_MODE=declared_edges" in hotpot
    assert "MAS_LOOP_DAG_AGENT_PARALLEL=1" in hotpot
    assert 'MAS_LOOP_BCP_SEARCH_ROUNDS="${HP_BCP_SEARCH_ROUNDS:-1}"' in hotpot
    assert 'MAS_LOOP_BCP_SEARCH_PLANNER_MAX_TOKENS="${HP_BCP_SEARCH_PLANNER_MAX_TOKENS:-64}"' in hotpot
    assert "ENABLE_PREFIX_CACHING" not in hotpot
    assert "ENABLE_PREFIX_CACHING" not in adapter
    assert "PEER_ENABLE_PREFIX_CACHING" not in generic
    assert "--enable-prefix-caching" not in generic
    assert "--no-enable-prefix-caching" in generic
    assert "prefix_cache=disabled" in generic
    assert "actor_rollout_ref.rollout.enable_prefix_caching=False" in generic
    # Cache reset before sleep is lifecycle cleanup, not prefix-cache reuse.
    assert "self.reset_prefix_cache()" in client


def test_hotpotqa_propagates_configurable_vllm_startup_wait() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    adapter = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")
    assert "HP_VLLM_STARTUP_ATTEMPTS=${HP_VLLM_STARTUP_ATTEMPTS:-720}" in hotpot
    assert 'DS_VLLM_STARTUP_ATTEMPTS="${HP_VLLM_STARTUP_ATTEMPTS}"' in hotpot
    assert "DS_VLLM_STARTUP_ATTEMPTS=${DS_VLLM_STARTUP_ATTEMPTS:-720}" in adapter
    assert 'MAS_LOOP_VLLM_STARTUP_ATTEMPTS="${DS_VLLM_STARTUP_ATTEMPTS}"' in adapter
    assert "MAS_LOOP_VLLM_STARTUP_ATTEMPTS=${MAS_LOOP_VLLM_STARTUP_ATTEMPTS:-720}" in generic
    assert "for ((i=1; i<=MAS_LOOP_VLLM_STARTUP_ATTEMPTS; i++))" in generic
    assert "for i in {1..180}" not in generic


def test_hotpotqa_propagates_dp_sleep_quiescence_and_uses_managed_smoke() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    adapter = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")
    client = _text("verl/mas_loop_rl/local_client.py")

    assert "HP_PHASE_LEASE_GRACE_SEC=${HP_PHASE_LEASE_GRACE_SEC:-3.0}" in hotpot
    assert "HP_VLLM_DP_QUIESCE_SEC=${HP_VLLM_DP_QUIESCE_SEC:-3.0}" in hotpot
    assert 'DS_PHASE_LEASE_GRACE_SEC="${HP_PHASE_LEASE_GRACE_SEC}"' in hotpot
    assert 'DS_VLLM_DP_QUIESCE_SEC="${HP_VLLM_DP_QUIESCE_SEC}"' in hotpot
    assert "DS_VLLM_DP_QUIESCE_SEC=${DS_VLLM_DP_QUIESCE_SEC:-0}" in adapter
    assert 'MAS_LOOP_VLLM_DP_QUIESCE_SEC="${DS_VLLM_DP_QUIESCE_SEC}"' in adapter
    assert "MAS_LOOP_VLLM_DP_QUIESCE_SEC=${MAS_LOOP_VLLM_DP_QUIESCE_SEC:-0}" in generic
    assert "result = BCPJudge(manage_sleep=True).judge(" in generic
    assert "result = BCPJudge(manage_sleep=False).judge(" not in generic
    assert "self.dp_quiesce_sec" in client
    assert "await asyncio.sleep(self.dp_quiesce_sec)" in client


def test_hotpotqa_launcher_fails_closed_without_gpu_but_allows_config_only() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    assert 'if [[ "${HP_CONFIG_ONLY:-0}" != "1" ]]; then' in hotpot
    assert "command -v ppu-smi" in hotpot
    assert "timeout --kill-after=2s 15s ppu-smi" in hotpot
    assert 'echo "[HP] PPU driver/device is unavailable' in hotpot
    assert 'exit 69' in hotpot


def test_hotpotqa_bounds_external_vllm_cpu_pools_per_numa_domain() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    adapter = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")

    assert "DEFAULT_CPUSET=0-47,96-143" in hotpot
    assert "DEFAULT_CPUSET=48-95,144-191" in hotpot
    assert "HP_EXTERNAL_VLLM_CPU_THREADS=${HP_EXTERNAL_VLLM_CPU_THREADS:-4}" in hotpot
    assert 'DS_EXTERNAL_VLLM_CPU_THREADS="${HP_EXTERNAL_VLLM_CPU_THREADS}"' in hotpot
    assert 'MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS="${DS_EXTERNAL_VLLM_CPU_THREADS}"' in adapter
    assert 'launch_cmd=(taskset --cpu-list "${MAS_LOOP_EXTERNAL_VLLM_CPUSET}"' in generic
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
        assert f'"{name}=${{MAS_LOOP_EXTERNAL_VLLM_CPU_THREADS}}"' in generic
    assert '"TOKENIZERS_PARALLELISM=false"' in generic


def test_hotpotqa_isolates_internal_vllm_ports_and_skips_duplicate_bootstrap() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    adapter = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")

    assert "DEFAULT_VLLM_PORT_BASE=41000" in hotpot
    assert "DEFAULT_VLLM_PORT_BASE=51000" in hotpot
    assert "HP_EXTERNAL_VLLM_PORT_BASE=${HP_EXTERNAL_VLLM_PORT_BASE:-${DEFAULT_VLLM_PORT_BASE}}" in hotpot
    assert 'DS_EXTERNAL_VLLM_PORT_BASE="${HP_EXTERNAL_VLLM_PORT_BASE}"' in hotpot
    assert "DS_EXTERNAL_VLLM_PORT_BASE=${DS_EXTERNAL_VLLM_PORT_BASE:-}" in adapter
    assert 'MAS_LOOP_EXTERNAL_VLLM_PORT_BASE="${DS_EXTERNAL_VLLM_PORT_BASE}"' in adapter
    assert "MAS_LOOP_BOOTSTRAP_V1_DEPS=0" in adapter
    assert 'launch_cmd+=("VLLM_PORT=${MAS_LOOP_EXTERNAL_VLLM_PORT_BASE}")' in generic



def test_hotpotqa_bounds_reward_worker_replication_and_ray_native_threads() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    adapter = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")

    assert "DS_REWARD_NUM_WORKERS=\"${HP_REWARD_NUM_WORKERS:-64}\"" in hotpot
    assert "HP_RAY_WORKER_CPU_THREADS=${HP_RAY_WORKER_CPU_THREADS:-1}" in hotpot
    assert "DS_RAY_WORKER_CPU_THREADS=\"${HP_RAY_WORKER_CPU_THREADS}\"" in hotpot
    assert "DS_RAY_WORKER_CPU_THREADS=${DS_RAY_WORKER_CPU_THREADS:-1}" in adapter
    assert "MAS_LOOP_RAY_WORKER_CPU_THREADS=\"${DS_RAY_WORKER_CPU_THREADS}\"" in adapter
    assert "export MAS_LOOP_RAY_WORKER_CPU_THREADS=${MAS_LOOP_RAY_WORKER_CPU_THREADS:-1}" in generic
    for name in (
        "OMP_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_NUM_THREADS",
        "MKL_DYNAMIC",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
    ):
        assert generic.count(f"+ray_kwargs.ray_init.runtime_env.env_vars.{name}=") == 2



def test_hotpotqa_reuses_identical_subagent_judge_engine_without_result_reuse() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    adapter = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")

    assert "HP_SHARE_SUBAGENT_JUDGE_VLLM=${HP_SHARE_SUBAGENT_JUDGE_VLLM:-1}" in hotpot
    assert "JUDGE_PORT=${HP_JUDGE_PORT:-${SUBAGENT_PORT}}" in hotpot
    assert "DS_SHARE_SUBAGENT_JUDGE_VLLM=\"${HP_SHARE_SUBAGENT_JUDGE_VLLM}\"" in hotpot
    assert "DS_SHARE_SUBAGENT_JUDGE_VLLM=${DS_SHARE_SUBAGENT_JUDGE_VLLM:-0}" in adapter
    assert "MAS_LOOP_SHARE_SUBAGENT_JUDGE_VLLM=\"${DS_SHARE_SUBAGENT_JUDGE_VLLM}\"" in adapter
    assert "served_model_names+=(\"judge\")" in generic
    assert "--served-model-name \"${served_model_names[@]}\"" in generic
    assert "Requests and outputs are never cached or deduplicated" in generic


def test_hotpotqa_propagates_bounded_joint_step_runs() -> None:
    hotpot = _text("verl/scripts/run_hotpotqa_mas_loop_full.sh")
    adapter = _text("verl/scripts/run_deepscaler_mas_loop_full.sh")
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")
    assert "HP_MAX_JOINT_STEPS_THIS_RUN=${HP_MAX_JOINT_STEPS_THIS_RUN:-0}" in hotpot
    assert 'DS_MAX_JOINT_STEPS_THIS_RUN="${HP_MAX_JOINT_STEPS_THIS_RUN}"' in hotpot
    assert "DS_MAX_JOINT_STEPS_THIS_RUN=${DS_MAX_JOINT_STEPS_THIS_RUN:-0}" in adapter
    assert 'MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN="${DS_MAX_JOINT_STEPS_THIS_RUN}"' in adapter
    assert "MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN=${MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN:-0}" in generic

def test_external_vllm_watchdog_detects_dead_dp_children() -> None:
    generic = _text("verl/scripts/run_mas_loop_adversarial_grpo.sh")
    assert "vllm_zombie_child(){" in generic
    assert '"/proc/${server_pid}/task/${server_pid}/children"' in generic
    assert '"/proc/${child}/status"' in generic
    assert '"${state}" == "Z"' in generic
    assert "editor-peer DP child became zombie" in generic
    assert "subagent DP child became zombie" in generic
    assert "judge DP child became zombie" in generic
    # The check must stay scoped to children of the owned API parent rather than
    # matching process names that could belong to another independent GPU run.
    function = generic.split("vllm_zombie_child(){", 1)[1].split("\n}", 1)[0]
    assert "pgrep" not in function
    assert "pkill" not in function
