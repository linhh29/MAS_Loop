from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_launcher_uses_strict_orchestra_then_editor_joint_steps() -> None:
    text = (REPO_ROOT / "scripts/run_mas_loop_adversarial_grpo.sh").read_text()
    joint = text[text.index("run_joint_training_round(){") :]

    orchestra_call = joint.index('run_orchestra_phase "$round"')
    replay_build = joint.index("python3 -m mas_loop_rl.build_editor_data_from_logs")
    editor_call = joint.index('run_editor_phase "$round"')
    manifest = joint.index('write_joint_step_manifest "$round"')
    assert orchestra_call < replay_build < editor_call < manifest
    assert '--optimizer-step "${step}"' in joint
    assert '--expected-groups "${EDITOR_TASK_GROUPS_PER_STEP}"' in joint
    assert '--expected-rows "${EDITOR_TRAIN_BATCH_SIZE}"' in joint
    assert joint.count("+trainer.max_steps_this_run=1") == 2
    assert joint.count("checkpoint.save_contents=\"['model','optimizer','extra','hf_model']\"") == 2
    assert joint.count("checkpoint.load_contents=\"['model','optimizer','extra']\"") == 2
    assert 'assert_latest_joint_checkpoint orchestra "$orchestra_root" "$step"' in joint
    assert 'assert_latest_joint_checkpoint editor "$editor_root" "$step"' in joint
    assert 'assert_joint_step_manifest_complete "${manifest}"' in joint
    assert "already complete; preserving its rollout/replay" in joint
    assert "next rollout will use O_${step} + E_${step}" in joint


def test_bounded_invocation_keeps_full_training_horizon() -> None:
    legacy = (REPO_ROOT / "verl/trainer/ppo/ray_trainer.py").read_text()
    active_v1 = (REPO_ROOT / "verl/trainer/ppo/v1/trainer_base.py").read_text()
    for text in (legacy, active_v1):
        assert 'max_steps_this_run = int(self.config.trainer.get("max_steps_this_run", 0) or 0)' in text
        assert "bounded_run_stop_step(" in text
        assert "is_last_step = self.global_steps >= run_stop_step" in text
    assert "self.global_steps <= run_stop_step" in active_v1


def test_bcp_wrapper_explicitly_enables_joint_step_training() -> None:
    text = (REPO_ROOT / "scripts/run_bcp_mas_loop_full.sh").read_text()
    assert "BCP_JOINT_STEP_TRAINING=${BCP_JOINT_STEP_TRAINING:-1}" in text
    assert 'JOINT_STEP_TRAINING="${BCP_JOINT_STEP_TRAINING}"' in text
    assert "BCP_ACTOR_TEMPERATURE=${BCP_ACTOR_TEMPERATURE:-0.7}" in text
    assert "BCP_EDITOR_TEMPERATURE=${BCP_EDITOR_TEMPERATURE:-0.7}" in text
    assert "BCP_ROLLOUT_TP=${BCP_ROLLOUT_TP:-8}" in text


def test_source_adversarial_v6_launcher_defaults_and_manifest_are_explicit() -> None:
    text = (REPO_ROOT / "scripts/run_mas_loop_adversarial_grpo.sh").read_text()
    assert "ROLLOUT_N=${ROLLOUT_N:-32}" in text
    assert "MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE=${MAS_LOOP_EDITOR_CANDIDATES_PER_SOURCE:-1}" in text
    assert "EDITOR_GROUP_SIZE=${EDITOR_GROUP_SIZE:-${ROLLOUT_N}}" in text
    assert "EDITOR_TRAIN_BATCH_SIZE=${EDITOR_TRAIN_BATCH_SIZE:-$((TRAIN_BATCH_SIZE * ROLLOUT_N))}" in text
    assert 'export MAS_LOOP_EDITOR_GROUP_SIZE=${MAS_LOOP_EDITOR_GROUP_SIZE:-${EDITOR_GROUP_SIZE}}' in text
    assert '"protocol": "mas-loop-source-adversarial-joint-step-v3"' in text
    for field in (
        "rollout_orchestra_checkpoint", "rollout_editor_checkpoint",
        "resulting_orchestra_checkpoint", "resulting_editor_checkpoint",
        "task_group_size", "source_executions", "updated_executions",
        "orchestra_reward_version", "refiner_reward_version", "structured_output_protocol", "patch_compiler_version",
        "canonical_group_count", "canonical_pair_count", "editor_replay_rows",
        "gpu_namespace", "run_dir", "data_seed",
    ):
        assert f'"{field}"' in text


def test_generic_launcher_auto_namespaces_both_eight_gpu_slots() -> None:
    text = (REPO_ROOT / "scripts/run_mas_loop_adversarial_grpo.sh").read_text()
    assert '"0,1,2,3,4,5,6,7") MAS_LOOP_PORT_NAMESPACE_DEFAULT=80; MAS_LOOP_RAY_PORT_DEFAULT=8025' in text
    assert '"8,9,10,11,12,13,14,15") MAS_LOOP_PORT_NAMESPACE_DEFAULT=81; MAS_LOOP_RAY_PORT_DEFAULT=8125' in text
    assert 'ORCHESTRA_PORT=${ORCHESTRA_PORT:-${MAS_LOOP_PORT_NAMESPACE}10}' in text
    assert 'EDITOR_PORT=${EDITOR_PORT:-${MAS_LOOP_PORT_NAMESPACE}11}' in text
    assert 'SUBAGENT_PORT=${SUBAGENT_PORT:-${MAS_LOOP_PORT_NAMESPACE}12}' in text
    assert 'JUDGE_PORT=${JUDGE_PORT:-${MAS_LOOP_PORT_NAMESPACE}13}' in text
    assert 'RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/mas_loop_ray_${MAS_LOOP_PORT_NAMESPACE}}' in text


def test_bcp_wrapper_auto_namespaces_service_and_ray_ports() -> None:
    text = (REPO_ROOT / "scripts/run_bcp_mas_loop_full.sh").read_text()
    assert '"8,9,10,11,12,13,14,15") BCP_PORT_NAMESPACE_DEFAULT=86' in text
    assert '*) BCP_PORT_NAMESPACE_DEFAULT=85' in text
    assert 'BCP_ORCHESTRA_PORT=${BCP_ORCHESTRA_PORT:-${BCP_PORT_NAMESPACE}10}' in text
    assert 'BCP_EDITOR_PORT=${BCP_EDITOR_PORT:-${BCP_PORT_NAMESPACE}11}' in text
    assert 'BCP_SUBAGENT_PORT=${BCP_SUBAGENT_PORT:-${BCP_PORT_NAMESPACE}12}' in text
    assert 'BCP_JUDGE_PORT=${BCP_JUDGE_PORT:-${BCP_PORT_NAMESPACE}13}' in text
    assert 'BCP_RAY_DASHBOARD_PORT=${BCP_RAY_DASHBOARD_PORT:-${BCP_PORT_NAMESPACE}25}' in text


def test_external_vllm_services_persist_across_joint_steps() -> None:
    text = (REPO_ROOT / "scripts/run_mas_loop_adversarial_grpo.sh").read_text()
    assert 'MAS_LOOP_PERSIST_EXTERNAL_VLLM=${MAS_LOOP_PERSIST_EXTERNAL_VLLM:-${MAS_LOOP_VLLM_SLEEP}}' in text
    assert 'ensure_vllm_sleeping "${SUBAGENT_PORT}" "${SUBAGENT_MODEL}" subagent' in text
    assert 'ensure_vllm_sleeping "${JUDGE_PORT}" "${JUDGE_MODEL}" judge' in text
    assert 'release_external_vllms_after_phase' in text
    assert 'reuse sleeping ${name}' in text
    assert 'stop_vllm_role editor-peer' in text

    orchestra = text[text.index("run_orchestra_phase(){") : text.index("run_editor_phase(){")]
    editor = text[text.index("run_editor_phase(){") : text.index("assert_joint_checkpoint(){")]
    assert "release_external_vllms_after_phase" in orchestra
    assert "release_external_vllms_after_phase" in editor
    assert "cleanup; cleanup_ray_temp" not in editor
    assert "Subagent/Judge engines remain sleeping" in editor


def test_joint_step_scheduler_filters_owned_overrides_and_can_stop_after_one_new_step() -> None:
    text = (REPO_ROOT / "scripts/run_mas_loop_adversarial_grpo.sh").read_text()
    joint = text[text.index("run_joint_training_round(){") :]
    assert 'local -a phase_overrides=()' in joint
    assert 'trainer.total_training_steps=*|trainer.total_epochs=*|+trainer.max_steps_this_run=*' in joint
    # Empty Bash arrays are considered unset under `set -u`; the + guard
    # expands to zero argv entries when no user overrides remain after filtering.
    assert 'run_orchestra_phase "$round" ${phase_overrides[@]+"${phase_overrides[@]}"}' in joint
    assert 'run_editor_phase "$round" ${phase_overrides[@]+"${phase_overrides[@]}"}' in joint
    assert 'completed_this_run=$((completed_this_run + 1))' in joint
    assert 'MAS_LOOP_MAX_JOINT_STEPS_THIS_RUN > 0' in joint
    assert 'stopping cleanly after joint step ${step}' in joint
