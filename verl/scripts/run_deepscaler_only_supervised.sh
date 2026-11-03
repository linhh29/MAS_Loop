#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ulimit -n 1048576 2>/dev/null || true

# DeepScaleR-only supervisor. It intentionally defines only DS_* experiment
# parameters, so BCP's launcher and hyper-parameter namespace remain untouched.
DS_MAS_ROOT=${DS_MAS_ROOT:-/path/to/MAS_Loop}
DS_VERL_ROOT=${DS_VERL_ROOT:-${DS_MAS_ROOT}/verl}
DS_ALL_GPUS=${DS_ALL_GPUS:-8,9,10,11,12,13,14,15}
DS_TIMESTAMP=${DS_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
DS_CKPT_ROOT=${DS_CKPT_ROOT:-/path/to/checkpoints/deepscaler/${DS_TIMESTAMP}}
DS_RUN_ROOT=${DS_RUN_ROOT:-${DS_CKPT_ROOT}}
DS_SESSION=${DS_SESSION:-masloop_deepscaler_${DS_TIMESTAMP}}
DS_MONITOR_SESSION=${DS_MONITOR_SESSION:-${DS_SESSION}}
DS_MONITOR_INTERVAL_SEC=${DS_MONITOR_INTERVAL_SEC:-1800}
DS_METRICS_INTERVAL_SEC=${DS_METRICS_INTERVAL_SEC:-60}
DS_GPU_WAIT_INTERVAL_SEC=${DS_GPU_WAIT_INTERVAL_SEC:-15}
DS_GPU_STABLE_CHECKS=${DS_GPU_STABLE_CHECKS:-2}
DS_GPU_WAIT_TIMEOUT_SEC=${DS_GPU_WAIT_TIMEOUT_SEC:-0}
DS_EXPECTED_SAMPLES=${DS_EXPECTED_SAMPLES:-40315}
DS_ROLLOUT_N=${DS_ROLLOUT_N:-8}
DS_EDITOR_GROUP_SIZE=${DS_EDITOR_GROUP_SIZE:-8}
# Save at most ten evenly spaced checkpoints including final; a run with ten
# or fewer optimizer steps saves every step.
DS_NUM_CHECKPOINTS=${DS_NUM_CHECKPOINTS:-10}
DS_SAVE_FREQ=${DS_SAVE_FREQ:-250}
DS_RUN_ORCHESTRA=${DS_RUN_ORCHESTRA:-1}
DS_RUN_EDITOR=${DS_RUN_EDITOR:-1}
DS_MAX_TRAIN_ROWS=${DS_MAX_TRAIN_ROWS:-0}
DS_RAY_DASHBOARD_PORT=${DS_RAY_DASHBOARD_PORT:-8425}
DS_RAY_TEMP_DIR=${DS_RAY_TEMP_DIR:-/tmp/mas_loop_deepscaler_ray_${DS_TIMESTAMP}}
DS_CONTROL_DIR=${DS_CONTROL_DIR:-/tmp/mas_loop_deepscaler_control_${DS_TIMESTAMP}}
DS_CONTROL_SOCKET=${DS_CONTROL_SOCKET:-${DS_CONTROL_DIR}/coordinator.sock}
DS_SLEEP_LOCK=${DS_SLEEP_LOCK:-${DS_CONTROL_DIR}/fallback.lock}
DS_SLEEP_STATE=${DS_SLEEP_STATE:-${DS_CONTROL_DIR}/state.json}
DS_SLEEP_EVENTS=${DS_SLEEP_EVENTS:-${DS_CONTROL_DIR}/events.jsonl}
DS_SLEEP_EVENTS_DURABLE=${DS_SLEEP_EVENTS_DURABLE:-${DS_RUN_ROOT}/vllm_sleep/events.jsonl}

[[ "${DS_MAS_ROOT}" == "/path/to/MAS_Loop" ]] || { echo "[DS-SUP] wrong root: ${DS_MAS_ROOT}" >&2; exit 2; }
[[ "${DS_ALL_GPUS}" == "8,9,10,11,12,13,14,15" ]] || { echo "[DS-SUP] only GPUs 8-15 allowed" >&2; exit 2; }
[[ -x "${DS_VERL_ROOT}/scripts/run_deepscaler_mas_loop_full.sh" ]] || { echo "[DS-SUP] missing DeepScaleR launcher" >&2; exit 2; }
[[ -f "${DS_MAS_ROOT}/datasets/DeepScaleR/deepscaler_train.json" ]] || exit 2
[[ -f /path/to/models/Qwen3-8B/config.json && -f /path/to/models/Qwen3-30B-A3B/config.json ]] || { echo "missing local Qwen models under /path/to/models" >&2; exit 2; }
mkdir -p "${DS_RUN_ROOT}/logs" "${DS_RUN_ROOT}/data" "${DS_RUN_ROOT}/monitor" "${DS_CONTROL_DIR}" "$(dirname "${DS_SLEEP_EVENTS_DURABLE}")"

publish_monitor_env() {
  local tmp="${DS_RUN_ROOT}/current_mas_loop_job.env.tmp.$$"
  cat > "${tmp}" <<MON
RUN_DIR=${DS_RUN_ROOT}
CHECKPOINT_ROOT=${DS_CKPT_ROOT}
RUN_NAME=$(basename "${DS_RUN_ROOT}")
GPU_SET=${DS_ALL_GPUS}
LAUNCH_SCRIPT=${DS_VERL_ROOT}/scripts/run_deepscaler_only_supervised.sh
START_TIME=$(date +%s)
SESSION=${DS_MONITOR_SESSION}
DATASET=deepscaler
EXPECTED_SAMPLES=${DS_EXPECTED_SAMPLES}
PAIR_GROUP_SIZE=${DS_EDITOR_GROUP_SIZE}
VLLM_SLEEP_STATE=${DS_SLEEP_STATE}
VLLM_SLEEP_EVENTS=${DS_SLEEP_EVENTS}
MON
  mv -f "${tmp}" "${DS_RUN_ROOT}/current_mas_loop_job.env"
}

monitor_loop() {
  local sleep_pid=""
  trap '
    if [[ -n "${sleep_pid}" ]]; then
      kill "${sleep_pid}" 2>/dev/null || true
      wait "${sleep_pid}" 2>/dev/null || true
    fi
    exit 0
  ' TERM INT

  while true; do
    local now
    now=$(date '+%F %T %Z')
    {
      echo "========== DeepScaleR supervisor check ${now} =========="
      gpustat --no-color -p 2>/dev/null | awk 'NR==1 || /^\[(8|9|10|11|12|13|14|15)\]/' || true
      ps -eo pid,ppid,etime,stat,cmd | grep -E '/path/to/MAS_Loop|mas_loop_rl|verl.trainer.main_ppo|vllm.entrypoints.openai.api_server' | grep -v grep || true
    } >> "${DS_RUN_ROOT}/monitor/status_30min.log" 2>&1
    gpustat --json >> "${DS_RUN_ROOT}/monitor/gpustat.jsonl" 2>/dev/null || true
    sleep "${DS_MONITOR_INTERVAL_SEC}" &
    sleep_pid=$!
    wait "${sleep_pid}" || return 0
    sleep_pid=""
  done
}

publish_monitor_env
monitor_loop &
DS_MONITOR_PID=$!
python3 "${DS_VERL_ROOT}/mas_loop_rl/metrics_monitor.py" \
  --log-file "${DS_RUN_ROOT}/logs/full_train.log" \
  --output-dir "${DS_RUN_ROOT}/metrics" \
  --interval "${DS_METRICS_INTERVAL_SEC}" \
  >> "${DS_RUN_ROOT}/monitor/metrics_monitor.log" 2>&1 &
DS_METRICS_PID=$!
cleanup() {
  trap - EXIT INT TERM
  kill "${DS_MONITOR_PID}" "${DS_METRICS_PID}" 2>/dev/null || true
  wait "${DS_MONITOR_PID}" 2>/dev/null || true
  wait "${DS_METRICS_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cat > "${DS_RUN_ROOT}/launch_config.env" <<CFG
DS_MAS_ROOT=${DS_MAS_ROOT}
DS_VERL_ROOT=${DS_VERL_ROOT}
DS_ALL_GPUS=${DS_ALL_GPUS}
DS_RUN_DIR=${DS_RUN_ROOT}
DS_CKPT_ROOT=${DS_CKPT_ROOT}
DS_SESSION=${DS_SESSION}
DS_ROLLOUT_N=${DS_ROLLOUT_N}
DS_EDITOR_GROUP_SIZE=${DS_EDITOR_GROUP_SIZE}
DS_NUM_CHECKPOINTS=${DS_NUM_CHECKPOINTS}
DS_SAVE_FREQ=${DS_SAVE_FREQ}
DS_RUN_ORCHESTRA=${DS_RUN_ORCHESTRA}
DS_RUN_EDITOR=${DS_RUN_EDITOR}
DS_MAX_TRAIN_ROWS=${DS_MAX_TRAIN_ROWS}
DS_METRICS_INTERVAL_SEC=${DS_METRICS_INTERVAL_SEC}
DS_GPU_WAIT_INTERVAL_SEC=${DS_GPU_WAIT_INTERVAL_SEC}
DS_GPU_STABLE_CHECKS=${DS_GPU_STABLE_CHECKS}
DS_GPU_WAIT_TIMEOUT_SEC=${DS_GPU_WAIT_TIMEOUT_SEC}
DS_RAY_DASHBOARD_PORT=${DS_RAY_DASHBOARD_PORT}
DS_RAY_TEMP_DIR=${DS_RAY_TEMP_DIR}
CFG

# Re-check ownership after BCP cleanup. A foreign sleeping vLLM may use little
# memory while retaining a process capable of waking later, so the guard checks
# both target-GPU compute processes and foreign project roots before launch.
echo "[DS-SUP] waiting for isolated physical GPUs ${DS_ALL_GPUS} at $(date '+%F %T %Z')" | tee "${DS_RUN_ROOT}/monitor/gpu_isolation.log"
PYTHONPATH="${DS_VERL_ROOT}:${DS_MAS_ROOT}:${PYTHONPATH:-}" \
python3 -m mas_loop_rl.wait_for_gpu_isolation \
  --gpus "${DS_ALL_GPUS}" --interval "${DS_GPU_WAIT_INTERVAL_SEC}" \
  --stable-checks "${DS_GPU_STABLE_CHECKS}" --timeout "${DS_GPU_WAIT_TIMEOUT_SEC}" \
  | tee -a "${DS_RUN_ROOT}/monitor/gpu_isolation.log"

echo "[DS-SUP] starting isolated DeepScaleR full train at $(date '+%F %T %Z')"
echo "[DS-SUP] session=${DS_SESSION} run_root=${DS_RUN_ROOT} gpus=${DS_ALL_GPUS}"
set +e
env \
  DS_MAS_ROOT="${DS_MAS_ROOT}" DS_VERL_ROOT="${DS_VERL_ROOT}" DS_ALL_GPUS="${DS_ALL_GPUS}" \
  DS_TIMESTAMP="${DS_TIMESTAMP}" DS_CKPT_ROOT="${DS_CKPT_ROOT}" DS_RUN_DIR="${DS_RUN_ROOT}" DS_LOG_DIR="${DS_RUN_ROOT}/logs" DS_DATA_DIR="${DS_RUN_ROOT}/data" \
  DS_ROLLOUT_N="${DS_ROLLOUT_N}" DS_EDITOR_GROUP_SIZE="${DS_EDITOR_GROUP_SIZE}" DS_NUM_CHECKPOINTS="${DS_NUM_CHECKPOINTS}" DS_SAVE_FREQ="${DS_SAVE_FREQ}" \
  DS_RUN_ORCHESTRA="${DS_RUN_ORCHESTRA}" DS_RUN_EDITOR="${DS_RUN_EDITOR}" DS_MAX_TRAIN_ROWS="${DS_MAX_TRAIN_ROWS}"     DS_CONTROL_DIR="${DS_CONTROL_DIR}" DS_CONTROL_SOCKET="${DS_CONTROL_SOCKET}" DS_SLEEP_LOCK="${DS_SLEEP_LOCK}" DS_SLEEP_STATE="${DS_SLEEP_STATE}" DS_SLEEP_EVENTS="${DS_SLEEP_EVENTS}" DS_SLEEP_EVENTS_DURABLE="${DS_SLEEP_EVENTS_DURABLE}" \
  bash "${DS_VERL_ROOT}/scripts/run_deepscaler_mas_loop_full.sh" "$@" 2>&1 | tee "${DS_RUN_ROOT}/logs/full_train.log"
rc=${PIPESTATUS[0]}
set -e
printf 'TRAIN_EXIT_CODE=%s\nEXIT_CODE=%s\nFINISHED_AT=%s\n' "${rc}" "${rc}" "$(date '+%F %T %Z')" > "${DS_RUN_ROOT}/exit.status"
echo "[DS-SUP] DeepScaleR exited rc=${rc} at $(date '+%F %T %Z')"
exit "${rc}"
