#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ulimit -n 1048576 2>/dev/null || true

# Strictly serial, dataset-isolated end-to-end supervisor:
# BCP full train -> DeepScaleR full train -> baseline + all checkpoint pairs.
# BCP_* and DS_* namespaces remain separate and only physical GPUs 8-15 are used.
MAS_ROOT=${MAS_ROOT:-/path/to/MAS_Loop}
VERL_ROOT=${VERL_ROOT:-${MAS_ROOT}/verl}
ALL_GPUS=${ALL_GPUS:-8,9,10,11,12,13,14,15}
TS=${TS:-$(date +%Y%m%d_%H%M%S)}
SERIAL_SESSION=${SERIAL_SESSION:-masloop_full_bcp_ds_eval_${TS}}
CKPT_BASE=${CKPT_BASE:-/path/to/checkpoints}
RUN_ROOT=${RUN_ROOT:-${CKPT_BASE}/serial/${TS}}
BCP_CKPT_ROOT=${BCP_CKPT_ROOT:-${CKPT_BASE}/bcp/${TS}}
DS_CKPT_ROOT=${DS_CKPT_ROOT:-${CKPT_BASE}/deepscaler/${TS}}
RUN_EVAL_MATRIX=${RUN_EVAL_MATRIX:-1}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT:-${RUN_ROOT}/eval_matrix}
EVAL_BASELINE_MODEL=${EVAL_BASELINE_MODEL:-/path/to/models/Qwen3-8B}
EVAL_SUBAGENT_MODEL=${EVAL_SUBAGENT_MODEL:-/path/to/models/Qwen3-30B-A3B}
EVAL_CONTINUE_ON_ERROR=${EVAL_CONTINUE_ON_ERROR:-1}
BCP_NUM_CHECKPOINTS=${BCP_NUM_CHECKPOINTS:-10}
DS_NUM_CHECKPOINTS=${DS_NUM_CHECKPOINTS:-10}
mkdir -p "${RUN_ROOT}"

[[ "${MAS_ROOT}" == "/path/to/MAS_Loop" ]] || { echo "[SERIAL] wrong root" >&2; exit 2; }
[[ "${ALL_GPUS}" == "8,9,10,11,12,13,14,15" ]] || { echo "[SERIAL] only physical GPUs 8-15 are allowed" >&2; exit 2; }
[[ -x "${VERL_ROOT}/scripts/run_bcp_only_supervised.sh" ]] || exit 2
[[ -x "${VERL_ROOT}/scripts/run_deepscaler_only_supervised.sh" ]] || exit 2
[[ -x "${MAS_ROOT}/datasets/run_all_mas_loop_checkpoint_matrix_eval.sh" ]] || exit 2

cat > "${RUN_ROOT}/serial_config.env" <<CFG
MAS_ROOT=${MAS_ROOT}
VERL_ROOT=${VERL_ROOT}
ALL_GPUS=${ALL_GPUS}
TS=${TS}
RUN_ROOT=${RUN_ROOT}
BCP_RUN_ROOT=${BCP_CKPT_ROOT}
BCP_CKPT_ROOT=${BCP_CKPT_ROOT}
DS_RUN_ROOT=${DS_CKPT_ROOT}
DS_CKPT_ROOT=${DS_CKPT_ROOT}
RUN_EVAL_MATRIX=${RUN_EVAL_MATRIX}
EVAL_OUTPUT_ROOT=${EVAL_OUTPUT_ROOT}
EVAL_BASELINE_MODEL=${EVAL_BASELINE_MODEL}
EVAL_SUBAGENT_MODEL=${EVAL_SUBAGENT_MODEL}
EVAL_CONTINUE_ON_ERROR=${EVAL_CONTINUE_ON_ERROR}
BCP_NUM_CHECKPOINTS=${BCP_NUM_CHECKPOINTS}
DS_NUM_CHECKPOINTS=${DS_NUM_CHECKPOINTS}
CFG

set +e
env BCP_MAS_ROOT="${MAS_ROOT}" BCP_VERL_ROOT="${VERL_ROOT}" BCP_ALL_GPUS="${ALL_GPUS}" \
  BCP_TIMESTAMP="${TS}" BCP_CKPT_ROOT="${BCP_CKPT_ROOT}" BCP_RUN_ROOT="${BCP_CKPT_ROOT}" BCP_RUN_DIR="${BCP_CKPT_ROOT}" \
  BCP_SESSION="masloop_bcp_full_${TS}" BCP_MONITOR_SESSION="${SERIAL_SESSION}" BCP_NUM_CHECKPOINTS="${BCP_NUM_CHECKPOINTS}" \
  bash "${VERL_ROOT}/scripts/run_bcp_only_supervised.sh" "$@" 2>&1 | tee "${RUN_ROOT}/bcp_supervisor.log"
bcp_rc=${PIPESTATUS[0]}
set -e
printf 'BCP_EXIT_CODE=%s\nBCP_FINISHED_AT=%s\n' "${bcp_rc}" "$(date '+%F %T %Z')" > "${RUN_ROOT}/serial.status"
if (( bcp_rc != 0 )); then
  printf 'DEEPSCALER_EXIT_CODE=NOT_STARTED\nEVAL_MATRIX_EXIT_CODE=NOT_STARTED\nEXIT_CODE=%s\n' "${bcp_rc}" >> "${RUN_ROOT}/serial.status"
  echo "[SERIAL] BCP failed rc=${bcp_rc}; DeepScaleR and evaluation will not start." >&2
  exit "${bcp_rc}"
fi

# DeepScaleR starts only after BCP trainer cleanup has completed. Its supervisor checks GPU
# isolation again before creating any model process.
set +e
env DS_MAS_ROOT="${MAS_ROOT}" DS_VERL_ROOT="${VERL_ROOT}" DS_ALL_GPUS="${ALL_GPUS}" \
  DS_TIMESTAMP="${TS}" DS_CKPT_ROOT="${DS_CKPT_ROOT}" DS_RUN_ROOT="${DS_CKPT_ROOT}" DS_RUN_DIR="${DS_CKPT_ROOT}" \
  DS_SESSION="masloop_deepscaler_full_${TS}" DS_MONITOR_SESSION="${SERIAL_SESSION}" DS_NUM_CHECKPOINTS="${DS_NUM_CHECKPOINTS}" \
  bash "${VERL_ROOT}/scripts/run_deepscaler_only_supervised.sh" "$@" 2>&1 | tee "${RUN_ROOT}/deepscaler_supervisor.log"
ds_rc=${PIPESTATUS[0]}
set -e
printf 'DEEPSCALER_EXIT_CODE=%s\nDEEPSCALER_FINISHED_AT=%s\n' "${ds_rc}" "$(date '+%F %T %Z')" >> "${RUN_ROOT}/serial.status"
if (( ds_rc != 0 )); then
  printf 'EVAL_MATRIX_EXIT_CODE=NOT_STARTED\nEXIT_CODE=%s\n' "${ds_rc}" >> "${RUN_ROOT}/serial.status"
  echo "[SERIAL] DeepScaleR failed rc=${ds_rc}; evaluation matrix will not start." >&2
  exit "${ds_rc}"
fi

# Evaluate the configured local base model as both roles,
# followed by every Orchestra checkpoint x every Editor checkpoint. The matrix
# runner hashes each dataset/round/step/source tuple into a unique directory.
if [[ "${RUN_EVAL_MATRIX}" == "1" ]]; then
  if ! find "${BCP_CKPT_ROOT}" -type f -path '*/actor/huggingface/config.json' -print -quit 2>/dev/null | grep -q . ||
     ! find "${DS_CKPT_ROOT}" -type f -path '*/actor/huggingface/config.json' -print -quit 2>/dev/null | grep -q .; then
    echo "[SERIAL] missing local checkpoint(s): bcp=${BCP_CKPT_ROOT} ds=${DS_CKPT_ROOT}" >&2
    printf 'EVAL_MATRIX_EXIT_CODE=3\nEXIT_CODE=3\n' >> "${RUN_ROOT}/serial.status"
    exit 3
  fi
  echo "[SERIAL] starting baseline + all checkpoint-pair evaluations at $(date '+%F %T %Z')"
  set +e
  env BCP_MATRIX_CHECKPOINT_ROOTS="${BCP_CKPT_ROOT}" \
      DEEPSCALER_MATRIX_CHECKPOINT_ROOTS="${DS_CKPT_ROOT}" \
      MAS_LOOP_MATRIX_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT}" \
      MAS_LOOP_MATRIX_BASELINE_MODEL="${EVAL_BASELINE_MODEL}" \
      MAS_LOOP_MATRIX_SUBAGENT_MODEL="${EVAL_SUBAGENT_MODEL}" \
      MAS_LOOP_MATRIX_CONTINUE_ON_ERROR="${EVAL_CONTINUE_ON_ERROR}" \
      bash "${MAS_ROOT}/datasets/run_all_mas_loop_checkpoint_matrix_eval.sh" \
      2>&1 | tee "${RUN_ROOT}/eval_matrix.log"
  eval_rc=${PIPESTATUS[0]}
  set -e
else
  eval_rc=0
  echo "[SERIAL] RUN_EVAL_MATRIX=0; post-training evaluation matrix skipped."
fi
printf 'EVAL_MATRIX_EXIT_CODE=%s\nEVAL_FINISHED_AT=%s\nEXIT_CODE=%s\n' "${eval_rc}" "$(date '+%F %T %Z')" "${eval_rc}" >> "${RUN_ROOT}/serial.status"
exit "${eval_rc}"
