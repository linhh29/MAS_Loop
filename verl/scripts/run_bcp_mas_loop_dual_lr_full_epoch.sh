#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
ulimit -n 1048576 2>/dev/null || true

# Launch two isolated 8-PPU BCP MAS-Loop full-epoch runs from the same Qwen3-8B
# initialization and the same data seed. The controlled variable is LR.
ROOT=${BCP_DUAL_ROOT:-/path/to/MAS_Loop}
TRAIN=${ROOT}/verl/scripts/run_bcp_mas_loop_full.sh
STAMP=${BCP_DUAL_STAMP:-$(date +%Y%m%d_%H%M%S)}
SEED=${BCP_DUAL_DATA_SEED:-0}
LR_A=${BCP_DUAL_LR_A:-5e-7}
LR_B=${BCP_DUAL_LR_B:-2e-6}
CKPT_PARENT=${BCP_DUAL_CKPT_PARENT:-/path/to/checkpoints/bcp}
RUN_A=${CKPT_PARENT}/${STAMP}_dual_lr5e-7_gpu0_7
RUN_B=${CKPT_PARENT}/${STAMP}_dual_lr2e-6_gpu8_15
AUDIT_ROOT=${BCP_DUAL_AUDIT_ROOT:-${ROOT}/datasets/eval_results/bcp/${STAMP}_dual_lr_full_epoch}
LAUNCH_MANIFEST=${AUDIT_ROOT}/launch_manifest.json
POINTER=/tmp/mas_loop_bcp_dual_lr_latest.json

[[ -x "${TRAIN}" || -f "${TRAIN}" ]] || { echo "missing trainer: ${TRAIN}" >&2; exit 2; }
[[ "${RUN_A}" != "${RUN_B}" ]] || { echo "run roots must differ" >&2; exit 2; }
mkdir -p "${RUN_A}" "${RUN_B}" "${AUDIT_ROOT}"

write_manifest() {
  local state=$1 pid_a=${2:-0} pid_b=${3:-0}
  python3 - "${LAUNCH_MANIFEST}" "${POINTER}" "${state}" "${STAMP}" "${SEED}" "${LR_A}" "${LR_B}" "${RUN_A}" "${RUN_B}" "${pid_a}" "${pid_b}" <<'PY_MANIFEST'
import json, os, sys, time
from pathlib import Path
(path, pointer, state, stamp, seed, lr_a, lr_b, run_a, run_b, pid_a, pid_b) = sys.argv[1:]
payload = {
    "schema_version": "mas_loop_bcp_dual_lr_full_epoch/1",
    "state": state,
    "updated_at_unix": time.time(),
    "stamp": stamp,
    "base_model": "/path/to/models/Qwen3-8B",
    "data_seed": int(seed),
    "dataset_rows": 662,
    "padded_rows": 704,
    "train_batch_size": 64,
    "rollout_n": 32,
    "expected_joint_steps": 11,
    "structured_output_protocol": "xml",
    "pairing_mode_for_eval": "same-step",
    "runs": {
        "A": {"gpus": "0,1,2,3,4,5,6,7", "orchestra_lr": lr_a, "editor_lr": lr_a, "run_root": run_a, "pid": int(pid_a)},
        "B": {"gpus": "8,9,10,11,12,13,14,15", "orchestra_lr": lr_b, "editor_lr": lr_b, "run_root": run_b, "pid": int(pid_b)},
    },
}
for target in (Path(path), Path(pointer)):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, target)
PY_MANIFEST
}

common_env=(
  BCP_MAS_ROOT="${ROOT}"
  BCP_DATA_SEED="${SEED}"
  BCP_MAX_TRAIN_ROWS=0
  BCP_DATA_VAL_RATIO=0
  BCP_DATA_PAD_TO_MULTIPLE=64
  BCP_TRAIN_BATCH_SIZE=64
  BCP_ROLLOUT_N=32
  BCP_EDITOR_CANDIDATES_PER_SOURCE=1
  BCP_EDITOR_GROUP_SIZE=32
  BCP_EDITOR_TRAIN_BATCH_SIZE=2048
  BCP_TOTAL_EPOCHS=1
  BCP_ADVERSARIAL_ROUNDS=1
  BCP_JOINT_STEP_TRAINING=1
  BCP_NUM_CHECKPOINTS=11
  BCP_STRUCTURED_OUTPUT_PROTOCOL=xml
  BCP_ACTOR_GUIDED_JSON=0
  BCP_EDITOR_GUIDED_JSON=0
  BCP_ROLLOUT_TP=8
  BCP_MAX_CONCURRENCY=512
  BCP_VLLM_MAX_ACTIVE=512
  BCP_ORCHESTRA_VLLM_MAX_ACTIVE=384
  BCP_EDITOR_VLLM_MAX_ACTIVE=256
  BCP_SUB_AGENT_VLLM_MAX_ACTIVE=256
  BCP_JUDGE_VLLM_MAX_ACTIVE=128
  BCP_DISPATCH_WORKERS_PER_PROCESS=64
  BCP_AGENT_LOOP_MAX_OUTSTANDING_PER_WORKER=64
  # Verl reserves async actor control-method capacity; values below 1000 are rejected.
  BCP_RAY_SERVER_MAX_CONCURRENCY=1024
  BCP_RAY_GENERATE_MAX_OUTSTANDING=512
  # PPU exposes torch.cuda but cannot rebuild CUDA IPC handles in vLLM children.
  VERL_VLLM_USE_SHM_WEIGHT_TRANSFER=1
)

launch_one() {
  local label=$1 gpus=$2 lr=$3 run=$4 namespace=$5
  env "${common_env[@]}" \
    BCP_TIMESTAMP="${STAMP}_${label}" \
    BCP_ALL_GPUS="${gpus}" \
    BCP_ORCH_LR="${lr}" BCP_EDITOR_LR="${lr}" \
    BCP_CKPT_ROOT="${run}" BCP_RUN_DIR="${run}" \
    BCP_DATA_DIR="${run}/data" BCP_LOG_DIR="${run}/logs" \
    BCP_PAIR_DIR="${run}/logs/pairs" \
    BCP_PORT_NAMESPACE="${namespace}" \
    BCP_RAY_TEMP_DIR="/tmp/mas_loop_bcp_ray_${STAMP}_${label}" \
    BCP_CONTROL_DIR="/tmp/mas_loop_bcp_control_${STAMP}_${label}" \
    bash "${TRAIN}" >"${run}/launcher.log" 2>&1
}

write_manifest launching 0 0
launch_one A 0,1,2,3,4,5,6,7 "${LR_A}" "${RUN_A}" 85 &
pid_a=$!
launch_one B 8,9,10,11,12,13,14,15 "${LR_B}" "${RUN_B}" 86 &
pid_b=$!
write_manifest running "${pid_a}" "${pid_b}"
echo "[dual-lr] A pid=${pid_a} root=${RUN_A} lr=${LR_A}"
echo "[dual-lr] B pid=${pid_b} root=${RUN_B} lr=${LR_B}"
echo "[dual-lr] manifest=${LAUNCH_MANIFEST}"

terminate_children() {
  kill -TERM "${pid_a}" "${pid_b}" 2>/dev/null || true
  wait "${pid_a}" 2>/dev/null || true
  wait "${pid_b}" 2>/dev/null || true
}
trap terminate_children INT TERM

rc_a=0; rc_b=0
wait "${pid_a}" || rc_a=$?
wait "${pid_b}" || rc_b=$?
if (( rc_a == 0 && rc_b == 0 )); then
  write_manifest complete "${pid_a}" "${pid_b}"
  exit 0
fi
write_manifest failed "${pid_a}" "${pid_b}"
echo "[dual-lr] failure: A=${rc_a} B=${rc_b}" >&2
exit 1
