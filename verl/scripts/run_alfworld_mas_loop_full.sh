#!/usr/bin/env bash
set -euo pipefail
ROOT=$(readlink -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")
SOURCE_TRAIN_JSON=${ALFWORLD_TASK_JSON:-${ROOT}/datasets/alfworld/data/alfworld_train.jsonl}
[[ -f "${SOURCE_TRAIN_JSON}" ]] || { echo "[ALFWORLD] missing training manifest: ${SOURCE_TRAIN_JSON}" >&2; exit 2; }
RUN_ROOT=${ALFWORLD_RUN_ROOT:-/path/to/checkpoints/alfworld/${ALFWORLD_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}}
TRAIN_JSON=${ALFWORLD_UNIQUE_TASK_JSON:-${RUN_ROOT}/source_snapshot/alfworld_train_4032_unique.jsonl}
PYTHONPATH="${ROOT}/verl:${ROOT}:${PYTHONPATH:-}" python3 -m mas_loop_rl.prepare_alfworld_unique_manifest \
  --input "${SOURCE_TRAIN_JSON}" --output "${TRAIN_JSON}" --expected-rows 4032
read -r TRAIN_ROWS < <(wc -l < "${TRAIN_JSON}")
[[ "${TRAIN_ROWS}" == "4032" ]] || { echo "[ALFWORLD] expected 4032 unique training rows, got ${TRAIN_ROWS}" >&2; exit 2; }
DATA_ROOT=${ALFWORLD_DATA_ROOT:-${ROOT}/../json_2.1.1}
if [[ "${ALFWORLD_CONFIG_ONLY:-0}" != "1" ]]; then
  [[ -d "${DATA_ROOT}" ]] || { echo "[ALFWORLD] missing simulator data root: ${DATA_ROOT}" >&2; exit 2; }
fi
# Run one stable-order epoch over the 4,032 unique tasks. With batch=256 this is
# 15 full batches plus one 192-task final batch; no task or Editor group is
# duplicated to fill either batch.
export HP_DATASET_NAME=alfworld
export HP_TASK_JSON="${TRAIN_JSON}"
export HP_DATA_PAD_TO_MULTIPLE=0
export HP_PREPARE_DATA_SHUFFLE=False
export HP_TRAIN_DATA_SHUFFLE=False
export HP_TRAIN_DROP_LAST=False
export HP_PAD_EDITOR_REPLAY_TO_EXPECTED_ROWS=False
# Invalid structured outputs can make a task-conditioned Editor group incomplete.
# Keep every complete group without duplicating/padding rows; the joint launcher
# derives valid Editor mini/micro batches from the exact replay cardinality.
export ALLOW_INCOMPLETE_EDITOR_REPLAY=True
export HP_PPO_MINI_BATCH_SIZE=${ALFWORLD_PPO_MINI_BATCH_SIZE:-64}
export HP_RETRIEVER=none
export HP_BM25_INDEX=""
export HP_USE_LLM_JUDGE=0
export HP_SHARE_SUBAGENT_JUDGE_VLLM=0
export HP_RUN_ROOT="${RUN_ROOT}"
export HP_ALFWORLD_DATA_ROOT="${DATA_ROOT}"
export HP_ALFWORLD_ACTIONS_PER_NODE=${ALFWORLD_ACTIONS_PER_NODE:-4}
export HP_ALFWORLD_MAX_RECOVERY_ROUNDS=${ALFWORLD_MAX_RECOVERY_ROUNDS:-12}
export HP_CONFIG_ONLY=${ALFWORLD_CONFIG_ONLY:-0}
exec bash "${ROOT}/verl/scripts/run_hotpotqa_mas_loop_full.sh" "$@"
