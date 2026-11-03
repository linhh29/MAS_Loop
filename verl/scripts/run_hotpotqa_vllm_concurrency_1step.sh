#!/usr/bin/env bash
set -euo pipefail
ROOT=/path/to/MAS_Loop
[[ "$(readlink -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")" == "$(readlink -f "${ROOT}")" ]] || { echo "Refusing to run outside ${ROOT}" >&2; exit 2; }
TS=${HP_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
BATCH=${HP_ONE_STEP_BATCH_SIZE:-8}
(( BATCH > 0 && BATCH % 8 == 0 )) || { echo "HotpotQA one-step batch must be a positive multiple of 8" >&2; exit 2; }
RUN_ROOT=${HP_RUN_ROOT:-/path/to/checkpoints/hotpotqa/one_step_c128_${TS}}
exec env \
  HP_TIMESTAMP="${TS}" HP_RUN_ROOT="${RUN_ROOT}" HP_TRAIN_BATCH_SIZE="${BATCH}" HP_EDITOR_TRAIN_BATCH_SIZE="$((BATCH * 8))" \
  HP_PPO_MINI_BATCH_SIZE="${BATCH}" HP_MAX_TRAIN_ROWS="${BATCH}" HP_DATA_PAD_TO_MULTIPLE="${BATCH}" \
  HP_TOTAL_EPOCHS=1 HP_ADVERSARIAL_ROUNDS=1 HP_SAVE_FREQ=1 HP_NUM_CHECKPOINTS=1 \
  HP_ORCHESTRA_PORT=8710 HP_EDITOR_PORT=8711 HP_SUBAGENT_PORT=8712 HP_RAY_DASHBOARD_PORT=8725 \
  bash "${ROOT}/verl/scripts/run_hotpotqa_mas_loop_full.sh" "$@"
