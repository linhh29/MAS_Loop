#!/usr/bin/env bash
set -euo pipefail

ROOT=$(readlink -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)")
LAUNCHER="${ROOT}/verl/scripts/run_hotpotqa_editor_ppo_token_cap_ab_1step.sh"
LOG_DIR="${ROOT}/run_logs"
[[ -x "${LAUNCHER}" ]] || { echo "Missing executable launcher: ${LAUNCHER}" >&2; exit 2; }
command -v tmux >/dev/null 2>&1 || { echo "tmux is required" >&2; exit 2; }

# Security: Only accept the launcher's numeric timestamp format. The value becomes
# part of tmux session/file names and must never inject shell syntax.
ts=${HP_AB_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
[[ "${ts}" =~ ^[0-9]{8}_[0-9]{6}$ ]] || {
  echo "HP_AB_TIMESTAMP must match YYYYMMDD_HHMMSS, got: ${ts}" >&2
  exit 2
}

session_a="editor_ppo32768_${ts}"
session_b="editor_ppo49152_${ts}"
log_a="${LOG_DIR}/${session_a}.log"
log_b="${LOG_DIR}/${session_b}.log"
meta="${LOG_DIR}/editor_ppo_token_cap_ab_${ts}.meta"
mkdir -p "${LOG_DIR}"

for session in "${session_a}" "${session_b}"; do
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Refusing to reuse existing tmux session: ${session}" >&2
    exit 2
  fi
done
for log in "${log_a}" "${log_b}" "${meta}"; do
  [[ ! -e "${log}" ]] || { echo "Refusing to overwrite existing file: ${log}" >&2; exit 2; }
done

printf -v train_a '%q ' env HP_AB_TIMESTAMP="${ts}" bash "${LAUNCHER}" editorppo32768-gpu0-7
printf -v train_b '%q ' env HP_AB_TIMESTAMP="${ts}" bash "${LAUNCHER}" editorppo49152-gpu8-15
printf -v quoted_log_a '%q' "${log_a}"
printf -v quoted_log_b '%q' "${log_b}"
cmd_a="set -o pipefail; ${train_a}2>&1 | tee ${quoted_log_a}"
cmd_b="set -o pipefail; ${train_b}2>&1 | tee ${quoted_log_b}"

# tmux is intentionally the host-side process owner; this script only creates
# detached sessions and records immutable run metadata for later supervision.
tmux new-session -d -s "${session_a}" -c "${ROOT}" bash -lc "${cmd_a}"
tmux set-option -t "${session_a}" remain-on-exit on >/dev/null
if ! tmux new-session -d -s "${session_b}" -c "${ROOT}" bash -lc "${cmd_b}"; then
  tmux kill-session -t "${session_a}" 2>/dev/null || true
  echo "Failed to launch second arm; first arm was stopped." >&2
  exit 1
fi
tmux set-option -t "${session_b}" remain-on-exit on >/dev/null

cat > "${meta}" <<META
timestamp=${ts}
session_a=${session_a}
session_b=${session_b}
log_a=${log_a}
log_b=${log_b}
run_a=${ROOT}/runs/hotpotqa/editor_ppo_token_cap_ab_${ts}/gpu0-7_editor_tp8_c1024_ppo32768
run_b=${ROOT}/runs/hotpotqa/editor_ppo_token_cap_ab_${ts}/gpu8-15_editor_tp8_c1024_ppo49152
META

printf 'Started A: %s\nStarted B: %s\nMetadata: %s\n' "${session_a}" "${session_b}" "${meta}"
printf 'Monitor: tmux list-panes -t %q -F %q\n' "${session_a}" '#{pane_dead} #{pane_dead_status} #{pane_current_command}'
printf 'Monitor: tmux list-panes -t %q -F %q\n' "${session_b}" '#{pane_dead} #{pane_dead_status} #{pane_current_command}'
