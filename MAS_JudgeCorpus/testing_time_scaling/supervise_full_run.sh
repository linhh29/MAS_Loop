#!/usr/bin/env bash
set -u

WORKSPACE="${WORKSPACE:-$(pwd)}"
RUN_ID="${RUN_ID:-full_vllm_16bank_xx}"
CONFIG="${CONFIG:-testing_time_scaling/config.local-16-c4.yaml}"
POLL_SECONDS="${POLL_SECONDS:-60}"
RUN_DIR="$WORKSPACE/testing_time_scaling/results/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
SUPERVISOR_LOG="$LOG_DIR/api_supervisor.log"

METHODS=(deepseek_v4_flash kimi_k3 qwen3_8_2_4t_a95b)
DATASETS=(hotpotqa bcp)

declare -A EXPECTED=( [hotpotqa]=200 [bcp]=168 )
mkdir -p "$LOG_DIR"

rows() {
  local method="$1" dataset="$2" file
  file="$RUN_DIR/methods/$method/$dataset/predictions.jsonl"
  if [[ -f "$file" ]]; then
    wc -l < "$file"
  else
    printf '0\n'
  fi
}

method_complete() {
  local method="$1" dataset count
  for dataset in "${DATASETS[@]}"; do
    count="$(rows "$method" "$dataset")"
    [[ "$count" -ge "${EXPECTED[$dataset]}" ]] || return 1
  done
}

method_running() {
  local method="$1"
  ps -eo args= | grep -F "testing_time_scaling/run.py select" | grep -F -- "--methods $method" | grep -v grep >/dev/null
}

launch_method() {
  local method="$1" stamp session log
  stamp="$(date +%Y%m%d_%H%M%S)"
  session="mas-judge-api-${method//_/-}-auto"
  log="$LOG_DIR/select.$method.auto.$stamp.log"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$WORKSPACE' && exec python testing_time_scaling/run.py select --config '$CONFIG' --run-id '$RUN_ID' --datasets hotpotqa bcp --methods '$method' >> '$log' 2>&1"
  printf '%s launched method=%s session=%s log=%s\n' "$(date --iso-8601=seconds)" "$method" "$session" "$log" | tee -a "$SUPERVISOR_LOG"
}

printf '%s supervisor_start run_id=%s\n' "$(date --iso-8601=seconds)" "$RUN_ID" | tee -a "$SUPERVISOR_LOG"
while true; do
  all_complete=1
  for method in "${METHODS[@]}"; do
    hp="$(rows "$method" hotpotqa)"
    bcp="$(rows "$method" bcp)"
    printf '%s progress method=%s hotpotqa=%s/200 bcp=%s/168\n' "$(date --iso-8601=seconds)" "$method" "$hp" "$bcp" >> "$SUPERVISOR_LOG"
    if ! method_complete "$method"; then
      all_complete=0
      if ! method_running "$method"; then
        launch_method "$method"
      fi
    fi
  done

  trajectory_hp="$(rows mas_judgerm_trajectory hotpotqa)"
  trajectory_bcp="$(rows mas_judgerm_trajectory bcp)"
  printf '%s progress method=mas_judgerm_trajectory hotpotqa=%s/200 bcp=%s/168\n' \
    "$(date --iso-8601=seconds)" "$trajectory_hp" "$trajectory_bcp" >> "$SUPERVISOR_LOG"

  if [[ "$all_complete" -eq 1 && "$trajectory_hp" -ge 200 && "$trajectory_bcp" -ge 168 ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

printf '%s selectors_complete; rebuilding tables\n' "$(date --iso-8601=seconds)" | tee -a "$SUPERVISOR_LOG"
python testing_time_scaling/run.py select \
  --config "$CONFIG" \
  --run-id "$RUN_ID" \
  --datasets hotpotqa bcp \
  --methods cot_lb >> "$LOG_DIR/finalize_tables.log" 2>&1

printf '%s running strict audit\n' "$(date --iso-8601=seconds)" | tee -a "$SUPERVISOR_LOG"
if python testing_time_scaling/audit_run.py \
  --run-dir "$RUN_DIR" \
  --require-api \
  --output "$RUN_DIR/audit.full.json" >> "$LOG_DIR/audit.full.log" 2>&1; then
  printf '%s audit_passed\n' "$(date --iso-8601=seconds)" | tee -a "$SUPERVISOR_LOG"
else
  code=$?
  printf '%s audit_failed exit_code=%s\n' "$(date --iso-8601=seconds)" "$code" | tee -a "$SUPERVISOR_LOG"
  exit "$code"
fi
