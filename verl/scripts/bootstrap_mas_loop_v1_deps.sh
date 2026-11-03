#!/usr/bin/env bash
set -euo pipefail

MAS_ROOT=${MAS_ROOT:-/path/to/MAS_Loop}
VERL_ROOT=${VERL_ROOT:-${MAS_ROOT}/verl}
DEPS_DIR=${MAS_LOOP_DEPS_DIR:-${VERL_ROOT}/.mas_loop_deps}
RUNTIME_DEPS_DIR=${MAS_LOOP_RUNTIME_DEPS_DIR:-${VERL_ROOT}/.mas_loop_runtime_deps}
mkdir -p "${DEPS_DIR}"

if PYTHONPATH="${DEPS_DIR}:${VERL_ROOT}:${MAS_ROOT}:${PYTHONPATH:-}" python3 - <<'PY'
from packaging.version import Version
import tensordict
import transfer_queue
import pyvers
assert Version(tensordict.__version__) >= Version("0.10.0")
PY
then
  echo "[MAS-Loop deps] source packages ready: ${DEPS_DIR}"
else
  echo "[MAS-Loop deps] installing project-local Verl V1/TransferQueue dependencies into ${DEPS_DIR}"
  python3 -m pip install --no-deps --target "${DEPS_DIR}" \
    'TransferQueue==0.1.8' \
    'tensordict==0.10.0' \
    'pyvers==0.1.0'
fi

# The install target may also contain packages left by older bootstrap attempts.
# Expose only the three V1 compatibility packages at runtime so an incomplete
# local transformers/tokenizers tree cannot shadow the environment validated by
# vLLM. Keep the allowlist explicit and fail closed if the runtime directory is
# ever populated with anything else.
mkdir -p "${RUNTIME_DEPS_DIR}"
runtime_entries=(
  tensordict
  tensordict-0.10.0.dist-info
  transfer_queue
  transferqueue-0.1.8.dist-info
  pyvers
  pyvers-0.1.0.dist-info
)
for entry in "${runtime_entries[@]}"; do
  [[ -e "${DEPS_DIR}/${entry}" ]] || {
    echo "[MAS-Loop deps] missing required local package entry: ${DEPS_DIR}/${entry}" >&2
    exit 2
  }
  ln -sfn "${DEPS_DIR}/${entry}" "${RUNTIME_DEPS_DIR}/${entry}"
done
for path in "${RUNTIME_DEPS_DIR}"/*; do
  [[ -e "${path}" || -L "${path}" ]] || continue
  name=${path##*/}
  allowed=0
  for entry in "${runtime_entries[@]}"; do
    if [[ "${name}" == "${entry}" ]]; then allowed=1; break; fi
  done
  if (( allowed == 0 )); then
    echo "[MAS-Loop deps] refusing unexpected runtime dependency entry: ${path}" >&2
    exit 2
  fi
done

MAS_LOOP_SOURCE_DEPS_DIR="${DEPS_DIR}" PYTHONPATH="${RUNTIME_DEPS_DIR}:${VERL_ROOT}:${MAS_ROOT}:${PYTHONPATH:-}" python3 - <<'PY'
from pathlib import Path
from packaging.version import Version
from transformers import AutoTokenizer
import transformers.models.encoder_decoder  # noqa: F401
import transformers, tensordict, transfer_queue, pyvers

assert Version(tensordict.__version__) >= Version("0.10.0")
source_deps = Path(__import__("os").environ["MAS_LOOP_SOURCE_DEPS_DIR"]).resolve()
transformers_path = Path(transformers.__file__).resolve()
# Security: fail closed if the broad project-local install target shadows the
# environment's complete transformers package again.
assert source_deps not in transformers_path.parents, (source_deps, transformers_path)
assert AutoTokenizer is not None
print("[MAS-Loop deps] transformers", transformers.__version__, transformers.__file__)
print("[MAS-Loop deps] tensordict", tensordict.__version__, tensordict.__file__)
print("[MAS-Loop deps] transfer_queue", transfer_queue.__file__)
print("[MAS-Loop deps] pyvers", pyvers.__file__)
PY
echo "[MAS-Loop deps] isolated runtime ready: ${RUNTIME_DEPS_DIR}"
