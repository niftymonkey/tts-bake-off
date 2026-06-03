#!/usr/bin/env bash
# setup.sh: recreate everything a fresh clone is missing.
#
# venv-*/ and models/ are gitignored (large; deps conflict so they cannot share an
# interpreter). This rebuilds each engine's uv venv from its committed lockfile in
# requirements/ and fetches the Kokoro model weights. The Chatterbox, XTTS, and Dia
# model weights are NOT fetched here; their libraries lazy-download from Hugging Face
# on first generate and cache under ~/.cache/huggingface.
#
# Usage:
#   ./setup.sh                 rebuild the wired-up engines (kokoro chatterbox xtts dia ui)
#   ./setup.sh --with-orpheus  also rebuild venv-orpheus (heavy; pulls vLLM, no worker yet)
#   ./setup.sh --only a,b      rebuild only these engines (e.g. --only kokoro,ui)
#   ./setup.sh --force         recreate venvs that already exist (default: skip them)
#   ./setup.sh --venvs-only    skip the model download
#   ./setup.sh --models-only   skip venv creation
#   ./setup.sh -h | --help     show this help
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

PY=3.12  # every venv was built on CPython 3.12

# Every engine that has a lockfile. orpheus has a venv but no worker yet and pulls vLLM
# (heavy), so it is left out of the default set and reachable via --with-orpheus / --only.
KNOWN=(kokoro chatterbox xtts dia ui orpheus)
DEFAULT_ENGINES=(kokoro chatterbox xtts dia ui)

# Kokoro is the only engine whose weights live in the repo (models/). Sizes are checked
# so a truncated download is re-fetched rather than silently used.
KOKORO_BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
declare -A KOKORO_FILES=(
  [kokoro-v1.0.onnx]=325532387
  [voices-v1.0.bin]=28214398
)

WITH_ORPHEUS=0
FORCE=0
DO_VENVS=1
DO_MODELS=1
ONLY=()

# Print the header comment block (everything after the shebang up to the first
# non-comment line) as help text, with the leading "# " stripped.
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; }

is_known() { local e; for e in "${KNOWN[@]}"; do [[ $e == "$1" ]] && return 0; done; return 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-orpheus) WITH_ORPHEUS=1 ;;
    --only)         shift; [[ $# -gt 0 ]] || { echo "--only needs a value, e.g. --only kokoro,ui" >&2; exit 2; }
                    IFS=',' read -ra ONLY <<< "$1" ;;
    --force)        FORCE=1 ;;
    --venvs-only)   DO_MODELS=0 ;;
    --models-only)  DO_VENVS=0 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; echo "try: ./setup.sh --help" >&2; exit 2 ;;
  esac
  shift
done

# Resolve the engine set: --only wins (validated), else the default set plus orpheus
# when requested.
if (( ${#ONLY[@]} )); then
  for e in "${ONLY[@]}"; do
    is_known "$e" || { echo "unknown engine: $e (known: ${KNOWN[*]})" >&2; exit 2; }
  done
  ENGINES=("${ONLY[@]}")
else
  ENGINES=("${DEFAULT_ENGINES[@]}")
  (( WITH_ORPHEUS )) && ENGINES+=(orpheus)
fi

# --- preflight -------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found. install it from https://github.com/astral-sh/uv" >&2
  exit 1
fi

# XTTS reaches FFmpeg through torchcodec at runtime via LD_LIBRARY_PATH. Warn early
# rather than fail mid-render; the venv still builds without it.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "warning: ffmpeg not on PATH. XTTS needs linuxbrew ffmpeg at runtime (see README)." >&2
fi

# dia's torch is the CUDA 12.6 build from the PyTorch index; every other engine
# resolves entirely from PyPI.
extra_index_for() {
  case "$1" in
    dia) echo "https://download.pytorch.org/whl/cu126" ;;
    *)   echo "" ;;
  esac
}

# --- venvs -----------------------------------------------------------------------

build_venv() {
  local engine="$1" venv="venv-$1" lock="requirements/$1.lock" idx
  if [[ ! -f $lock ]]; then
    echo "  skip $engine: $lock missing" >&2
    return
  fi
  if [[ -d $venv && $FORCE -eq 0 ]]; then
    echo "  $venv exists, skipping (use --force to recreate)"
    return
  fi
  echo "  building $venv from $lock"
  rm -rf "$venv"
  uv venv "$venv" --python "$PY" >/dev/null
  local args=(--python "$venv/bin/python" -r "$lock")
  idx="$(extra_index_for "$engine")"
  # With an extra index, uv's default strategy sources each package from the first
  # index that has it, so PyPI-only pins (e.g. certifi) become unsatisfiable. Both
  # indexes are trusted here, so consider all versions across them.
  [[ -n $idx ]] && args+=(--extra-index-url "$idx" --index-strategy unsafe-best-match)
  uv pip install "${args[@]}"
}

if (( DO_VENVS )); then
  echo "=== venvs (${ENGINES[*]}) ==="
  for engine in "${ENGINES[@]}"; do
    build_venv "$engine"
  done
fi

# --- models ----------------------------------------------------------------------

fetch_kokoro() {
  local name="$1" want="$2" dest="models/$1" have
  if [[ -f $dest ]]; then
    have="$(stat -c %s "$dest")"
    if [[ $have == "$want" ]]; then
      echo "  $name present ($have bytes), skipping"
      return
    fi
    echo "  $name wrong size ($have != $want), re-fetching"
  fi
  echo "  fetching $name"
  curl -fL --retry 3 -o "$dest" "$KOKORO_BASE/$name"
  have="$(stat -c %s "$dest")"
  if [[ $have != "$want" ]]; then
    echo "error: $name downloaded $have bytes, expected $want" >&2
    exit 1
  fi
}

if (( DO_MODELS )); then
  echo "=== models (kokoro weights -> models/) ==="
  mkdir -p models
  for name in "${!KOKORO_FILES[@]}"; do
    fetch_kokoro "$name" "${KOKORO_FILES[$name]}"
  done
  echo "  (Chatterbox / XTTS / Dia weights lazy-download from Hugging Face on first use)"
fi

echo "done. run: venv-ui/bin/python app.py"
