# Environment configuration — usually the only file you need to edit.
# Every value can be overridden by exporting it before this file is sourced.

# --- conda ----------------------------------------------------------------
# Root of the conda/mamba installation. Auto-discovered when unset; export
# CONDA_ROOT explicitly if the guesses below do not fit your machine.
if [ -z "${CONDA_ROOT:-}" ]; then
    CONDA_ROOT="$(conda info --base 2>/dev/null || true)"
fi
if [ -z "${CONDA_ROOT:-}" ] && [ -n "${CONDA_EXE:-}" ]; then
    CONDA_ROOT="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
fi
if [ -z "${CONDA_ROOT:-}" ] && [ -d "${HOME}/miniconda3" ]; then
    CONDA_ROOT="${HOME}/miniconda3"
fi

# Name of the conda env holding the training dependencies
# (see requirements.txt / docs/INSTALL.md).
CONDA_ENV_NAME="${CONDA_ENV_NAME:-diffusion_nft}"

# --- reward model ---------------------------------------------------------
# Default reward model served over vLLM. See README "Reward Server".
VLM_RM_MODEL_DEFAULT="${VLM_RM_MODEL_DEFAULT:-CodeGoat24/WorldReward-qwen35-9b}"

# --- derived — do not edit ------------------------------------------------
CONDA_SH="${CONDA_ROOT:-}/etc/profile.d/conda.sh"
