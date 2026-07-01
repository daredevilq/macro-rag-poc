#!/bin/bash -l
#SBATCH -J build_rule_kb
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH -A plgagentsmith-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/build_kb_%j.out
#SBATCH --error=logs/build_kb_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"
mkdir -p logs

module purge
module load GCCcore/13.2.0
module load Python/3.11.5
module load CUDA/12.1.1

nvidia-smi -L || true

SCRATCH_DIR="${SCRATCH:-$PWD/.scratch}"
export XDG_CACHE_HOME="$SCRATCH_DIR/.cache"
export PIP_CACHE_DIR="$SCRATCH_DIR/.cache/pip"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

if [[ ! -d ".venv-proto" ]]; then
  python3 -m venv .venv-proto
fi
source .venv-proto/bin/activate
python3 -m pip install --index-url https://download.pytorch.org/whl/cu124 torch
python3 -m pip install -r requirements-athena.txt

srun python3 -u -m prototype.expert_kb.vector_store

echo "LanceDB in results/lancedb/"
