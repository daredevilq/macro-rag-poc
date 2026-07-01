#!/bin/bash -l
#SBATCH -J whisper_transcribe
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH -A plgearninigsconf-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/whisper_%j.out
#SBATCH --error=logs/whisper_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$SUBMIT_DIR"
mkdir -p logs

module purge
module load GCCcore/13.2.0
module load Python/3.11.5
module load CUDA/12.1.1

nvidia-smi -L || true

if [[ -n "${SCRATCH:-}" ]]; then
  export HF_HOME="$SCRATCH/.cache/huggingface"
  mkdir -p "$HF_HOME"
fi

mkdir -p results/transcripts

if [[ ! -d ".venv-transcribe" ]]; then
  python3 -m venv .venv-transcribe
fi
source .venv-transcribe/bin/activate
python3 -m pip install --upgrade pip

python3 -m pip install -r transcription/requirements.txt
python3 -m pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"

NV_LIBS="$(python3 -c 'import importlib,os; print(":".join(os.path.join(p,"lib") for m in ("nvidia.cudnn","nvidia.cublas","nvidia.cuda_nvrtc") for p in getattr(importlib.import_module(m),"__path__",[]) if os.path.isdir(os.path.join(p,"lib"))))')"
export LD_LIBRARY_PATH="${NV_LIBS}:${LD_LIBRARY_PATH:-}"

srun python3 -u transcription/transcribe.py

echo "job completed"
