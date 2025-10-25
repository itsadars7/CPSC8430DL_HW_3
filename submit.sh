#!/bin/bash
#SBATCH --job-name=hw3_bert_squad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu_v100_16gb|gpu_v100_32gb'
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o logs/%x_%A.out
#SBATCH -e logs/%x_%A.err

set -euo pipefail
mkdir -p logs .hf_cache .ds_cache

PY="$(command -v python3 || command -v python)"

# Ensure GPU PyTorch + transformers stack
"$PY" -m pip install --upgrade pip
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio
"$PY" -m pip install transformers datasets evaluate accelerate

# --- RUNTIME TUNING ---
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HF_HOME=$PWD/.hf_cache
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# --- RUN ---
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# Run training 
srun "$PY" Adarsha_BERT_SQuAD_1.py