#!/bin/bash
#SBATCH --array=0-191%8
#SBATCH --error=/cluster/tufts/hugheslab/dloevl01/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/dloevl01/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=48:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Hyperparameter sweep for OASIS-3 MRI experiments
# Learning rates: 0.1, 0.01, 0.001, 0.0001 (4 values)
# Alpha (weight decay): 1.0, 0.1, 0.01, 0.001, 0.0001, 1e-05, 1e-06, 0.0 (8 values)
# Seeds: 1001, 2001, 3001 (3 values)
# Total: 4 * 8 * 3 = 96 combinations for each approach (embedding/prediction)
# Grand total: 96 * 2 = 192 experiments

# Arrays of hyperparameters
lrs=(0.1 0.01 0.001 0.0001)
alphas=(1.0 0.1 0.01 0.001 0.0001 1e-05 1e-06 0.0)
seeds=(1001 2001 3001)

# Calculate indices
task_id=$SLURM_ARRAY_TASK_ID

# Determine if embedding-level (first 96) or prediction-level (last 96)
if [ $task_id -lt 96 ]; then
    embedding_flag="--embedding_level"
    approach="embedding"
    idx=$task_id
else
    embedding_flag=""
    approach="prediction"
    idx=$((task_id - 96))
fi

# Calculate hyperparameter indices
# idx = lr_idx * 24 + alpha_idx * 3 + seed_idx
lr_idx=$((idx / 24))
remainder=$((idx % 24))
alpha_idx=$((remainder / 3))
seed_idx=$((remainder % 3))

lr=${lrs[$lr_idx]}
alpha=${alphas[$alpha_idx]}
seed=${seeds[$seed_idx]}

# Dataset and output directories
dataset_dir="/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/ViT_B_16/seed=${seed}"
experiments_dir="/cluster/tufts/hugheslab/dloevl01/Dec_2025/pooling/experiments/oasis-3_mri/${approach}"

# Model name
model_name="alpha=${alpha}_lr=${lr}_pooling=smAP_seed=${seed}"

echo "Running: approach=${approach}, lr=${lr}, alpha=${alpha}, seed=${seed}"

python ../src/oasis-3.py \
    --alpha=${alpha} \
    --batch_size=64 \
    --criterion='L1' \
    --dataset_dir="${dataset_dir}" \
    ${embedding_flag} \
    --epochs=1000 \
    --experiments_dir="${experiments_dir}" \
    --lr=${lr} \
    --model_name="${model_name}" \
    --pooling='SmAP' \
    --save \
    --seed=${seed} \
    --weight_decay=0.0

conda deactivate
