#!/bin/bash
#SBATCH --array=0-2%4
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=168:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    "python ../src/encode_oasis-3.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/MedSAM_neck=False/seed=1001' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --seed=1001"
    "python ../src/encode_oasis-3.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/MedSAM_neck=False/seed=2001' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --seed=2001"
    "python ../src/encode_oasis-3.py --encoded_dir='/cluster/tufts/hugheslab/eharve06/encoded_OASIS-3_MRI/MedSAM_neck=False/seed=3001' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --seed=3001"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate