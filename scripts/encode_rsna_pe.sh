#!/bin/bash
#SBATCH --array=0-2%3
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --nodelist=cc1gpu[001,002,003,004,005]
#SBATCH --nodes=1
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=gpu
#SBATCH --time=168:00:00

source ~/.bashrc
conda activate l3d_2024f_cuda12_1

# Define an array of commands
experiments=(
    'python ../src/encode_rsna_pe.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE_best_possible_bag-level/ViT_B_16/seed=1001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_PE_numpy" --seed=1001'
    'python ../src/encode_rsna_pe.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE_best_possible_bag-level/ViT_B_16/seed=2001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_PE_numpy" --seed=2001'
    'python ../src/encode_rsna_pe.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE_best_possible_bag-level/ViT_B_16/seed=3001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_PE_numpy" --seed=3001'



    'python ../src/encode_rsna_pe.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=1001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_PE_numpy" --seed=1001'
    'python ../src/encode_rsna_pe.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=2001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_PE_numpy" --seed=2001'
    'python ../src/encode_rsna_pe.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=3001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_PE_numpy" --seed=3001'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate