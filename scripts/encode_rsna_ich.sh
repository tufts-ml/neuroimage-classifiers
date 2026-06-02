#!/bin/bash
#SBATCH --array=9-11%8
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
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=1001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv" --seed=1001'
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=2001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv" --seed=2001'
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset/ViT_B_16/seed=3001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv" --seed=3001'
    
    
    
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_subset/ViT_B_16/seed=1001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/subset_labels.csv" --seed=1001'
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_subset/ViT_B_16/seed=2001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/subset_labels.csv" --seed=2001'
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_subset/ViT_B_16/seed=3001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/subset_labels.csv" --seed=3001'
    
    
    
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset_best_possible_bag-level/ViT_B_16/seed=1001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv" --seed=1001'
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset_best_possible_bag-level/ViT_B_16/seed=2001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv" --seed=2001'
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset_best_possible_bag-level/ViT_B_16/seed=3001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv" --seed=3001'
    
    
    
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset_best_possible_instance-level/ViT_B_16/seed=1001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv" --seed=1001'
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset_best_possible_instance-level/ViT_B_16/seed=2001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv" --seed=2001'
    'python ../src/encode_rsna_ich.py --encoded_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_ICH_full_dataset_best_possible_instance-level/ViT_B_16/seed=3001" --encoder="ViT-B/16" --numpy_dir="/cluster/tufts/hugheslab/datasets/RSNA_ICH_numpy" --csv_path="/cluster/tufts/hugheslab/datasets/RSNA_ICH/full_dataset_labels.csv" --seed=3001'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate