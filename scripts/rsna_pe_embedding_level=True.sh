#!/bin/bash
#SBATCH --array=0-11%10
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
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=1001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.1 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.1_pooling=TransMIL_seed=1001" --pooling="TransMIL" --save --seed=1001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=2001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.1 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.1_pooling=TransMIL_seed=2001" --pooling="TransMIL" --save --seed=2001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=3001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.1 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.1_pooling=TransMIL_seed=3001" --pooling="TransMIL" --save --seed=3001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=1001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.01 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.01_pooling=TransMIL_seed=1001" --pooling="TransMIL" --save --seed=1001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=2001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.01 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.01_pooling=TransMIL_seed=2001" --pooling="TransMIL" --save --seed=2001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=3001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.01 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.01_pooling=TransMIL_seed=3001" --pooling="TransMIL" --save --seed=3001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=1001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.001 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.001_pooling=TransMIL_seed=1001" --pooling="TransMIL" --save --seed=1001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=2001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.001 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.001_pooling=TransMIL_seed=2001" --pooling="TransMIL" --save --seed=2001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=3001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.001 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.001_pooling=TransMIL_seed=3001" --pooling="TransMIL" --save --seed=3001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=1001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.0001 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.0001_pooling=TransMIL_seed=1001" --pooling="TransMIL" --save --seed=1001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=2001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.0001 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.0001_pooling=TransMIL_seed=2001" --pooling="TransMIL" --save --seed=2001 --weight_decay=0.0'
    'python ../src/oasis-3.py --alpha=1e-05 --batch_size=64 --criterion="GuidedL1" --dataset_dir="/cluster/tufts/hugheslab/datasets/encoded_RSNA_PE/ViT_B_16/seed=3001" --epochs=1000 --embedding_level --experiments_dir="/cluster/tufts/hugheslab/eharve06/pooling/experiments/RSNA_PE_beta2=1.0_embedding_level=True" --lr=0.0001 --model_name="alpha=1e-05_criterion=GuidedL1_lr=0.0001_pooling=TransMIL_seed=3001" --pooling="TransMIL" --save --seed=3001 --weight_decay=0.0'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate