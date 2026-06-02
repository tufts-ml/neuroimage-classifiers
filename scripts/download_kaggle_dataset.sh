#!/bin/bash
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --mem=16g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=168:00:00

source ~/.bashrc

conda activate brain-scan-classifiers
#conda activate l3d_2024f_cuda12_1

#export KAGGLE_API_TOKEN=KGAT_cf194ee616b2e7060b1913e6a352bf01

#kaggle competitions download -c rsna-intracranial-hemorrhage-detection --path="/cluster/tufts/hugheslab/datasets/RSNA_ICH"
#kaggle competitions download -c rsna-2023-abdominal-trauma-detection --path="/cluster/tufts/hugheslab/datasets/RSNA_AT"
#kaggle competitions download -c rsna-str-pulmonary-embolism-detection --path="/cluster/tufts/hugheslab/datasets/RSNA_PE"

#cd /cluster/tufts/hugheslab/datasets/RSNA_PE
#unzip rsna-str-pulmonary-embolism-detection.zip

python ../src/preprocess_rsna_pe.py

conda deactivate
