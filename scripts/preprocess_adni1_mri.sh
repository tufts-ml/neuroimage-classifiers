#!/bin/bash
#SBATCH --array=0-9%10
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --mem=64g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=preempt
#SBATCH --time=168:00:00

source ~/.bashrc
conda activate brain-scan-classifiers

# Define an array of commands
experiments=(
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=0 --stop=229"
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=229 --stop=458"
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=458 --stop=688"
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=688 --stop=917"
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=917 --stop=1147"
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=1147 --stop=1376"
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=1376 --stop=1605"
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=1605 --stop=1835"
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=1835 --stop=2064"
    "python ../src/preprocess_adni1_mri.py --csv_filename='ADNI1_Complete_1Yr_1.5T.csv' --nifti_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T' --numpy_dir='/cluster/tufts/hugheslab/datasets/ADNI1_Complete_1Yr_1.5T_numpy' --start=2064 --stop=2294"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate