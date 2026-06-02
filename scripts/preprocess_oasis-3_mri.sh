#!/bin/bash
#SBATCH --array=0-9%10
#SBATCH --error=/cluster/tufts/hugheslab/eharve06/slurmlog/err/log_%j.err
#SBATCH --mem=64g
#SBATCH --ntasks=4
#SBATCH --output=/cluster/tufts/hugheslab/eharve06/slurmlog/out/log_%j.out
#SBATCH --partition=hugheslab
#SBATCH --time=120:00:00

source ~/.bashrc
conda activate brain-scan-classifiers

# Define an array of commands
experiments=(
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=0 --stop=162"
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=162 --stop=324"
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=324 --stop=486"
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=486 --stop=648"
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=648 --stop=810"
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=810 --stop=972"
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=972 --stop=1134"
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=1134 --stop=1296"
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=1296 --stop=1458"
    "python ../src/preprocess_oasis-3_mri.py --condition='AND' --nifti_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI' --numpy_directory='/cluster/tufts/hugheslab/datasets/OASIS-3_MRI_numpy' --scan_type 'T1w' 'T2w' --start=1458 --stop=1620"
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate