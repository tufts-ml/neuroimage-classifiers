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
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=0 --stop=80'
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=80 --stop=160'
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=160 --stop=240'
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=240 --stop=320'
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=320 --stop=400'
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=400 --stop=480'
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=480 --stop=560'
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=560 --stop=640'
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=640 --stop=720'
    'python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy" --start=720 --stop=800'
)

eval "${experiments[$SLURM_ARRAY_TASK_ID]}"

conda deactivate