import argparse
import os
import numpy as np
import pandas as pd
# Importing neuroimaging package(s)
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
os.environ['ANTS_RANDOM_SEED'] = '42'
import ants
# Importing our custom module(s)
import mri
import utils

# python ../src/preprocess_kpsc_mri.py --csv_filename="ADDF_2024_External_MRimage_800.csv" --dicom_dir="/cluster/home/eharve06/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800" --nifti_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_nifti" --numpy_dir="/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_numpy" --start=0 --stop=800
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--csv_filename", help=".csv filename", type=str)
    parser.add_argument("--dicom_dir", help="Directory to dicom dataset", type=str)
    parser.add_argument("--nifti_dir", help="Directory to nifti dataset", type=str)
    parser.add_argument("--numpy_dir", help="Directory to save numpy dataset", type=str)
    parser.add_argument("--start", default=0, help="Start index (default: 0)", type=int)
    parser.add_argument("--stop", default=10, help="End index (default: 10)",  type=int)
    args = parser.parse_args()

    os.makedirs(args.numpy_dir, exist_ok=True)
    
    labels_df = pd.read_csv(f"{args.dicom_dir}/{args.csv_filename}")

    for study_id in labels_df.iloc[args.start:args.stop]["STUDY_ID"]:

        print(f"{args.numpy_dir}/{study_id}.npz")
        #if not os.path.exists(f"{args.numpy_dir}/{study_id}.npz"):
            
        t1 = ants.image_read(f"{args.nifti_dir}/{study_id}/T1.nii.gz")
        t1 = mri.correct_bias(t1)
        #t1 = mri.register_image(t1, ref_slices=23)
        t1 = mri.register_image(t1)
        #t1 = mri.crop_nifti(ants.to_nibabel(t1), ref_slices=23)
        t1 = mri.crop_nifti(ants.to_nibabel(t1))
        t1 = ants.convert_nibabel.from_nibabel(t1)

        t2 = ants.image_read(f"{args.nifti_dir}/{study_id}/T2.nii.gz")
        t2 = mri.correct_bias(t2)
        #t2 = mri.register_image(t2, ref_slices=23)
        t2 = mri.register_image(t2)
        #t2 = mri.crop_nifti(ants.to_nibabel(t2), ref_slices=23)
        t2 = mri.crop_nifti(ants.to_nibabel(t2))
        t2 = ants.convert_nibabel.from_nibabel(t2)

        np.savez(f"{args.numpy_dir}/{study_id}.npz", np.stack((t1.numpy(), t2.numpy())))
