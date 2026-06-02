import argparse
import os
import shutil
import pandas as pd
# Importing neuroimaging package(s)
import SimpleITK as sitk
# Importing our custom module(s)
import mri

# python ../src/convert_kpsc_mri.py --csv_filename='ADDF_2024_External_MRimage_800.csv' --dicom_dir='/cluster/tufts/hugheslabkp/data_irb_required/ADDF_2024_External_MRimage_800' --nifti_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_nifti'
if __name__== '__main__':
    parser = argparse.ArgumentParser(description='main.py') 
    parser.add_argument('--csv_filename', help='.csv filename', type=str)
    parser.add_argument('--dicom_dir', help='Directory to dicom dataset', type=str)
    parser.add_argument('--nifti_dir', help='Directory to save nifti dataset', type=str)
    args = parser.parse_args()

    labels_df = pd.read_csv(f'{args.dicom_dir}/{args.csv_filename}')
    exceptions = {'STUDY_0230': 8, 'STUDY_0462': 9, 'STUDY_0836': 9, 'STUDY_1109': 4}
    
    for study_id in labels_df['STUDY_ID']:
        
        os.makedirs(f'{args.nifti_dir}/{study_id}', exist_ok=True)
        
        grouped_df = mri.group_dicom_files_in_dir(f'{args.dicom_dir}/ADDF_2024_External_STD_T1_T2/{study_id}')
        
        if study_id in exceptions:
            grouped_df = grouped_df[grouped_df.SeriesNumber != exceptions[study_id]]

        t1_rows = grouped_df[grouped_df.SeriesDescription.str.contains('T1', case=False, na=False)]
        t2_rows = grouped_df[grouped_df.SeriesDescription.str.contains('T2', case=False, na=False)]
        t1_row = t1_rows.loc[t1_rows['SeriesNumber'].idxmax()]
        t2_row = t2_rows.loc[t2_rows['SeriesNumber'].idxmax()]
        t1_filepaths = [f'{args.dicom_dir}/ADDF_2024_External_STD_T1_T2/{study_id}/{filename}' for filename in t1_row.Filenames]
        t2_filepaths = [f'{args.dicom_dir}/ADDF_2024_External_STD_T1_T2/{study_id}/{filename}' for filename in t2_row.Filenames]
        
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(t1_filepaths)
        image = reader.Execute()
        sitk.WriteImage(image, f'{args.nifti_dir}/{study_id}/T1.nii.gz')
        reader.SetFileNames(t2_filepaths)
        image = reader.Execute()
        sitk.WriteImage(image, f'{args.nifti_dir}/{study_id}/T2.nii.gz')
        