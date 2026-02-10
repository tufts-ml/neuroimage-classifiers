import os
import re
import requests
import numpy as np
import pandas as pd
# Importing neuroimaging package(s)
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
os.environ['ANTS_RANDOM_SEED'] = '42'
import pydicom
import ants
from nilearn.image import resample_to_img
import torch
import torchmetrics

def correct_bias(image, iters=[50, 50, 50, 50]):
    '''
    Adapted from Clinica (see 
    github.com/aramis-lab/clinica/blob/dev/clinica/pipelines/t1_linear/anat_linear_utils.py)
    '''
    return ants.n4_bias_field_correction(
        image,
        mask=ants.get_mask(image),
        convergence={'iters': iters, 'tol': 1e-7},
        spline_param=600,
    )

def crop_nifti(image, ref_slices=None):
    '''
    Adapted from Clinica (see 
    github.com/aramis-lab/clinica/blob/dev/clinica/pipelines/t1_linear/anat_linear_utils.py)
    '''
    target_img = ants.image_read(get_ref_crop())
    if ref_slices is not None:
        target_img = ants.resample_image(target_img, (169, 208, ref_slices), True)
    return resample_to_img(
        source_img=image, 
        target_img=ants.to_nibabel(target_img),
        force_resample=True,
    )
    
def get_largest_scan(directory, scan_type):
    filepaths = [f'{root}/{filename}' for root, _, files in os.walk(directory) for filename in files if scan_type in filename and filename.endswith('.nii.gz') and len(ants.image_header_info(f'{root}/{filename}')['dimensions']) == 3]
    # Sort filepaths by 'anat' number in descending order.
    filepaths = sorted(filepaths, key=lambda f: int(re.search(r'anat(\d+)', f).group(1)), reverse=True)
    volumes = [np.prod(ants.image_header_info(filepath)['dimensions']) for filepath in filepaths]
    return '' if not len(filepaths) else filepaths[np.argmax(volumes)]

def get_ref_template():
    if not os.path.exists(os.path.abspath('./mni_icbm152_t1_tal_nlin_sym_09c.nii')):
        url = 'https://aramislab.paris.inria.fr/files/data/img_t1_linear/mni_icbm152_t1_tal_nlin_sym_09c.nii'
        response = requests.get(url)
        with open('./mni_icbm152_t1_tal_nlin_sym_09c.nii', 'wb') as file:
            file.write(response.content)
    return os.path.abspath('./mni_icbm152_t1_tal_nlin_sym_09c.nii')

def get_ref_crop():
    if not os.path.exists(os.path.abspath('./ref_cropped_template.nii.gz')):
        url = 'https://aramislab.paris.inria.fr/files/data/img_t1_linear/ref_cropped_template.nii.gz'
        response = requests.get(url)
        with open('./ref_cropped_template.nii.gz', 'wb') as file:
            file.write(response.content)
    return os.path.abspath('./ref_cropped_template.nii.gz')

def group_dicom_files_in_dir(directory):

    columns = ['Filename', 'SeriesNumber', 'SeriesDescription', 'InstanceNumber']
    dtypes = {'Filename': 'string', 'SeriesNumber': 'int', 'SeriesDescription': 'string', 'InstanceNumber': 'int'}
    folder_df = pd.DataFrame(columns=columns).astype(dtypes)

    for filename in os.listdir(directory):
        ds = pydicom.dcmread(f'{directory}/{filename}')
        folder_df.loc[len(folder_df)] = [filename, ds.SeriesNumber, ds.SeriesDescription, ds.InstanceNumber]
    
    folder_df = folder_df.sort_values(by=['SeriesNumber', 'InstanceNumber'])
    groupby_columns = ['SeriesNumber', 'SeriesDescription']
    groupby_dict = {'Filename': list, 'InstanceNumber': list}
    grouped_df = folder_df.groupby(groupby_columns).agg(groupby_dict).reset_index()
    grouped_df = grouped_df.rename(columns={'Filename': 'Filenames'})
        
    return grouped_df

def label_oasis3(directory, scan_types, condition):
    
    scan_df = pd.read_csv(f'{directory}/MR_Sessions.csv')
    scan_df['scan_day'] = scan_df['MR ID'].apply(lambda item: int(item.split('_')[-1][1:]))
    
    if condition.upper() == 'AND':
        scan_df = scan_df[scan_df['Scans'].apply(lambda item: all(scan_type in str(item) for scan_type in scan_types))]
    elif condition.upper() == 'OR':
        scan_df = scan_df[scan_df['Scans'].apply(lambda item: any(scan_type in str(item) for scan_type in scan_types))]
    else:
        raise ValueError('Condition must be \'AND\' or \'OR\'.')
        
    diagnosis_df = pd.read_csv(f'{directory}/ADRC_Clinical_Data.csv')
    diagnosis_df['diagnosis_day'] = diagnosis_df['ADRC_ADRCCLINICALDATA ID'].apply(lambda item: int(item.split('_')[-1][1:]))
    
    merged_df = pd.merge(scan_df, diagnosis_df, on='Subject', how='inner')
    merged_df['diff'] = merged_df['scan_day'] - merged_df['diagnosis_day']
    merged_df['abs_diff'] = abs(merged_df['scan_day'] - merged_df['diagnosis_day'])
    # Diagnosis date must be between 80 days before or 365 days after scan date.
    merged_df = merged_df[(merged_df['diff'] <= 80) & (merged_df['diff'] >= -365)]
    merged_df = merged_df.loc[merged_df.groupby(['MR ID'])['abs_diff'].idxmin()]
    
    merged_df['Alzheimer\'s'] = merged_df['cdr'].apply(lambda item: 0 if item == 0.0 else 1)
    merged_df['paths'] = merged_df['MR ID'].apply(lambda item: [get_largest_scan(f'{directory}/{item}', scan_type=scan_type) for scan_type in scan_types])
    
    return merged_df

def register_image(moving, ref_slices=None):
    '''
    Adapted from Clinica (see 
    github.com/aramis-lab/clinica/blob/dev/clinica/pipelines/t1_linear/anat_linear_utils.py)
    '''
    fixed = ants.image_read(get_ref_template())
    if ref_slices is not None:
        fixed = ants.resample_image(fixed, (193, 229, ref_slices), True)
    return ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform='antsRegistrationSyNQuick[a]',
        random_seed=42,
    )['warpedmovout']
