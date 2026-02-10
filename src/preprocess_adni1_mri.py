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

def get_adni1_nifti_paths(row, root):
    subject_id, image_id = row['Subject'], row['Image Data ID']
    paths = [
        f'{root}/{filename}'
        for root, _, files in os.walk(f'{root}/{subject_id}')
        for filename in files
        if filename.endswith(f'{image_id}.nii')
    ]
    assert len(paths) == 1
    return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--csv_filename', help='.csv filename', type=str)
    parser.add_argument('--nifti_dir', help='Directory to nifti dataset', type=str)
    parser.add_argument('--numpy_dir', help='Directory to save numpy dataset', type=str)
    parser.add_argument('--start', default=0, help='Start index (default: 0)', type=int)
    parser.add_argument('--stop', default=10, help='End index (default: 10)',  type=int)
    args = parser.parse_args()
    
    os.makedirs(args.numpy_dir, exist_ok=True)
    
    labels_df = pd.read_csv(f'{args.nifti_dir}/{args.csv_filename}')
    labels_df['paths'] = labels_df.apply(get_adni1_nifti_paths, root=args.nifti_dir, axis=1)
    
    for index, row in labels_df.iloc[args.start:args.stop].iterrows():
        
        subject_id, image_id = row['Subject'], row['Image Data ID']
        
        print(f'{args.numpy_dir}/{subject_id}__{image_id}.npz')
        if not os.path.exists(f'{args.numpy_dir}/{subject_id}__{image_id}.npz'):
        
            images = []

            for path in row.paths:

                image = ants.image_read(path)
                image = mri.correct_bias(image)
                image = mri.register_image(image)
                image = mri.crop_nifti(ants.to_nibabel(image))
                image = ants.convert_nibabel.from_nibabel(image)
                images.append(image.numpy())

            #print(f'{args.numpy_dir}/{subject_id}__{image_id}.npz')
            np.savez(f'{args.numpy_dir}/{subject_id}__{image_id}.npz', np.array(images))
