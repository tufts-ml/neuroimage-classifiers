import argparse
import os
import numpy as np
# Importing neuroimaging package(s)
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
os.environ['ANTS_RANDOM_SEED'] = '42'
import ants
# Importing our custom module(s)
import mri

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--condition", default="ALL", help="TODO")
    parser.add_argument("--nifti_dir", help="Directory to nifti dataset", type=str)
    parser.add_argument("--numpy_dir", help="Directory to save numpy dataset", type=str)
    parser.add_argument("--scan_types", default=["T1w", "T2w"], help="Scan types (default: [\"T1w\", \"T2w\"])", nargs="+", type=str)
    parser.add_argument("--start", default=0, help="Start index (default: 0)", type=int)
    parser.add_argument("--stop", default=10, help="End index (default: 10)",  type=int)
    args = parser.parse_args()
    
    os.makedirs(args.numpy_dir, exist_ok=True)
    
    labels_df = mri.label_oasis3(args.nifti_dir, args.scan_types, args.condition)
    
    for index, row in labels_df.iloc[args.start:args.stop].iterrows():
        
        images = []
        
        for path in row.paths:
            
            image = ants.image_read(path)
            image = mri.correct_bias(image)
            image = mri.register_image(image)
            image = mri.crop_nifti(ants.to_nibabel(image))
            image = ants.convert_nibabel.from_nibabel(image)
            images.append(image.numpy())
            
        print(f"{args.numpy_dir}/{row['MR ID']}.npz")
        np.savez(f"{args.numpy_dir}/{row['MR ID']}.npz", np.array(images))
        