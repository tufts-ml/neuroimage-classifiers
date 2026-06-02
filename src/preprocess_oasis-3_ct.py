import argparse
import os
import numpy as np
# Importing neuroimaging package(s)
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
os.environ['ANTS_RANDOM_SEED'] = '42'
import ants
# Importing our custom module(s)
import ct

# python ../src/preprocess_oasis-3_ct.py --nifti_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_CT' --numpy_dir='/cluster/tufts/hugheslab/datasets/OASIS-3_CT_numpy' --start=0 --stop=662
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument("--nifti_dir", help="Directory to nifti dataset", type=str)
    parser.add_argument("--numpy_dir", help="Directory to save numpy dataset", type=str)
    parser.add_argument("--start", default=0, help="Start index (default: 0)", type=int)
    parser.add_argument("--stop", default=10, help="End index (default: 10)",  type=int)
    args = parser.parse_args()
    
    os.makedirs(args.numpy_dir, exist_ok=True)
    
    labels_df = ct.label_oasis3(args.nifti_dir)
    labels_df = labels_df[labels_df["paths"].apply(lambda paths: all(path != "" for path in paths))]
    
    for index, row in labels_df.iloc[args.start:args.stop].iterrows():
        
        images = []
        
        for path in row.paths:

            image = ants.image_read(path).numpy()
            image = ct.strip_skull(image)
            images.append(image)
            
        print(f"{args.numpy_dir}/{row['XNAT_CTSESSIONDATA ID']}.npz")
        np.savez(f"{args.numpy_dir}/{row['XNAT_CTSESSIONDATA ID']}.npz", np.array(images))
        