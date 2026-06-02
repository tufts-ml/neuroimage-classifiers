import ast
import numpy as np
import pandas as pd
# Importing neuroimaging package(s)
import pydicom

# python ../src/preprocess_rsna_pe.py 
if __name__ == "__main__":

    dicom_dir = "/cluster/tufts/hugheslab/datasets/RSNA_PE"
    
    #instance_level_labels_df = pd.read_csv(f"{dicom_dir}/train.csv")
    #instance_level_labels_df["path"] = instance_level_labels_df.apply(lambda row: f"{dicom_dir}/train/{row.StudyInstanceUID}/{row.SeriesInstanceUID}/{row.SOPInstanceUID}.dcm", axis=1)
    #instance_level_labels_df["instance_number"] = instance_level_labels_df.apply(lambda row: pydicom.dcmread(row.path, specific_tags=["InstanceNumber"], stop_before_pixels=True).InstanceNumber, axis=1)
    #instance_level_labels_df = instance_level_labels_df.sort_values(by=["StudyInstanceUID", "SeriesInstanceUID", "instance_number"], ascending=[True, True, True])
    #labels_df = instance_level_labels_df.groupby(["StudyInstanceUID", "SeriesInstanceUID"]).agg(list).reset_index()
    #labels_df.to_csv(f"{dicom_dir}/labels.csv", index=False)
    
    labels_df = pd.read_csv(f"{dicom_dir}/labels.csv")
    columns = ["pe_present_on_image", "path", "instance_number"]
    labels_df[columns] = labels_df[columns].apply(lambda col: col.map(ast.literal_eval))
    
    numpy_dir = "/cluster/tufts/hugheslab/datasets/RSNA_PE_numpy"
    for row_index, row in labels_df.iterrows():
        scan = []
        for path in row.path:
            ds = pydicom.dcmread(path)
            pixel_array = ds.pixel_array.astype(np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1))
            intercept = float(getattr(ds, "RescaleIntercept", 0))
            rescaled = pixel_array * slope + intercept
            scan.append(rescaled)
        print(np.expand_dims(np.stack(scan, axis=-1), axis=0).shape)
        np.savez(f"{numpy_dir}/{row.StudyInstanceUID}.npz", np.expand_dims(np.stack(scan, axis=-1), axis=0))
        
    np.savez(f"{numpy_dir}/done.npz", np.array([1.0]))
        