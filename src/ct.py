import os
import re
import numpy as np
import pandas as pd
# Importing neuroimaging package(s)
os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = '1'
os.environ['ANTS_RANDOM_SEED'] = '42'
import ants

def get_latest_scan(directory):
    filepaths = [f"{root}/{filename}" for root, _, files in os.walk(directory) for filename in files if filename.endswith(".nii.gz") and len(ants.image_header_info(f"{root}/{filename}")["dimensions"]) == 3]
    return "" if not len(filepaths) else filepaths[-1]

def label_oasis3(directory):
    
    scan_df = pd.read_csv(f"{directory}/CT_Sessions.csv")
    scan_df["scan_day"] = scan_df["XNAT_CTSESSIONDATA ID"].apply(lambda item: int(item.split('_')[-1][1:]))
            
    diagnosis_df = pd.read_csv(f"{directory}/ADRC_Clinical_Data.csv")
    diagnosis_df["diagnosis_day"] = diagnosis_df["ADRC_ADRCCLINICALDATA ID"].apply(lambda item: int(item.split('_')[-1][1:]))
    
    merged_df = pd.merge(scan_df, diagnosis_df, on="Subject", how="inner")
    merged_df["diff"] = merged_df["scan_day"] - merged_df["diagnosis_day"]
    merged_df["abs_diff"] = abs(merged_df["scan_day"] - merged_df["diagnosis_day"])
    # Diagnosis date must be between 80 days before or 360 days after scan date.
    # NOTE:
    merged_df = merged_df[(merged_df["diff"]<=80)&(merged_df["diff"]>=-365)]
    merged_df = merged_df.loc[merged_df.groupby(["XNAT_CTSESSIONDATA ID"])["abs_diff"].idxmin()]
    
    merged_df["Alzheimer\'s"] = merged_df["cdr"].apply(lambda item: 0 if item == 0.0 else 1)
    merged_df["paths"] = merged_df["XNAT_CTSESSIONDATA ID"].apply(lambda item: [get_latest_scan(f"{directory}/{item}")])
    
    return merged_df

def strip_skull(image, minimum=-100, maximum=300):
    image[image < minimum] = minimum
    image[image > maximum] = maximum
    return image
