

# Embeddings

```python
train_data = torch.load("/data/TUFTS_Datasets/TUFTS_MRI_Training_Dataset_encoded/ViT_B_16/test_site_ids=13_14_train_site_ids=1_2_3_6_7_9_10_11_val_site_ids=12/train.pth", map_location=torch.device("cpu"), weights_only=False)
val_data = torch.load("/data/TUFTS_Datasets/TUFTS_MRI_Training_Dataset_encoded/ViT_B_16/test_site_ids=13_14_train_site_ids=1_2_3_6_7_9_10_11_val_site_ids=12/val.pth", map_location=torch.device("cpu"), weights_only=False)
test_data = torch.load("/data/TUFTS_Datasets/TUFTS_MRI_Training_Dataset_encoded/ViT_B_16/test_site_ids=13_14_train_site_ids=1_2_3_6_7_9_10_11_val_site_ids=12/test.pth", map_location=torch.device("cpu"), weights_only=False)

# Slice-level embeddings for all scans concatenated together.
data["X"].shape == (number_of_slices_slices, embedding_dim)
# Number of slices in each scan.
data["lengths"] = (number_of_slices_in_scan_1, ..., number_of_slices_in_scan_N)
# Scan-level labels.
data["y"].shape = (number_of_scans, 2)
CBI_label, WMD_label = data["y"][i]

# Extract T1 and T2 embeddings.
T1_embedding = data["X"][i, :756]
T2_embedding = data["X"][i, 756:]
```
