import argparse
import os
import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
# Importing our custom module(s)
import datasets
import utils

class Squeeze(torch.nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze() if self.dim is None else x.squeeze(self.dim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="encode_rsna.py")
    parser.add_argument("--encoded_dir", help="Directory to save encoded dataset", type=str)
    parser.add_argument("--encoder", help="Encoder pre-trained on ImageNet", type=str)
    parser.add_argument("--numpy_dir", help="Directory to numpy dataset", type=str)
    parser.add_argument("--seed", default=42, help="Random seed (default: 42)", type=int)
    args = parser.parse_args()

    os.makedirs(args.encoded_dir, exist_ok=True)

    labels_df = pd.read_csv(f"{args.numpy_dir}/labels.csv")
    columns = ["pe_present_on_image", "instance_number"]
    labels_df[columns] = labels_df[columns].apply(lambda col: col.map(ast.literal_eval))

    grouped_df = labels_df.groupby("StudyInstanceUID")["PE"].agg(lambda x: x.mode()[0]).reset_index()
    ids, id_labels = grouped_df["StudyInstanceUID"], grouped_df["PE"]
    train_and_val_ids, test_ids, train_and_val_id_labels, test_id_labels = train_test_split(ids, id_labels, test_size=1/6, random_state=args.seed, stratify=id_labels)
    train_ids, val_ids = train_test_split(train_and_val_ids, test_size=1/5, random_state=args.seed, stratify=train_and_val_id_labels)

    train_df = labels_df[labels_df["StudyInstanceUID"].isin(train_ids)]
    val_df = labels_df[labels_df["StudyInstanceUID"].isin(val_ids)]
    test_df = labels_df[labels_df["StudyInstanceUID"].isin(test_ids)]
    
    with torch.no_grad():

        resize_size = 1024 if args.encoder == "MedSAM" else 224
        transform = torchvision.transforms.Compose([
            lambda path: utils.read_npz(path),
            lambda image: image.permute(3, 0, 1, 2),
            lambda image: utils.pad_image(image),
            torchvision.transforms.Resize(size=(resize_size, resize_size)),
        ])

        #train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[["PE"]].values, dtype=torch.float32), transform)
        train_dataset = datasets.MILPathDataset(train_df.path.values, list(zip(torch.tensor(labels_df[["PE"]].values, dtype=torch.float32), labels_df["pe_present_on_image"].values)), transform)

        means, stds = [], []

        for image, num_slices, label in train_dataset:
            means.append(torch.mean(image, dim=(0, 2, 3)).tolist())
            stds.append(torch.std(image, dim=(0, 2, 3)).tolist())

        mean = torch.tensor(means).mean(dim=0)
        std = torch.tensor(stds).mean(dim=0)
            
    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        lambda image: image.permute(3, 0, 1, 2),
        lambda image: utils.pad_image(image),
        torchvision.transforms.Resize(size=(resize_size, resize_size)),
        lambda image: (image - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1),
    ])

    #train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[["PE"]].values, dtype=torch.float32), transform)
    train_dataset = datasets.MILPathDataset(train_df.path.values, list(zip(torch.tensor(train_df[["PE"]].values, dtype=torch.float32), train_df["pe_present_on_image"].values)), transform)
    #val_dataset = datasets.MILPathDataset(val_df.path.values, torch.tensor(val_df[["PE"]].values, dtype=torch.float32), transform)
    val_dataset = datasets.MILPathDataset(val_df.path.values, list(zip(torch.tensor(val_df[["PE"]].values, dtype=torch.float32), val_df["pe_present_on_image"].values)), transform)
    #test_dataset = datasets.MILPathDataset(test_df.path.values, torch.tensor(test_df[["PE"]].values, dtype=torch.float32), transform)
    test_dataset = datasets.MILPathDataset(test_df.path.values, list(zip(torch.tensor(test_df[["PE"]].values, dtype=torch.float32), test_df["pe_present_on_image"].values)), transform)

    assert args.encoder in ["ViT-B/16", "ConvNeXt-Tiny", "MedSAM"]
    if args.encoder == "ViT-B/16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights(weights))
        model.conv_proj.weight.data = model.conv_proj.weight.data.sum(dim=1, keepdim=True)
        model.conv_proj.in_channels = 1
        model.heads = torch.nn.Identity()
    elif args.encoder == "ConvNeXt-Tiny":
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights(weights))
        model.features[0][0].weight.data = model.features[0][0].weight.data.sum(dim=1, keepdim=True)
        model.features[0][0].in_channels = 1
        model.classifier[2] = torch.nn.Identity()
    elif args.encoder == "MedSAM":
        from segment_anything import sam_model_registry
        checkpoint_path = '/cluster/tufts/hugheslab/eharve06/pooling/models/medsam_vit_b.pth'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        medsam = sam_model_registry['vit_b']()
        medsam.load_state_dict(checkpoint)
        model = medsam.image_encoder
        model.patch_embed.proj.weight.data = model.patch_embed.proj.weight.data.sum(dim=1, keepdim=True)
        model.patch_embed.proj.in_channels = 1
        model.neck = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Squeeze(dim=(2, 3)),
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    #X, lengths, y = [], [], []
    X, lengths, y, lengths_y = [], [], [], []
    
    for image, length, label in train_dataset:
        
        label, length_labels = label
        
        embeddings = torch.cat([
            utils.encode_image(model, image[:,c].unsqueeze(1)) 
            for c in range(image.shape[1])
        ], dim=-1)
        #embeddings = torch.cat([
        #    torch.cat([
        #        utils.encode_image(model, image[s:s+5,c].unsqueeze(1))
        #        for s in range(0, image.shape[0], 5)
        #    ], dim=0)
        #    for c in range(image.shape[1])
        #], dim=-1)
        
        if label == 0.0:
            while True:
                sampled_length_labels = train_df[train_df["PE"] == 1.0].sample(n=1).iloc[0].pe_present_on_image
                indices = np.linspace(start=0, stop=len(sampled_length_labels) - 1, num=len(embeddings)).round().astype(int)
                length_labels = [sampled_length_labels[i] for i in indices]
                if sum(length_labels) != 0:
                    break            
            embeddings = torch.sum((torch.tensor(length_labels).view(-1, 1) / sum(length_labels)) * embeddings, dim=0, keepdim=True)
            length = 1
        else:
            embeddings = torch.sum((torch.tensor(length_labels).view(-1, 1) / sum(length_labels)) * embeddings, dim=0, keepdim=True)
            length = 1

        X.append(embeddings)
        lengths.append(length)
        y.append(label)
        
#         if label == 0.0:
#             for _ in range(length):
#                 lengths.append(1)
#                 y.append(label)
#         else:
#             for length_label in length_labels:
#                 lengths.append(1)
#                 y.append(torch.tensor([length_label], dtype=torch.float32))            
        
        lengths_y.append(length_labels)
        
    torch.save({
        "X": torch.cat(X),
        "lengths": tuple(lengths),
        "y": torch.stack(y),
        "lengths_y": lengths_y,
    }, f"{args.encoded_dir}/train.pt")
        
    #X, lengths, y = [], [], []
    X, lengths, y, lengths_y = [], [], [], []
    
    for image, length, label in val_dataset:
        
        label, length_labels = label
        
        embeddings = torch.cat([
            utils.encode_image(model, image[:,c].unsqueeze(1)) 
            for c in range(image.shape[1])
        ], dim=-1)
        #embeddings = torch.cat([
        #    torch.cat([
        #        utils.encode_image(model, image[s:s+5,c].unsqueeze(1))
        #        for s in range(0, image.shape[0], 5)
        #    ], dim=0)
        #    for c in range(image.shape[1])
        #], dim=-1)
        
        if label == 0.0:
            while True:
                sampled_length_labels = val_df[val_df["PE"] == 1.0].sample(n=1).iloc[0].pe_present_on_image
                indices = np.linspace(start=0, stop=len(sampled_length_labels) - 1, num=len(embeddings)).round().astype(int)
                length_labels = [sampled_length_labels[i] for i in indices]
                if sum(length_labels) != 0:
                    break            
            embeddings = torch.sum((torch.tensor(length_labels).view(-1, 1) / sum(length_labels)) * embeddings, dim=0, keepdim=True)
            length = 1
        else:
            embeddings = torch.sum((torch.tensor(length_labels).view(-1, 1) / sum(length_labels)) * embeddings, dim=0, keepdim=True)
            length = 1
        
        X.append(embeddings)
        lengths.append(length)
        y.append(label)
        
#         if label == 0.0:
#             for _ in range(length):
#                 lengths.append(1)
#                 y.append(label)
#         else:
#             for length_label in length_labels:
#                 lengths.append(1)
#                 y.append(torch.tensor([length_label], dtype=torch.float32))            
        
        lengths_y.append(length_labels)
        
    torch.save({
        "X": torch.cat(X),
        "lengths": tuple(lengths),
        "y": torch.stack(y),
        "lengths_y": lengths_y,
    }, f"{args.encoded_dir}/val.pt")
    
    #X, lengths, y = [], [], []
    X, lengths, y, lengths_y = [], [], [], []
    
    for image, length, label in test_dataset:
        
        label, length_labels = label
        
        embeddings = torch.cat([
            utils.encode_image(model, image[:,c].unsqueeze(1)) 
            for c in range(image.shape[1])
        ], dim=-1)
        #embeddings = torch.cat([
        #    torch.cat([
        #        utils.encode_image(model, image[s:s+5,c].unsqueeze(1))
        #        for s in range(0, image.shape[0], 5)
        #    ], dim=0)
        #    for c in range(image.shape[1])
        #], dim=-1)
        
        if label == 0.0:
            while True:
                sampled_length_labels = test_df[test_df["PE"] == 1.0].sample(n=1).iloc[0].pe_present_on_image
                indices = np.linspace(start=0, stop=len(sampled_length_labels) - 1, num=len(embeddings)).round().astype(int)
                length_labels = [sampled_length_labels[i] for i in indices]
                if sum(length_labels) != 0:
                    break            
            embeddings = torch.sum((torch.tensor(length_labels).view(-1, 1) / sum(length_labels)) * embeddings, dim=0, keepdim=True)
            length = 1
        else:
            embeddings = torch.sum((torch.tensor(length_labels).view(-1, 1) / sum(length_labels)) * embeddings, dim=0, keepdim=True)
            length = 1
            
        X.append(embeddings)
        lengths.append(length)
        y.append(label)
        
#         if label == 0.0:
#             for _ in range(length):
#                 lengths.append(1)
#                 y.append(label)
#         else:
#             for length_label in length_labels:
#                 lengths.append(1)
#                 y.append(torch.tensor([length_label], dtype=torch.float32))            
        
        lengths_y.append(length_labels)
        
    torch.save({
        "X": torch.cat(X),
        "lengths": tuple(lengths),
        "y": torch.stack(y),
        "lengths_y": lengths_y,
    }, f"{args.encoded_dir}/test.pt")
    