import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
# Importing our custom module(s)
import datasets
import utils

# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ViT_B_16/test_site_ids=9_train_site_ids=1_2_3_4_6_7_8_10_11_val_site_ids=5' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 9 --train_site_ids 1 2 3 4 6 7 8 10 11 --val_site_ids 5
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ViT_B_16/test_site_ids=1_4_7_10_11_train_site_ids=2_3_5_6_8_val_site_ids=9' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 1 4 7 10 11 --train_site_ids 2 3 5 6 8 --val_site_ids 9
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ViT_B_16/test_site_ids=2_6_train_site_ids=3_5_8_9_val_site_ids=1_4_7_10_11' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 2 6 --train_site_ids 3 5 8 9 --val_site_ids 1 4 7 10 11
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ViT_B_16/test_site_ids=3_8_train_site_ids=1_4_5_7_9_10_11_val_site_ids=2_6' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 3 8 --train_site_ids 1 4 5 7 9 10 11 --val_site_ids 2 6
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ViT_B_16/test_site_ids=5_train_site_ids=1_2_4_6_7_9_10_11_val_site_ids=3_8' --encoder='ViT-B/16' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 5 --train_site_ids 1 2 4 6 7 9 10 11 --val_site_ids 3 8

# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ConvNeXt_Tiny/test_site_ids=9_train_site_ids=1_2_3_4_6_7_8_10_11_val_site_ids=5' --encoder='ConvNeXt-Tiny' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 9 --train_site_ids 1 2 3 4 6 7 8 10 11 --val_site_ids 5
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ConvNeXt_Tiny/test_site_ids=1_4_7_10_11_train_site_ids=2_3_5_6_8_val_site_ids=9' --encoder='ConvNeXt-Tiny' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 1 4 7 10 11 --train_site_ids 2 3 5 6 8 --val_site_ids 9
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ConvNeXt_Tiny/test_site_ids=2_6_train_site_ids=3_5_8_9_val_site_ids=1_4_7_10_11' --encoder='ConvNeXt-Tiny' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 2 6 --train_site_ids 3 5 8 9 --val_site_ids 1 4 7 10 11
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ConvNeXt_Tiny/test_site_ids=3_8_train_site_ids=1_4_5_7_9_10_11_val_site_ids=2_6' --encoder='ConvNeXt-Tiny' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 3 8 --train_site_ids 1 4 5 7 9 10 11 --val_site_ids 2 6
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/ConvNeXt_Tiny/test_site_ids=5_train_site_ids=1_2_4_6_7_9_10_11_val_site_ids=3_8' --encoder='ConvNeXt-Tiny' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 5 --train_site_ids 1 2 4 6 7 9 10 11 --val_site_ids 3 8

# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=9_train_site_ids=1_2_3_4_6_7_8_10_11_val_site_ids=5' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 9 --train_site_ids 1 2 3 4 6 7 8 10 11 --val_site_ids 5
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=1_4_7_10_11_train_site_ids=2_3_5_6_8_val_site_ids=9' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 1 4 7 10 11 --train_site_ids 2 3 5 6 8 --val_site_ids 9
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=2_6_train_site_ids=3_5_8_9_val_site_ids=1_4_7_10_11' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 2 6 --train_site_ids 3 5 8 9 --val_site_ids 1 4 7 10 11
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=3_8_train_site_ids=1_4_5_7_9_10_11_val_site_ids=2_6' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 3 8 --train_site_ids 1 4 5 7 9 10 11 --val_site_ids 2 6
# python ../src/encode_kpsc.py --encoded_dir='/cluster/tufts/hugheslabkp/data_irb_required/encoded_KPSC_MRI_800/MedSAM/test_site_ids=5_train_site_ids=1_2_4_6_7_9_10_11_val_site_ids=3_8' --encoder='MedSAM' --numpy_dir='/cluster/tufts/hugheslabkp/data_irb_required/KPSC_MRI_800_numpy' --test_site_ids 5 --train_site_ids 1 2 4 6 7 9 10 11 --val_site_ids 3 8

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main.py')
    parser.add_argument('--encoded_dir', help='Directory to save encoded dataset', type=str)
    parser.add_argument('--encoder', help='Encoder pre-trained on ImageNet', type=str)
    parser.add_argument('--numpy_dir', help='Directory to numpy dataset', type=str)
    parser.add_argument('--test_site_ids', default=[9], help='Test splits (default: [9])', nargs='+', type=int)
    parser.add_argument('--train_site_ids', default=[1, 2, 3, 4, 6, 7, 8, 10, 11], help='Train splits (default: [1, 2, 3, 4, 6, 7, 8, 10, 11])', nargs='+', type=int)
    parser.add_argument('--val_site_ids', default=[5], help='Test splits (default: [5])', nargs='+', type=int)
    args = parser.parse_args()
    
    os.makedirs(args.encoded_dir, exist_ok=True)

    labels_df = pd.read_csv(f'{args.numpy_dir}/labels.csv')
    
    train_df = labels_df[labels_df['SiteID'].isin(args.train_site_ids)]
    val_df = labels_df[labels_df['SiteID'].isin(args.val_site_ids)]
    test_df = labels_df[labels_df['SiteID'].isin(args.test_site_ids)]
    
    # TODO: Make resize size an argument
    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        lambda image: image.permute(3, 0, 1, 2),
        lambda image: utils.pad_image(image),
        torchvision.transforms.Resize(size=(1024, 1024)),
        lambda image: torch.rot90(image, k=1, dims=[-2, -1]),
    ])
    
    train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[['idCBI', 'idWMD']].values, dtype=torch.float32), transform)

    means, stds = [], []

    for image, length, label in train_dataset:
        means.append(torch.mean(image, dim=(0, 2, 3)).tolist())
        stds.append(torch.std(image, dim=(0, 2, 3)).tolist())

    mean = torch.tensor(means).mean(dim=0)
    std = torch.tensor(stds).mean(dim=0)

    transform = torchvision.transforms.Compose([
        lambda path: utils.read_npz(path),
        lambda image: image.permute(3, 0, 1, 2),
        lambda image: utils.pad_image(image),
        lambda image: torch.rot90(image, k=1, dims=[-2, -1]),
        torchvision.transforms.Resize(size=(1024, 1024)),
        lambda image: (image - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1),
    ])

    train_dataset = datasets.MILPathDataset(train_df.path.values, torch.tensor(train_df[['idCBI', 'idWMD']].values, dtype=torch.float32), transform)
    val_dataset = datasets.MILPathDataset(val_df.path.values, torch.tensor(val_df[['idCBI', 'idWMD']].values, dtype=torch.float32), transform)
    test_dataset = datasets.MILPathDataset(test_df.path.values, torch.tensor(test_df[['idCBI', 'idWMD']].values, dtype=torch.float32), transform)
    
    assert args.encoder in ['ViT-B/16', 'ConvNeXt-Tiny', 'MedSAM']
    if args.encoder == 'ViT-B/16':
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights(weights))
        model.conv_proj.weight.data = model.conv_proj.weight.data.sum(dim=1, keepdim=True)
        model.conv_proj.in_channels = 1
        model.heads = torch.nn.Identity()
    elif args.encoder == 'ConvNeXt-Tiny':
        weights = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights(weights))
        model.features[0][0].weight.data = model.features[0][0].weight.data.sum(dim=1, keepdim=True)
        model.features[0][0].in_channels = 1
        model.classifier[2] = torch.nn.Identity()
    elif args.encoder == 'MedSAM':
        from segment_anything import sam_model_registry
        checkpoint_path = '/cluster/tufts/hugheslab/eharve06/pooling/models/medsam_vit_b.pth'
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
        medsam = sam_model_registry['vit_b']()
        medsam.load_state_dict(checkpoint)
        model = medsam.image_encoder
        model.patch_embed.proj.weight.data = model.patch_embed.proj.weight.data.sum(dim=1, keepdim=True)
        model.patch_embed.proj.in_channels = 1
        model.neck.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    
    X, lengths, y = [], [], []
    
    for image, length, label in train_dataset:
        
        embeddings = torch.cat([
            utils.encode_image(model, image[:,c].unsqueeze(1)) 
            for c in range(image.shape[1])
        ], dim=-1)
        
        X.append(embeddings)
        lengths.append(length)
        y.append(label)
        
    torch.save({
        'X': torch.cat(X),
        'lengths': tuple(lengths),
        'y': torch.stack(y),
    }, f'{args.encoded_dir}/train.pth')
        
    X, lengths, y = [], [], []
    
    for image, length, label in val_dataset:
        
        embeddings = torch.cat([
            utils.encode_image(model, image[:,c].unsqueeze(1)) 
            for c in range(image.shape[1])
        ], dim=-1)
        
        X.append(embeddings)
        lengths.append(length)
        y.append(label)
        
    torch.save({
        'X': torch.cat(X),
        'lengths': tuple(lengths),
        'y': torch.stack(y),
    }, f'{args.encoded_dir}/val.pth')
    
    X, lengths, y = [], [], []
    
    for image, length, label in test_dataset:
        
        embeddings = torch.cat([
            utils.encode_image(model, image[:,c].unsqueeze(1)) 
            for c in range(image.shape[1])
        ], dim=-1)
        
        X.append(embeddings)
        lengths.append(length)
        y.append(label)
        
    torch.save({
        'X': torch.cat(X),
        'lengths': tuple(lengths),
        'y': torch.stack(y),
    }, f'{args.encoded_dir}/test.pth')
    