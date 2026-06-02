#!/usr/bin/env python3
"""
Find best hyperparameters based on validation AUROC and report test AUROC.

Usage:
    python find_best_results.py /path/to/experiments/folder

Example:
    python find_best_results.py /cluster/tufts/hugheslab/dloevl01/Dec_2025/pooling/experiments/oasis-3_ct/embedding
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np


def find_best_for_seed(folder_path, seed):
    """Find the best model (by val_auroc) for a given seed and return its test_auroc."""
    pattern = os.path.join(folder_path, f"*seed={seed}*.csv")
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"  Warning: No CSV files found for seed={seed}")
        return None, None, None

    best_val_auroc = -1
    best_test_auroc = None
    best_file = None
    best_epoch = None

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            if 'val_auroc' not in df.columns or 'test_auroc' not in df.columns:
                continue

            # Filter to only rows where val_auroc < train_auroc (not overfitting)
            if 'train_auroc' in df.columns:
                valid_df = df[df['val_auroc'] <= df['train_auroc']]
                if valid_df.empty:
                    continue
            else:
                valid_df = df

            # Find row with highest val_auroc among valid rows
            idx = valid_df['val_auroc'].idxmax()
            val_auroc = df.loc[idx, 'val_auroc']
            test_auroc = df.loc[idx, 'test_auroc']
            train_auroc = df.loc[idx, 'train_auroc'] if 'train_auroc' in df.columns else None
            epoch = df.loc[idx, 'epoch'] if 'epoch' in df.columns else idx

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_test_auroc = test_auroc
                best_file = os.path.basename(csv_file)
                best_epoch = epoch

        except Exception as e:
            print(f"  Error reading {csv_file}: {e}")
            continue

    return best_test_auroc, best_val_auroc, best_file, best_epoch


def main():
    parser = argparse.ArgumentParser(description='Find best results by validation AUROC')
    parser.add_argument('folder', type=str, help='Path to folder containing CSV result files')
    args = parser.parse_args()

    folder_path = args.folder

    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory")
        return

    seeds = [1001, 2001, 3001]
    results = []

    print(f"\nSearching in: {folder_path}\n")
    print("=" * 80)

    for seed in seeds:
        test_auroc, val_auroc, best_file, best_epoch = find_best_for_seed(folder_path, seed)

        if test_auroc is not None:
            results.append(test_auroc)
            print(f"Seed {seed}:")
            print(f"  Best file:    {best_file}")
            print(f"  Best epoch:   {best_epoch}")
            print(f"  Val AUROC:    {val_auroc:.4f}")
            print(f"  Test AUROC:   {test_auroc:.4f}")
            print()
        else:
            print(f"Seed {seed}: No valid results found\n")

    print("=" * 80)

    if results:
        mean_auroc = np.mean(results)
        std_auroc = np.std(results)
        print(f"\nSUMMARY ({len(results)} seeds):")
        print(f"  Test AUROCs:  {[f'{r:.4f}' for r in results]}")
        print(f"  Mean:         {mean_auroc:.4f}")
        print(f"  Std:          {std_auroc:.4f}")
        print(f"  Mean ± Std:   {mean_auroc:.4f} ± {std_auroc:.4f}")
    else:
        print("\nNo valid results found.")


if __name__ == "__main__":
    main()
