#!/usr/bin/env python3
"""
Test script to compare loop-based vs matrix-based ApproxSm implementations.
"""

import torch
import time
import sys
sys.path.insert(0, '.')
import layers


def test_correctness(neighbors_list=[1, 5, 10, 50, 100], seq_len=200, feature_dim=768):
    """Test that both implementations produce the same output."""
    print("=" * 60)
    print("CORRECTNESS TEST: Comparing loop vs matrix implementations")
    print("=" * 60)

    for neighbors in neighbors_list:
        # Create both versions
        sm_loop = layers.ApproxSm(alpha=0.5, neighbors=neighbors, num_steps=10, use_matrix=False)
        sm_matrix = layers.ApproxSm(alpha=0.5, neighbors=neighbors, num_steps=10, use_matrix=True)

        # Sync alpha parameters
        sm_matrix.raw_alpha = sm_loop.raw_alpha

        # Create test input
        f = torch.randn(seq_len, feature_dim)

        # Run both
        with torch.no_grad():
            out_loop = sm_loop(f)
            out_matrix = sm_matrix(f)

        # Compare
        max_diff = (out_loop - out_matrix).abs().max().item()
        mean_diff = (out_loop - out_matrix).abs().mean().item()

        status = "PASS" if max_diff < 1e-5 else "FAIL"
        print(f"neighbors={neighbors:3d}: max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e} [{status}]")


def test_speed(neighbors_list=[1, 5, 10, 50, 100, 150], seq_len=200, feature_dim=768, num_trials=10):
    """Test speed of both implementations."""
    print("\n" + "=" * 60)
    print("SPEED TEST: Comparing loop vs matrix implementations")
    print(f"(seq_len={seq_len}, feature_dim={feature_dim}, num_trials={num_trials})")
    print("=" * 60)

    for neighbors in neighbors_list:
        sm_loop = layers.ApproxSm(alpha=0.5, neighbors=neighbors, num_steps=10, use_matrix=False)
        sm_matrix = layers.ApproxSm(alpha=0.5, neighbors=neighbors, num_steps=10, use_matrix=True)

        f = torch.randn(seq_len, feature_dim)

        # Warmup
        with torch.no_grad():
            _ = sm_loop(f)
            _ = sm_matrix(f)

        # Time loop version
        start = time.time()
        with torch.no_grad():
            for _ in range(num_trials):
                _ = sm_loop(f)
        time_loop = (time.time() - start) / num_trials

        # Time matrix version
        start = time.time()
        with torch.no_grad():
            for _ in range(num_trials):
                _ = sm_matrix(f)
        time_matrix = (time.time() - start) / num_trials

        speedup = time_loop / time_matrix
        print(f"neighbors={neighbors:3d}: loop={time_loop*1000:.2f}ms, matrix={time_matrix*1000:.2f}ms, speedup={speedup:.2f}x")


if __name__ == "__main__":
    test_correctness()
    test_speed()
