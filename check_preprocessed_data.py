#!/usr/bin/env python3
"""
检查预处理数据的维度格式
帮助诊断channel维度问题
"""

import numpy as np
from pathlib import Path
import argparse


def check_preprocessed_data(preprocessed_folder: str, num_samples: int = 5):
    """
    检查预处理数据的格式

    参数:
        preprocessed_folder: 预处理数据文件夹
        num_samples: 检查多少个样本
    """
    print("=" * 80)
    print("Checking Preprocessed Data Format")
    print("=" * 80)

    # 查找.npz文件
    folder = Path(preprocessed_folder)
    npz_files = sorted(list(folder.glob("*.npz")))

    if len(npz_files) == 0:
        npz_files = sorted(list(folder.glob("**/*.npz")))

    if len(npz_files) == 0:
        print(f"✗ No .npz files found in {preprocessed_folder}")
        return

    print(f"Found {len(npz_files)} .npz files")
    print(f"Checking first {min(num_samples, len(npz_files))} samples...\n")

    # 检查样本
    shapes = []
    for i, npz_file in enumerate(npz_files[:num_samples]):
        print(f"Sample {i+1}: {npz_file.name}")
        print("-" * 80)

        # 加载数据
        data_dict = np.load(npz_file, allow_pickle=True)

        # 显示包含的keys
        print(f"  Keys: {list(data_dict.keys())}")

        # 检查data
        if 'data' in data_dict.keys():
            data = data_dict['data']
            print(f"  Data shape: {data.shape}")
            print(f"  Data dtype: {data.dtype}")
            print(f"  Data range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  Data mean: {data.mean():.4f}")
            print(f"  Data std: {data.std():.4f}")

            # 分析维度
            if len(data.shape) == 3:
                print(f"  → 3D data (no channel dimension)")
                print(f"  → Likely format: [D, H, W]")
                print(f"  → Need to add channel dim: [1, D, H, W]")
            elif len(data.shape) == 4:
                print(f"  → 4D data")
                # 检查哪个维度最可能是channel
                dim_sizes = data.shape
                min_dim_idx = np.argmin(dim_sizes)
                min_dim_size = dim_sizes[min_dim_idx]

                if min_dim_idx == 0 and min_dim_size <= 3:
                    print(f"  → Likely format: [C, D, H, W] with C={min_dim_size}")
                    print(f"  → Channel dim: axis 0, size {min_dim_size}")
                elif min_dim_size == 1:
                    print(f"  → Has dimension of size 1 at axis {min_dim_idx}")
                    print(f"  → This might be the channel dimension")
                    print(f"  → May need to transpose to [C, D, H, W]")
                else:
                    print(f"  → Uncertain format")
                    print(f"  → Smallest dim: axis {min_dim_idx}, size {min_dim_size}")

                # 检查第一个维度
                if data.shape[0] > 10:
                    print(f"  ⚠ WARNING: First dimension is {data.shape[0]} (> 10)")
                    print(f"  ⚠ This is likely a SPATIAL dimension, not channel!")
                    print(f"  ⚠ Data format might be wrong")

            shapes.append(data.shape)

        # 检查properties
        if 'properties' in data_dict.keys():
            props = data_dict['properties']
            if isinstance(props, np.ndarray):
                props = props.item()  # Convert numpy scalar to dict
            if isinstance(props, dict):
                print(f"  Properties keys: {list(props.keys())}")
                if 'spacing' in props:
                    print(f"    Spacing: {props['spacing']}")

        print()

    # 总结
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    # 检查所有样本是否有相同的shape
    if len(set(shapes)) == 1:
        print(f"✓ All samples have the same shape: {shapes[0]}")
    else:
        print(f"⚠ Samples have different shapes:")
        for shape in set(shapes):
            count = shapes.count(shape)
            print(f"  {shape}: {count} samples")

    # 给出建议
    print("\nRecommendation:")
    print("-" * 80)

    typical_shape = shapes[0]
    if len(typical_shape) == 3:
        print("✓ Data is 3D (no channel dim)")
        print("✓ Script will add channel dimension automatically")
        print("✓ Expected to work correctly")
    elif len(typical_shape) == 4:
        if typical_shape[0] == 1:
            print("✓ Data is 4D with 1 channel at axis 0")
            print("✓ Format: [1, D, H, W]")
            print("✓ Expected to work correctly")
        elif typical_shape[0] <= 3:
            print("✓ Data is 4D with channel at axis 0")
            print(f"✓ Format: [{typical_shape[0]}, D, H, W]")
            print(f"⚠ Warning: {typical_shape[0]} channels detected")
            print("  For brain MRI, expected 1 channel")
        else:
            print("✗ Data format issue detected!")
            print(f"✗ First dimension is {typical_shape[0]}, too large for channel")
            print("✗ Likely format problem - dimension order might be wrong")
            print("\nPossible solutions:")
            print("1. Check nnUNet preprocessing settings")
            print("2. Script will attempt to fix automatically")
            print("3. If still fails, manual intervention needed")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check preprocessed data format")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Preprocessed data folder")
    parser.add_argument("--num_samples", "-n", type=int, default=5,
                       help="Number of samples to check")

    args = parser.parse_args()

    check_preprocessed_data(args.input, args.num_samples)
