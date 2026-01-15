#!/usr/bin/env python3
"""
最简单的重建预测脚本
直接加载模型和数据，不使用nnUNet的预测流程
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import argparse


def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None,
                 return_slicer=False, shape_must_be_divisible_by=None):
    """
    Pad image to target shape or ensure divisibility

    参数:
        image: numpy array to pad
        new_shape: target shape (if None, use shape_must_be_divisible_by)
        mode: padding mode
        kwargs: additional kwargs for np.pad
        return_slicer: if True, also return slicer to remove padding
        shape_must_be_divisible_by: ensure shape is divisible by this value
    """
    if kwargs is None:
        kwargs = {'constant_values': 0}

    old_shape = np.array(image.shape)

    if new_shape is not None:
        new_shape = np.array(new_shape)
    elif shape_must_be_divisible_by is not None:
        # Calculate new shape that is divisible
        new_shape = old_shape + (shape_must_be_divisible_by - old_shape % shape_must_be_divisible_by) % shape_must_be_divisible_by
    else:
        return image if not return_slicer else (image, tuple([slice(None)] * len(old_shape)))

    # Calculate padding amounts
    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference - pad_below

    # Create padding list
    pad_list = [[int(pad_below[i]), int(pad_above[i])] for i in range(len(old_shape))]

    # Pad image
    res = np.pad(image, pad_list, mode, **kwargs)

    if return_slicer:
        # Create slicer to remove padding later
        slicer = tuple([slice(int(pad_below[i]), int(pad_below[i] + old_shape[i]))
                       for i in range(len(old_shape))])
        return res, slicer
    else:
        return res


def simple_predict(
    checkpoint_path: str,
    input_folder: str,
    output_folder: str,
    device: str = 'cuda',
    use_preprocessed: bool = True
):
    """
    最简单的预测方法

    参数:
        checkpoint_path: checkpoint路径
        input_folder: 输入文件夹（.npz预处理文件或.nii.gz原始文件）
        output_folder: 输出文件夹
        device: 'cuda' or 'cpu'
        use_preprocessed: 是否使用预处理后的.npz文件
    """
    print("=" * 80)
    print("Simple Reconstruction Prediction")
    print("=" * 80)

    # 加载checkpoint
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_dir():
        # 查找checkpoint文件
        if (checkpoint_file / "checkpoint_best.pth").exists():
            checkpoint_file = checkpoint_file / "checkpoint_best.pth"
        elif (checkpoint_file / "checkpoint_final.pth").exists():
            checkpoint_file / "checkpoint_final.pth"
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

    print(f"\nLoading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    # 获取配置
    plans = checkpoint['init_args']['plans']
    configuration = checkpoint['init_args']['configuration']
    dataset_json = checkpoint['init_args']['dataset_json']

    print(f"Configuration: {configuration}")

    # 导入并实例化trainer
    from umamba.nnunetv2.training.nnUNetTrainer.nnUNetTrainerBrainEncoderReconstruction import nnUNetTrainerBrainEncoderReconstruction

    trainer = nnUNetTrainerBrainEncoderReconstruction(
        plans=plans,
        configuration=configuration,
        fold=0,
        dataset_json=dataset_json,
        unpack_dataset=False,
        device=torch.device(device)
    )

    # 初始化网络
    trainer.initialize()

    # 加载权重
    trainer.network.load_state_dict(checkpoint['network_weights'])
    trainer.network.eval()

    print(f"✓ Model loaded on {device}")

    # 获取预处理信息
    preprocessing_info = None
    if 'dataset_properties' in checkpoint['init_args']:
        preprocessing_info = checkpoint['init_args']['dataset_properties']

    # 创建输出目录
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 查找输入文件
    input_path = Path(input_folder)

    if use_preprocessed:
        # 使用预处理后的.npz文件
        input_files = sorted(list(input_path.glob("*.npz")))
        if len(input_files) == 0:
            input_files = sorted(list(input_path.glob("**/*.npz")))
        print(f"\nFound {len(input_files)} preprocessed files (.npz)")
    else:
        # 使用原始.nii.gz文件
        input_files = sorted(list(input_path.glob("*_0000.nii.gz")))
        print(f"\nFound {len(input_files)} raw files (.nii.gz)")

    if len(input_files) == 0:
        raise FileNotFoundError(f"No input files found in {input_folder}")

    # 获取网络的patch size（用于padding）
    # nnUNet的网络通常要求输入能被16或32整除
    divisible_by = 16  # U-Net with 4 pooling layers: 2^4 = 16

    # 预测
    print("\nProcessing...")
    print(f"Note: Padding inputs to be divisible by {divisible_by}")

    successful = 0
    failed = 0

    with torch.no_grad():
        for input_file in tqdm(input_files, desc="Predicting"):
            try:
                if use_preprocessed:
                    # 从.npz加载
                    data_dict = np.load(input_file, allow_pickle=True)
                    data = data_dict['data']  # 可能是 [C, D, H, W] 或 [D, H, W]
                    case_id = input_file.stem

                    # 检查并确保有channel维度
                    # nnUNet预处理后的数据：单模态通常是 [D, H, W]，多模态是 [C, D, H, W]
                    if len(data.shape) == 3:
                        # 3D数据无channel维度，添加channel维度
                        data = data[np.newaxis, ...]  # [1, D, H, W]
                    elif len(data.shape) == 4:
                        # 4D数据，第一维应该是channel
                        # 但需要检查channel数是否合理（通常是1）
                        if data.shape[0] > 10:
                            # 第一维很大，可能是空间维度，不是channel
                            # 假设是 [D, H, W, C] 格式，转换为 [C, D, H, W]
                            data = np.transpose(data, (3, 0, 1, 2))
                            # 如果还是只有1个channel，去掉最后的维度
                            if data.shape[0] == 1:
                                data = data[0]  # [D, H, W]
                                data = data[np.newaxis, ...]  # [1, D, H, W]

                else:
                    # 从.nii.gz加载
                    nii = nib.load(str(input_file))
                    data = nii.get_fdata()  # [H, W, D]

                    # 添加channel维度
                    data = data[np.newaxis, ...]  # [1, H, W, D]

                    # 标准化（简单方式）
                    data = (data - data.mean()) / (data.std() + 1e-8)

                    case_id = input_file.stem.replace('_0000', '')

                # 确保数据是 [C, ...] 格式，C应该是1
                if data.shape[0] != 1:
                    print(f"\n⚠ Warning: Unexpected channel count {data.shape[0]} for {case_id}")
                    print(f"  Data shape: {data.shape}")
                    # 如果channel数异常，可能是维度顺序问题
                    # 强制reshape为单channel
                    if len(data.shape) == 4:
                        # 假设是 [D, H, W, C] 或其他格式，强制为 [1, D, H, W]
                        # 找最小的维度作为channel
                        min_dim = np.argmin(data.shape)
                        if data.shape[min_dim] == 1:
                            # 移动这个维度到第一位
                            axes = list(range(len(data.shape)))
                            axes.insert(0, axes.pop(min_dim))
                            data = np.transpose(data, axes)
                            print(f"  Reshaped to: {data.shape}")

                # 最后确认：必须是 [1, D, H, W] 格式
                assert data.shape[0] == 1, f"Channel dimension must be 1, got {data.shape[0]}"

                # Pad data to ensure divisibility
                # data shape: [1, D, H, W]
                original_shape = data.shape

                # Pad only spatial dimensions (skip channel dimension)
                # Calculate target shape for spatial dimensions only (keep channel as-is)
                spatial_shape = np.array(data.shape[1:])  # [D, H, W]
                target_spatial_shape = spatial_shape + (divisible_by - spatial_shape % divisible_by) % divisible_by

                # Pad spatial dimensions only
                pad_amounts = target_spatial_shape - spatial_shape
                pad_below = pad_amounts // 2
                pad_above = pad_amounts - pad_below

                # Create padding list: [(0,0) for channel, then actual padding for spatial dims]
                pad_list = [(0, 0)]  # No padding for channel dimension
                for i in range(len(spatial_shape)):
                    pad_list.append((int(pad_below[i]), int(pad_above[i])))

                # Apply padding
                data_padded = np.pad(data, pad_list, mode='constant', constant_values=0)

                # Create slicer to remove padding later (keep all channels, slice spatial dims)
                slicer = [slice(None)]  # Keep all channels
                for i in range(len(spatial_shape)):
                    slicer.append(slice(int(pad_below[i]), int(pad_below[i] + spatial_shape[i])))
                slicer = tuple(slicer)

                # 转为tensor
                data_tensor = torch.from_numpy(data_padded).float()

                # 添加batch维度 [1, C, D, H, W]
                if len(data_tensor.shape) == 4:  # [C, D, H, W]
                    data_tensor = data_tensor.unsqueeze(0)  # [1, C, D, H, W]

                data_tensor = data_tensor.to(device)

                # 预测
                output = trainer.network(data_tensor)  # [1, C_out, D, H, W]

                # 后处理输出
                output = output.squeeze(0).cpu().numpy()  # [C_out, D, H, W]

                # Remove padding
                output = output[slicer]

                # 如果是多通道输出，取平均
                if output.shape[0] > 1:
                    output = output.mean(axis=0)  # [D, H, W]
                else:
                    output = output[0]  # [D, H, W]

                # 验证输出形状与原始输入匹配
                if output.shape != original_shape[1:]:
                    print(f"\n⚠ Warning: Output shape {output.shape} != expected {original_shape[1:]}")
                    print(f"  Reshaping output...")

                # 保存
                output_file = output_path / f"{case_id}.nii.gz"
                nii_img = nib.Nifti1Image(output, affine=np.eye(4))
                nib.save(nii_img, str(output_file))

                successful += 1

            except Exception as e:
                print(f"\n✗ Error processing {input_file.name}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                continue

    print("\n" + "=" * 80)
    print("✓ Prediction completed!")
    print(f"  Successful: {successful}/{len(input_files)}")
    print(f"  Failed: {failed}/{len(input_files)}")
    print(f"✓ Results saved to: {output_folder}")
    print("=" * 80)

    if use_preprocessed:
        print("\n⚠ Note: Results are in preprocessed space (same as training).")
        print("For accurate evaluation, use preprocessed input images too.")
    else:
        print("\n⚠ Note: Using raw images may have preprocessing mismatch.")
        print("For best results, use preprocessed .npz files.")

    if failed > 0:
        print(f"\n⚠ Warning: {failed} files failed to process.")
        print("  Check the error messages above for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple reconstruction prediction")

    parser.add_argument("--checkpoint", "-chk", type=str, required=True,
                       help="Path to checkpoint file or folder")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input folder")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output folder")
    parser.add_argument("--device", type=str, default='cuda',
                       help="Device: cuda or cpu")
    parser.add_argument("--use_preprocessed", action='store_true',
                       help="Use preprocessed .npz files instead of raw .nii.gz")

    args = parser.parse_args()

    simple_predict(
        checkpoint_path=args.checkpoint,
        input_folder=args.input,
        output_folder=args.output,
        device=args.device,
        use_preprocessed=args.use_preprocessed
    )
