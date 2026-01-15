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

    # 预测
    print("\nProcessing...")
    with torch.no_grad():
        for input_file in tqdm(input_files):
            try:
                if use_preprocessed:
                    # 从.npz加载
                    data_dict = np.load(input_file, allow_pickle=True)
                    data = data_dict['data']  # [C, D, H, W] 或 [C, H, W, D]
                    case_id = input_file.stem
                else:
                    # 从.nii.gz加载
                    nii = nib.load(str(input_file))
                    data = nii.get_fdata()  # [H, W, D]

                    # 添加channel维度
                    data = data[np.newaxis, ...]  # [1, H, W, D]

                    # 标准化（简单方式）
                    data = (data - data.mean()) / (data.std() + 1e-8)

                    case_id = input_file.stem.replace('_0000', '')

                # 转为tensor
                data_tensor = torch.from_numpy(data).float()

                # 确保维度正确 [1, C, D, H, W]
                if len(data_tensor.shape) == 4:  # [C, D, H, W]
                    data_tensor = data_tensor.unsqueeze(0)  # [1, C, D, H, W]

                data_tensor = data_tensor.to(device)

                # 预测
                output = trainer.network(data_tensor)  # [1, C_out, D, H, W]

                # 后处理输出
                output = output.squeeze(0).cpu().numpy()  # [C_out, D, H, W]

                # 如果是多通道输出，取平均
                if output.shape[0] > 1:
                    output = output.mean(axis=0)  # [D, H, W]
                else:
                    output = output[0]  # [D, H, W]

                # 保存
                output_file = output_path / f"{case_id}.nii.gz"
                nii_img = nib.Nifti1Image(output, affine=np.eye(4))
                nib.save(nii_img, str(output_file))

            except Exception as e:
                print(f"\n✗ Error processing {input_file.name}: {e}")
                continue

    print("\n" + "=" * 80)
    print("✓ Prediction completed!")
    print(f"✓ Results saved to: {output_folder}")
    print("=" * 80)

    if use_preprocessed:
        print("\n⚠ Note: Results are in preprocessed space (same as training).")
        print("For accurate evaluation, use preprocessed input images too.")
    else:
        print("\n⚠ Note: Using raw images may have preprocessing mismatch.")
        print("For best results, use preprocessed .npz files.")


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
