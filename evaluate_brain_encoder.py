#!/usr/bin/env python3
"""
评估脑MRI重建模型的性能

功能：
1. 计算定量指标（PSNR, MAE, SSIM, MSE）
2. 可视化重建结果（原图 vs 重建图）
3. 分析encoder特征质量
4. 生成详细的评估报告
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import nibabel as nib
from tqdm import tqdm
import argparse
import json
from typing import Dict, List, Tuple


def load_model_and_checkpoint(checkpoint_path: str, device: torch.device):
    """加载训练好的模型"""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 从checkpoint中获取网络架构和参数
    network = checkpoint['network_weights']

    return checkpoint, network


def calculate_psnr(img1: np.ndarray, img2: np.ndarray, data_range: float = None) -> float:
    """计算PSNR"""
    if data_range is None:
        data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')

    return 20 * np.log10(data_range / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, data_range: float = None) -> float:
    """计算SSIM（3D版本）"""
    if data_range is None:
        data_range = max(img1.max(), img2.max()) - min(img1.min(), img2.min())

    return ssim(img1, img2, data_range=data_range)


def calculate_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算MAE"""
    return np.mean(np.abs(img1 - img2))


def calculate_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """计算MSE"""
    return np.mean((img1 - img2) ** 2)


def load_nifti(file_path: str) -> np.ndarray:
    """加载NIfTI文件"""
    nii = nib.load(file_path)
    return nii.get_fdata()


def evaluate_reconstruction(
    model_checkpoint: str,
    test_images_dir: str,
    output_dir: str,
    num_samples: int = 20,
    device_id: int = 0
):
    """
    评估重建性能

    参数:
        model_checkpoint: 模型checkpoint路径（如checkpoint_best.pth）
        test_images_dir: 测试图像目录（nnUNet格式的imagesTs或imagesTr）
        output_dir: 输出目录
        num_samples: 可视化的样本数量
        device_id: GPU设备ID
    """
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vis_dir = output_path / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # 加载模型
    print("\n" + "="*60)
    print("Step 1: Loading model...")
    print("="*60)

    try:
        checkpoint = torch.load(model_checkpoint, map_location=device)
        print(f"✓ Checkpoint loaded successfully")
        print(f"  - Epoch: {checkpoint.get('current_epoch', 'N/A')}")

        # 获取trainer以便使用其推理功能
        # 注意：这里需要根据实际情况调整
        from umamba.nnunetv2.training.nnUNetTrainer.nnUNetTrainerBrainEncoderReconstruction import nnUNetTrainerBrainEncoderReconstruction

        # 尝试从checkpoint恢复trainer
        plans = checkpoint['init_args']['plans']
        configuration = checkpoint['init_args']['configuration']

        print(f"  - Configuration: {configuration}")

    except Exception as e:
        print(f"✗ Error loading checkpoint: {e}")
        print("\n提示：如果遇到导入错误，可以使用简化版本的评估（直接加载预处理后的数据）")
        return

    # 查找测试图像
    print("\n" + "="*60)
    print("Step 2: Finding test images...")
    print("="*60)

    test_images_path = Path(test_images_dir)
    image_files = sorted(list(test_images_path.glob("*_0000.nii.gz")))

    if len(image_files) == 0:
        print(f"✗ No images found in {test_images_dir}")
        print("  Expected format: *_0000.nii.gz")
        return

    print(f"✓ Found {len(image_files)} test images")

    # 评估指标
    print("\n" + "="*60)
    print("Step 3: Evaluating reconstruction...")
    print("="*60)

    metrics = {
        'psnr': [],
        'mae': [],
        'mse': [],
        'ssim': []
    }

    # 随机选择样本进行可视化
    vis_indices = np.random.choice(len(image_files), min(num_samples, len(image_files)), replace=False)

    for idx, img_file in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            # 加载原始图像
            original = load_nifti(str(img_file))

            # 这里需要模型推理
            # 由于直接运行模型比较复杂，我们提供一个简化的方案：
            # 用户可以先用nnUNet的预测命令生成重建结果，然后我们评估

            # 如果有对应的预测结果，加载它
            pred_file = output_path / "predictions" / img_file.name.replace("_0000.nii.gz", ".nii.gz")

            if pred_file.exists():
                reconstructed = load_nifti(str(pred_file))

                # 计算指标
                psnr = calculate_psnr(original, reconstructed)
                mae = calculate_mae(original, reconstructed)
                mse = calculate_mse(original, reconstructed)
                ssim_val = calculate_ssim(original, reconstructed)

                metrics['psnr'].append(psnr)
                metrics['mae'].append(mae)
                metrics['mse'].append(mse)
                metrics['ssim'].append(ssim_val)

                # 可视化选中的样本
                if idx in vis_indices:
                    visualize_reconstruction(
                        original, reconstructed,
                        vis_dir / f"sample_{idx:03d}.png",
                        case_name=img_file.stem
                    )

        except Exception as e:
            print(f"  ✗ Error processing {img_file.name}: {e}")
            continue

    # 计算统计信息
    print("\n" + "="*60)
    print("Step 4: Computing statistics...")
    print("="*60)

    if len(metrics['psnr']) > 0:
        stats = {}
        for metric_name, values in metrics.items():
            stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

        # 打印结果
        print("\n📊 Reconstruction Metrics:")
        print("-" * 60)
        for metric_name, stat in stats.items():
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean ± Std:  {stat['mean']:.4f} ± {stat['std']:.4f}")
            print(f"  Median:      {stat['median']:.4f}")
            print(f"  Range:       [{stat['min']:.4f}, {stat['max']:.4f}]")

        # 保存结果
        results = {
            'checkpoint': str(model_checkpoint),
            'num_samples': len(metrics['psnr']),
            'metrics': stats
        }

        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")

        # 绘制指标分布图
        plot_metrics_distribution(metrics, output_path / "metrics_distribution.png")
        print(f"✓ Metrics distribution plot saved")

    else:
        print("\n⚠ No predictions found!")
        print(f"Please run prediction first using:")
        print(f"  nnUNetv2_predict -i {test_images_dir} \\")
        print(f"    -o {output_path / 'predictions'} \\")
        print(f"    -d 800 -c 3d_fullres \\")
        print(f"    -tr nnUNetTrainerBrainEncoderReconstruction \\")
        print(f"    -chk {model_checkpoint} \\")
        print(f"    -f all")

    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


def visualize_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: Path,
    case_name: str = ""
):
    """可视化重建结果（显示中间切片）"""

    # 选择3个正交平面的中间切片
    d, h, w = original.shape
    slices = [
        (original[d//2, :, :], reconstructed[d//2, :, :], "Axial"),
        (original[:, h//2, :], reconstructed[:, h//2, :], "Coronal"),
        (original[:, :, w//2], reconstructed[:, :, w//2], "Sagittal")
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    for i, (orig_slice, recon_slice, plane) in enumerate(slices):
        # 原始图像
        axes[i, 0].imshow(orig_slice, cmap='gray')
        axes[i, 0].set_title(f'Original ({plane})')
        axes[i, 0].axis('off')

        # 重建图像
        axes[i, 1].imshow(recon_slice, cmap='gray')
        axes[i, 1].set_title(f'Reconstructed ({plane})')
        axes[i, 1].axis('off')

        # 差异图
        diff = np.abs(orig_slice - recon_slice)
        im = axes[i, 2].imshow(diff, cmap='hot')
        axes[i, 2].set_title(f'Absolute Difference ({plane})')
        axes[i, 2].axis('off')
        plt.colorbar(im, ax=axes[i, 2], fraction=0.046)

    plt.suptitle(f'Reconstruction Quality: {case_name}', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_metrics_distribution(metrics: Dict[str, List[float]], save_path: Path):
    """绘制指标分布图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    metric_names = ['psnr', 'mae', 'mse', 'ssim']
    titles = ['PSNR (dB)', 'MAE', 'MSE', 'SSIM']

    for idx, (metric_name, title) in enumerate(zip(metric_names, titles)):
        if metric_name in metrics and len(metrics[metric_name]) > 0:
            values = metrics[metric_name]

            axes[idx].hist(values, bins=30, edgecolor='black', alpha=0.7)
            axes[idx].axvline(np.mean(values), color='r', linestyle='--',
                            label=f'Mean: {np.mean(values):.4f}')
            axes[idx].axvline(np.median(values), color='g', linestyle='--',
                            label=f'Median: {np.median(values):.4f}')
            axes[idx].set_xlabel(title)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{title} Distribution')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def simple_evaluation_from_predictions(
    original_images_dir: str,
    predicted_images_dir: str,
    output_dir: str,
    num_vis: int = 10
):
    """
    简化版评估：直接从预测结果文件夹评估

    适用场景：已经用nnUNetv2_predict生成了预测结果

    参数:
        original_images_dir: 原始图像目录
        predicted_images_dir: 预测/重建结果目录
        output_dir: 输出目录
        num_vis: 可视化样本数
    """
    print("="*60)
    print("Simple Evaluation from Predictions")
    print("="*60)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vis_dir = output_path / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # 查找图像对
    orig_path = Path(original_images_dir)
    pred_path = Path(predicted_images_dir)

    orig_files = sorted(list(orig_path.glob("*.nii.gz")))

    if len(orig_files) == 0:
        print(f"✗ No images found in {original_images_dir}")
        return

    print(f"Found {len(orig_files)} original images")

    metrics = {
        'psnr': [],
        'mae': [],
        'mse': [],
        'ssim': []
    }

    vis_indices = np.random.choice(len(orig_files), min(num_vis, len(orig_files)), replace=False)

    for idx, orig_file in enumerate(tqdm(orig_files, desc="Evaluating")):
        # 找到对应的预测文件
        # 移除_0000后缀
        pred_name = orig_file.name.replace("_0000.nii.gz", ".nii.gz")
        pred_file = pred_path / pred_name

        if not pred_file.exists():
            print(f"  ⚠ Prediction not found for {orig_file.name}")
            continue

        try:
            # 加载图像
            original = load_nifti(str(orig_file))
            reconstructed = load_nifti(str(pred_file))

            # 计算指标
            psnr = calculate_psnr(original, reconstructed)
            mae = calculate_mae(original, reconstructed)
            mse = calculate_mse(original, reconstructed)
            ssim_val = calculate_ssim(original, reconstructed)

            metrics['psnr'].append(psnr)
            metrics['mae'].append(mae)
            metrics['mse'].append(mse)
            metrics['ssim'].append(ssim_val)

            # 可视化
            if idx in vis_indices:
                visualize_reconstruction(
                    original, reconstructed,
                    vis_dir / f"sample_{idx:03d}.png",
                    case_name=orig_file.stem
                )

        except Exception as e:
            print(f"  ✗ Error processing {orig_file.name}: {e}")
            continue

    # 输出统计结果
    if len(metrics['psnr']) > 0:
        print("\n" + "="*60)
        print("📊 Evaluation Results")
        print("="*60)

        stats = {}
        for metric_name, values in metrics.items():
            stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

            print(f"\n{metric_name.upper()}:")
            print(f"  Mean ± Std:  {stats[metric_name]['mean']:.4f} ± {stats[metric_name]['std']:.4f}")
            print(f"  Median:      {stats[metric_name]['median']:.4f}")
            print(f"  Range:       [{stats[metric_name]['min']:.4f}, {stats[metric_name]['max']:.4f}]")

        # 保存结果
        results = {
            'num_samples': len(metrics['psnr']),
            'metrics': stats
        }

        results_file = output_path / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved to: {results_file}")

        # 绘图
        plot_metrics_distribution(metrics, output_path / "metrics_distribution.png")
        print(f"✓ Distribution plot: {output_path / 'metrics_distribution.png'}")
        print(f"✓ Visualizations: {vis_dir}")

        # 性能评估
        print("\n" + "="*60)
        print("🎯 Performance Assessment")
        print("="*60)

        psnr_mean = stats['psnr']['mean']
        ssim_mean = stats['ssim']['mean']
        mae_mean = stats['mae']['mean']

        print(f"\nReconstruction Quality:")
        if psnr_mean > 30:
            print(f"  ✓ PSNR > 30 dB: Excellent reconstruction quality")
        elif psnr_mean > 25:
            print(f"  ○ PSNR 25-30 dB: Good reconstruction quality")
        else:
            print(f"  ✗ PSNR < 25 dB: May need improvement")

        if ssim_mean > 0.9:
            print(f"  ✓ SSIM > 0.9: Excellent structural similarity")
        elif ssim_mean > 0.8:
            print(f"  ○ SSIM 0.8-0.9: Good structural similarity")
        else:
            print(f"  ✗ SSIM < 0.8: May need improvement")

        if mae_mean < 0.05:
            print(f"  ✓ MAE < 0.05: Very accurate reconstruction")
        elif mae_mean < 0.1:
            print(f"  ○ MAE 0.05-0.1: Acceptable reconstruction")
        else:
            print(f"  ✗ MAE > 0.1: May need improvement")

        print("\n" + "="*60)
        print("Evaluation completed!")
        print("="*60)
    else:
        print("\n✗ No valid image pairs found for evaluation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate brain MRI reconstruction model")
    parser.add_argument("--mode", type=str, choices=['simple', 'full'], default='simple',
                       help="Evaluation mode: 'simple' (from predictions) or 'full' (with model)")

    # Simple mode arguments
    parser.add_argument("--original_dir", type=str,
                       help="Directory containing original images")
    parser.add_argument("--predicted_dir", type=str,
                       help="Directory containing predicted/reconstructed images")

    # Full mode arguments
    parser.add_argument("--checkpoint", type=str,
                       help="Path to model checkpoint (for full mode)")
    parser.add_argument("--test_dir", type=str,
                       help="Directory containing test images (for full mode)")

    # Common arguments
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for evaluation results")
    parser.add_argument("--num_vis", type=int, default=10,
                       help="Number of samples to visualize")
    parser.add_argument("--device", type=int, default=0,
                       help="GPU device ID")

    args = parser.parse_args()

    if args.mode == 'simple':
        if not args.original_dir or not args.predicted_dir:
            print("Error: --original_dir and --predicted_dir are required for simple mode")
            exit(1)

        simple_evaluation_from_predictions(
            args.original_dir,
            args.predicted_dir,
            args.output_dir,
            args.num_vis
        )

    else:  # full mode
        if not args.checkpoint or not args.test_dir:
            print("Error: --checkpoint and --test_dir are required for full mode")
            exit(1)

        evaluate_reconstruction(
            args.checkpoint,
            args.test_dir,
            args.output_dir,
            args.num_vis,
            args.device
        )
