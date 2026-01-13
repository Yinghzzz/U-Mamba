"""
Brain Encoder特征提取脚本
从训练好的U-Mamba模型中提取脑MRI的隐空间特征，用于跨模态生成任务
"""
import torch
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import argparse
import json
from typing import List, Tuple, Union
from torch import nn
import pickle


def load_trained_encoder(
    model_folder: str,
    checkpoint: str = 'checkpoint_final.pth',
    device: str = 'cuda'
):
    """
    加载训练好的encoder模型

    参数:
        model_folder: 模型文件夹路径（包含checkpoint和plans.json）
        checkpoint: checkpoint文件名
        device: 'cuda' 或 'cpu'

    返回:
        model, plans, preprocessing_config
    """
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from batchgenerators.utilities.file_and_folder_operations import load_json

    device = torch.device(device)

    # 加载plans
    plans_file = Path(model_folder) / 'plans.json'
    plans = load_json(str(plans_file))

    # 加载checkpoint
    checkpoint_path = Path(model_folder) / checkpoint
    checkpoint_data = torch.load(str(checkpoint_path), map_location=device)

    # 提取网络权重
    if 'network_weights' in checkpoint_data:
        state_dict = checkpoint_data['network_weights']
    else:
        state_dict = checkpoint_data

    # 重建网络架构
    # 注意：需要根据plans重建相同的网络结构
    from nnunetv2.nets.UMambaEnc_3d import get_umamba_enc_3d_from_plans
    from nnunetv2.nets.UMambaEnc_2d import get_umamba_enc_2d_from_plans

    plans_manager = PlansManager(plans)
    configuration = 'train_config' if 'train_config' in plans['configurations'] else list(plans['configurations'].keys())[0]
    configuration_manager = plans_manager.get_configuration(configuration)

    # 加载dataset_json
    dataset_json_path = Path(model_folder).parent.parent / 'dataset.json'
    dataset_json = load_json(str(dataset_json_path))

    # 构建网络
    num_input_channels = len(plans['configurations'][configuration]['modality'])

    if len(configuration_manager.patch_size) == 3:
        model = get_umamba_enc_3d_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=False  # 特征提取时不需要deep supervision
        )
    else:
        model = get_umamba_enc_2d_from_plans(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            deep_supervision=False
        )

    # 加载权重
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"✓ 模型加载成功: {checkpoint_path}")
    print(f"  配置: {configuration}")
    print(f"  Patch size: {configuration_manager.patch_size}")
    print(f"  输入通道数: {num_input_channels}")

    return model, plans, configuration_manager


def preprocess_image(
    image_path: str,
    target_spacing: Tuple[float, ...] = None,
    intensity_properties: dict = None
) -> np.ndarray:
    """
    预处理输入图像

    参数:
        image_path: 图像路径
        target_spacing: 目标spacing（如果需要重采样）
        intensity_properties: 强度归一化参数

    返回:
        预处理后的numpy数组
    """
    # 读取图像
    img = sitk.ReadImage(image_path)
    img_array = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()

    print(f"原始图像形状: {img_array.shape}")
    print(f"原始spacing: {spacing}")

    # 强度归一化
    # 使用简单的z-score归一化
    if intensity_properties:
        mean = intensity_properties.get('mean', np.mean(img_array))
        std = intensity_properties.get('std', np.std(img_array))
    else:
        # 计算非零区域的均值和标准差
        mask = img_array > 0
        if np.any(mask):
            mean = np.mean(img_array[mask])
            std = np.std(img_array[mask])
        else:
            mean = np.mean(img_array)
            std = np.std(img_array)

    img_array = (img_array - mean) / (std + 1e-8)

    print(f"归一化后 - mean: {np.mean(img_array):.3f}, std: {np.std(img_array):.3f}")

    return img_array


def extract_encoder_features(
    model: nn.Module,
    image_array: np.ndarray,
    device: str = 'cuda',
    return_all_stages: bool = True,
    patch_size: Tuple[int, ...] = None
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    从预处理的图像中提取encoder特征

    参数:
        model: 加载的模型
        image_array: 预处理后的图像数组
        device: 设备
        return_all_stages: 是否返回所有stage的特征
        patch_size: patch大小（如果图像太大需要分patch处理）

    返回:
        特征数组或特征列表
    """
    device = torch.device(device)

    # 转换为tensor
    img_tensor = torch.from_numpy(image_array).float()
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W] 或 [1, 1, H, W]

    # 如果需要，调整大小以匹配patch_size
    if patch_size is not None:
        # 这里可以添加padding或cropping逻辑
        pass

    img_tensor = img_tensor.to(device)

    # 提取特征
    with torch.no_grad():
        # 通过stem
        x = img_tensor
        if model.encoder.stem is not None:
            x = model.encoder.stem(x)

        # 通过所有encoder stages
        features = []
        for s in range(len(model.encoder.stages)):
            x = model.encoder.stages[s](x)
            x = model.encoder.mamba_layers[s](x)
            features.append(x.cpu().numpy())

            print(f"Stage {s} 特征形状: {features[-1].shape}")

    if return_all_stages:
        return features
    else:
        return features[-1]  # bottleneck特征


def pool_features(
    features: np.ndarray,
    pooling_method: str = 'adaptive_avg'
) -> np.ndarray:
    """
    对特征进行池化，得到固定维度的特征向量

    参数:
        features: 特征数组 [B, C, D, H, W] 或 [B, C, H, W]
        pooling_method: 池化方法 ('adaptive_avg', 'max', 'avg')

    返回:
        池化后的特征向量 [B, C]
    """
    if pooling_method == 'adaptive_avg':
        # 全局平均池化
        spatial_dims = tuple(range(2, len(features.shape)))
        pooled = np.mean(features, axis=spatial_dims)
    elif pooling_method == 'max':
        spatial_dims = tuple(range(2, len(features.shape)))
        pooled = np.max(features, axis=spatial_dims)
    elif pooling_method == 'avg':
        spatial_dims = tuple(range(2, len(features.shape)))
        pooled = np.mean(features, axis=spatial_dims)
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")

    print(f"池化后特征形状: {pooled.shape}")
    return pooled


def save_features(
    features: Union[np.ndarray, List[np.ndarray]],
    output_path: str,
    metadata: dict = None
):
    """
    保存提取的特征

    参数:
        features: 特征数组或列表
        output_path: 输出路径
        metadata: 元数据（如图像路径、配置等）
    """
    output_path = Path(output_path)

    # 保存为.npz格式
    if isinstance(features, list):
        # 多尺度特征
        save_dict = {f'stage_{i}': feat for i, feat in enumerate(features)}
    else:
        save_dict = {'features': features}

    if metadata:
        save_dict['metadata'] = metadata

    np.savez_compressed(str(output_path), **save_dict)
    print(f"✓ 特征已保存到: {output_path}")


def batch_extract_features(
    model_folder: str,
    image_folder: str,
    output_folder: str,
    checkpoint: str = 'checkpoint_final.pth',
    pooling: bool = True,
    pooling_method: str = 'adaptive_avg',
    device: str = 'cuda'
):
    """
    批量提取特征

    参数:
        model_folder: 模型文件夹
        image_folder: 图像文件夹
        output_folder: 输出文件夹
        checkpoint: checkpoint名称
        pooling: 是否池化特征
        pooling_method: 池化方法
        device: 设备
    """
    # 加载模型
    model, plans, config_manager = load_trained_encoder(
        model_folder,
        checkpoint,
        device
    )

    # 创建输出目录
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    image_files = sorted(Path(image_folder).glob("*.nii.gz"))
    print(f"\n找到 {len(image_files)} 个图像文件")

    # 逐个处理
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n处理 [{idx}/{len(image_files)}]: {img_path.name}")

        try:
            # 预处理
            img_array = preprocess_image(str(img_path))

            # 提取特征
            features = extract_encoder_features(
                model,
                img_array,
                device=device,
                return_all_stages=True,
                patch_size=config_manager.patch_size
            )

            # 池化bottleneck特征
            if pooling:
                bottleneck_feat = features[-1]
                pooled_feat = pool_features(bottleneck_feat, pooling_method)
            else:
                pooled_feat = None

            # 保存
            output_path = output_folder / f"{img_path.stem}_features.npz"
            save_dict = {f'stage_{i}': feat for i, feat in enumerate(features)}

            if pooled_feat is not None:
                save_dict['pooled'] = pooled_feat

            save_dict['metadata'] = {
                'source_image': str(img_path),
                'pooling_method': pooling_method if pooling else None,
            }

            np.savez_compressed(str(output_path), **save_dict)
            print(f"  ✓ 特征已保存")

        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            continue

    print(f"\n完成! 所有特征已保存到: {output_folder}")


def main():
    parser = argparse.ArgumentParser(
        description='从训练好的U-Mamba Brain Encoder提取特征'
    )
    parser.add_argument(
        '--model_folder',
        type=str,
        required=True,
        help='模型文件夹路径（包含checkpoint和plans.json）'
    )
    parser.add_argument(
        '--image_folder',
        type=str,
        required=True,
        help='输入图像文件夹路径'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        required=True,
        help='输出特征文件夹路径'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoint_final.pth',
        help='Checkpoint文件名'
    )
    parser.add_argument(
        '--pooling',
        action='store_true',
        help='是否对特征进行全局池化'
    )
    parser.add_argument(
        '--pooling_method',
        type=str,
        default='adaptive_avg',
        choices=['adaptive_avg', 'max', 'avg'],
        help='池化方法'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备'
    )

    args = parser.parse_args()

    batch_extract_features(
        model_folder=args.model_folder,
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        checkpoint=args.checkpoint,
        pooling=args.pooling,
        pooling_method=args.pooling_method,
        device=args.device
    )


if __name__ == '__main__':
    main()


# ========== 使用示例 ==========
"""
# 单个图像提取特征
python extract_brain_features.py \\
    --model_folder /path/to/nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoder__nnUNetPlans__3d_fullres \\
    --image_folder /path/to/brain/images \\
    --output_folder /path/to/output/features \\
    --pooling \\
    --device cuda

# Python API使用
from extract_brain_features import load_trained_encoder, extract_encoder_features, pool_features

# 1. 加载模型
model, plans, config = load_trained_encoder(
    model_folder='path/to/model',
    checkpoint='checkpoint_final.pth'
)

# 2. 预处理图像
img_array = preprocess_image('brain.nii.gz')

# 3. 提取特征
features = extract_encoder_features(model, img_array)

# 4. 池化得到固定维度向量
feature_vector = pool_features(features[-1], pooling_method='adaptive_avg')

# feature_vector形状: [1, C]，C是通道数（如512或1024）
# 这个向量可以用于跨模态生成任务
"""
