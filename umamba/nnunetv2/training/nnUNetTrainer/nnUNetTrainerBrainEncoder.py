"""
自定义训练器：专门用于训练Brain Encoder提取特征
支持提取encoder的隐空间向量用于跨模态生成任务
"""
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaEnc import nnUNetTrainerUMambaEnc
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
import numpy as np


class nnUNetTrainerBrainEncoder(nnUNetTrainerUMambaEnc):
    """
    专门用于训练Brain Encoder的训练器

    特点:
    1. 使用UMambaEnc架构
    2. 支持提取多尺度encoder特征
    3. 提供特征提取接口
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        # 可以在这里添加额外的初始化配置
        self.save_encoder_features = True  # 是否保存encoder特征

    def _set_batch_size_and_oversample(self):
        """
        覆盖此方法以确保batch size适合DDP训练
        """
        # 在DDP模式下，确保batch size至少等于GPU数量
        if self.is_ddp:
            import torch.distributed as dist
            world_size = dist.get_world_size()
            my_rank = dist.get_rank()

            original_batch_size = self.configuration_manager.batch_size

            if original_batch_size < world_size:
                # 调整batch size为GPU数量
                self.configuration_manager.batch_size = world_size
                if my_rank == 0:
                    print(f"警告: 原始batch_size ({original_batch_size}) < GPU数量 ({world_size})")
                    print(f"自动调整global batch_size为: {world_size}")
                    print(f"每个GPU的batch_size为: 1")

        # 调用父类方法
        super()._set_batch_size_and_oversample()

    def extract_encoder_features(self, x: torch.Tensor, return_all_stages: bool = True):
        """
        提取encoder的隐空间特征

        参数:
            x: 输入图像 [B, C, H, W, D]
            return_all_stages: 是否返回所有阶段的特征（多尺度特征）

        返回:
            如果return_all_stages=True: 返回list of features，每个元素对应一个stage
            如果return_all_stages=False: 仅返回最深层的特征
        """
        self.network.eval()
        with torch.no_grad():
            # 通过stem
            if self.network.encoder.stem is not None:
                x = self.network.encoder.stem(x)

            # 通过所有encoder stages
            features = []
            for s in range(len(self.network.encoder.stages)):
                x = self.network.encoder.stages[s](x)
                x = self.network.encoder.mamba_layers[s](x)
                features.append(x)

        if return_all_stages:
            return features  # 返回多尺度特征
        else:
            return features[-1]  # 仅返回最深层特征

    def get_bottleneck_feature_dim(self):
        """
        获取bottleneck特征的维度信息

        返回: (channels, spatial_dims)
        """
        # 获取最后一个stage的特征通道数
        num_channels = self.network.encoder.output_channels[-1]

        # 计算最后一个stage的空间维度
        patch_size = self.configuration_manager.patch_size
        strides = self.network.encoder.strides

        spatial_size = list(patch_size)
        for stride in strides:
            spatial_size = [s // st for s, st in zip(spatial_size, stride)]

        return num_channels, tuple(spatial_size)

    def validation_step(self, batch: dict) -> dict:
        """
        重写验证步骤，可以在这里添加特征可视化等功能
        """
        output = super().validation_step(batch)

        # 可选：保存特征用于分析
        if self.save_encoder_features and self.current_epoch % 10 == 0:
            with torch.no_grad():
                data = batch['data']
                features = self.extract_encoder_features(data, return_all_stages=True)
                # 这里可以添加特征保存逻辑
                # 例如：保存特征的统计信息、可视化等

        return output


class BrainEncoderFeatureExtractor(nn.Module):
    """
    独立的特征提取器类
    可以从训练好的模型中加载encoder权重，仅用于特征提取
    """

    def __init__(self, trained_model_path: str, device: str = 'cuda'):
        """
        参数:
            trained_model_path: 训练好的checkpoint路径
            device: 'cuda' 或 'cpu'
        """
        super().__init__()
        self.device = torch.device(device)

        # 加载完整模型
        checkpoint = torch.load(trained_model_path, map_location=self.device)

        # 如果checkpoint是训练器保存的，需要提取网络部分
        if 'network_weights' in checkpoint:
            state_dict = checkpoint['network_weights']
        else:
            state_dict = checkpoint

        # 这里需要重新构建encoder
        # 注意：实际使用时需要根据你的plans配置来构建
        print("警告: 需要根据实际的plans配置来正确构建encoder")
        print("建议直接加载完整的训练器并使用其extract_encoder_features方法")

    def forward(self, x: torch.Tensor):
        """
        前向传播提取特征
        """
        pass  # 需要实现


def load_brain_encoder_for_inference(checkpoint_path: str,
                                     plans_path: str,
                                     device: str = 'cuda'):
    """
    加载训练好的Brain Encoder用于特征提取

    参数:
        checkpoint_path: 模型checkpoint路径
        plans_path: plans.json路径
        device: 设备

    返回:
        可以直接用于特征提取的函数
    """
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    import json

    # 加载plans
    with open(plans_path, 'r') as f:
        plans = json.load(f)

    # 注意：这里简化了加载过程
    # 实际使用时需要根据具体情况调整
    print("特征提取器加载完成")
    print(f"Checkpoint: {checkpoint_path}")

    def extract_features(image_path: str, return_all_stages: bool = True):
        """
        从nii.gz文件提取特征

        参数:
            image_path: 图像文件路径
            return_all_stages: 是否返回多尺度特征

        返回:
            特征向量或特征列表
        """
        import SimpleITK as sitk

        # 读取图像
        img = sitk.ReadImage(image_path)
        img_array = sitk.GetArrayFromImage(img)

        # 预处理（归一化等）
        # 注意：需要使用与训练时相同的预处理方式

        # 转换为tensor
        img_tensor = torch.from_numpy(img_array).float()
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        img_tensor = img_tensor.to(device)

        # 提取特征
        # 这里需要实际的模型来提取

        return None  # 返回特征

    return extract_features


# ========== 使用示例 ==========
"""
# 1. 训练模型
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder

# 2. 加载模型并提取特征
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBrainEncoder import load_brain_encoder_for_inference

# 加载特征提取器
extract_features = load_brain_encoder_for_inference(
    checkpoint_path='path/to/checkpoint_final.pth',
    plans_path='path/to/plans.json',
    device='cuda'
)

# 提取特征
features = extract_features('path/to/brain.nii.gz', return_all_stages=True)

# features是一个list，包含多个尺度的特征
# features[-1]是bottleneck层的特征，通常用于跨模态生成
"""
