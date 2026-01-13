# U-Mamba Brain Encoder 训练指南

用于训练3D脑MRI Brain Encoder，提取隐空间特征用于跨模态生成任务（脑-人脸）

## 📋 目录

1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [特征提取](#特征提取)
5. [用于跨模态生成](#用于跨模态生成)
6. [常见问题](#常见问题)

---

## 1. 环境准备

### 1.1 安装依赖

```bash
# 1. 安装U-Mamba
cd U-Mamba/umamba
pip install -e .

# 2. 安装mamba_ssm（必需）
pip install mamba-ssm

# 3. 安装其他依赖
pip install SimpleITK
pip install batchgenerators
```

### 1.2 配置路径

编辑 `umamba/nnunetv2/paths.py`：

```python
# 设置你的数据路径
base = '/path/to/your/data'  # 修改这里
nnUNet_raw = join(base, 'nnUNet_raw')
nnUNet_preprocessed = join(base, 'nnUNet_preprocessed')
nnUNet_results = join(base, 'nnUNet_results')
```

或者设置环境变量：

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

---

## 2. 数据准备

### 2.1 数据格式要求

你的数据应该是：
- **格式**: `.nii.gz` (NIfTI格式)
- **维度**: 128×128×128 (3D)
- **类型**: 单模态或多模态MRI

### 2.2 运行数据准备脚本

```bash
# 编辑 prepare_brain_dataset.py 中的路径配置
python prepare_brain_dataset.py
```

**配置说明**：
```python
SOURCE_IMAGE_DIR = "/path/to/your/brain/images"  # 你的原始图像目录
SOURCE_LABEL_DIR = None  # 如果有标签，设置标签目录；否则保持None
DATASET_ID = 800  # 数据集ID（建议800+）
DATASET_NAME = "BrainMRI"  # 数据集名称
```

### 2.3 数据目录结构

运行后会生成以下结构：

```
nnUNet_raw/
└── Dataset800_BrainMRI/
    ├── imagesTr/
    │   ├── brain_0001_0000.nii.gz
    │   ├── brain_0002_0000.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── brain_0001.nii.gz  # 如果有标签
    │   ├── brain_0002.nii.gz
    │   └── ...
    └── dataset.json
```

### 2.4 验证和预处理

```bash
# 验证数据完整性并生成预处理计划
nnUNetv2_plan_and_preprocess -d 800 --verify_dataset_integrity

# 这一步会：
# 1. 检查所有文件是否存在
# 2. 分析数据集统计信息（spacing、强度分布等）
# 3. 生成训练计划（plans.json）
# 4. 预处理数据（重采样、归一化等）
```

**预处理输出**：
- `nnUNet_preprocessed/Dataset800_BrainMRI/`：预处理后的数据
- `nnUNet_preprocessed/Dataset800_BrainMRI/nnUNetPlans.json`：训练计划

---

## 3. 模型训练

### 3.1 选择训练配置

根据你的数据维度选择：

- **2D**: 适用于切片数据
- **3D_lowres**: 适用于大尺寸3D数据（会降采样）
- **3D_fullres**: 适用于标准3D数据（推荐用于128³）
- **3D_cascade**: 先低分辨率再高分辨率（用于非常大的数据）

对于128×128×128的数据，推荐使用 **3d_fullres**。

### 3.2 开始训练

```bash
# 使用自定义的BrainEncoder训练器
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder

# 参数说明:
# 800: 数据集ID
# 3d_fullres: 配置
# all: 使用所有fold（如果只想用fold 0，改为0）
# -tr nnUNetTrainerBrainEncoder: 使用自定义训练器
```

### 3.3 训练选项

```bash
# 从checkpoint继续训练
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder --c

# 使用预训练权重
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder \\
    --pretrained_weights /path/to/checkpoint.pth

# 指定GPU
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder

# 调整批次大小（如果显存不足）
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder \\
    --disable_amp  # 禁用混合精度以节省显存
```

### 3.4 监控训练

训练日志保存在：
```
nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoder__nnUNetPlans__3d_fullres/
├── fold_all/
│   ├── training_log_*.txt      # 训练日志
│   ├── checkpoint_best.pth     # 最佳模型
│   ├── checkpoint_final.pth    # 最终模型
│   └── progress.png            # 训练曲线
```

查看实时日志：
```bash
tail -f nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoder__nnUNetPlans__3d_fullres/fold_all/training_log_*.txt
```

### 3.5 训练时间估计

根据硬件配置：
- **GPU**: NVIDIA RTX 3090 / A100
- **数据量**: 100个样本，128³
- **训练时间**: 约2-5小时/100 epochs

---

## 4. 特征提取

训练完成后，使用训练好的encoder提取特征。

### 4.1 批量提取特征

```bash
python extract_brain_features.py \\
    --model_folder nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoder__nnUNetPlans__3d_fullres/fold_all \\
    --image_folder /path/to/your/brain/images \\
    --output_folder /path/to/output/features \\
    --checkpoint checkpoint_final.pth \\
    --pooling \\
    --pooling_method adaptive_avg \\
    --device cuda
```

**参数说明**：
- `--model_folder`: 训练好的模型文件夹
- `--image_folder`: 要提取特征的图像文件夹
- `--output_folder`: 特征输出文件夹
- `--checkpoint`: 使用哪个checkpoint（`checkpoint_best.pth` 或 `checkpoint_final.pth`）
- `--pooling`: 是否全局池化（推荐开启）
- `--pooling_method`: 池化方法
  - `adaptive_avg`: 自适应平均池化（推荐）
  - `max`: 最大池化
  - `avg`: 平均池化

### 4.2 输出格式

每个图像会生成一个 `.npz` 文件：

```python
# 加载特征
features = np.load('brain_0001_0000_features.npz')

# 包含的内容：
# - stage_0: 第1个stage的特征 [1, C0, D0, H0, W0]
# - stage_1: 第2个stage的特征 [1, C1, D1, H1, W1]
# - ...
# - stage_N: 最后一个stage的特征（bottleneck）[1, CN, DN, HN, WN]
# - pooled: 池化后的特征向量 [1, CN]（如果使用--pooling）
# - metadata: 元数据

# 获取pooled特征用于跨模态生成
pooled_feature = features['pooled']  # shape: [1, CN]，如 [1, 512]
```

### 4.3 Python API使用

```python
from extract_brain_features import (
    load_trained_encoder,
    preprocess_image,
    extract_encoder_features,
    pool_features
)

# 1. 加载模型
model, plans, config = load_trained_encoder(
    model_folder='nnUNet_results/Dataset800_BrainMRI/...',
    checkpoint='checkpoint_final.pth',
    device='cuda'
)

# 2. 预处理单个图像
img_array = preprocess_image('brain.nii.gz')

# 3. 提取多尺度特征
features = extract_encoder_features(
    model,
    img_array,
    device='cuda',
    return_all_stages=True
)

# features是一个list:
# features[0]: stage 0特征
# features[1]: stage 1特征
# ...
# features[-1]: bottleneck特征（最深层）

# 4. 池化得到固定维度向量
feature_vector = pool_features(
    features[-1],
    pooling_method='adaptive_avg'
)

print(f"特征向量维度: {feature_vector.shape}")  # [1, C]
```

---

## 5. 用于跨模态生成

### 5.1 特征维度

提取的特征维度取决于模型配置：

- **Bottleneck特征（未池化）**: `[1, C, D, H, W]`
  - C: 通道数（如256, 512, 1024）
  - D, H, W: 空间维度（如 4×4×4 或 8×8×8）

- **Pooled特征（池化后）**: `[1, C]`
  - 固定维度的向量，适合直接输入到生成模型

### 5.2 与生成模型集成

#### 方案1: 使用pooled特征作为条件

```python
import torch
import numpy as np

# 加载脑特征
brain_features = np.load('brain_features.npz')['pooled']  # [1, 512]
brain_features = torch.from_numpy(brain_features).float()

# 输入到生成模型
class CrossModalGenerator(nn.Module):
    def __init__(self, brain_feat_dim=512, ...):
        super().__init__()
        # brain encoder的特征作为条件
        self.condition_proj = nn.Linear(brain_feat_dim, ...)
        # 你的生成器架构
        ...

    def forward(self, noise, brain_condition):
        # brain_condition: [B, 512]
        condition_feat = self.condition_proj(brain_condition)
        # 生成人脸
        ...

# 使用
generator = CrossModalGenerator(brain_feat_dim=512)
noise = torch.randn(1, 128)
generated_face = generator(noise, brain_features)
```

#### 方案2: 使用多尺度特征

```python
# 加载多尺度特征
features = np.load('brain_features.npz')
stage_features = [features[f'stage_{i}'] for i in range(num_stages)]

# 用于更精细的生成控制
class MultiScaleGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # 在生成器的不同层注入不同尺度的脑特征
        ...

    def forward(self, noise, multi_scale_brain_features):
        # 在不同的上采样阶段融合不同尺度的特征
        ...
```

### 5.3 训练策略建议

#### 如果有配对数据（脑-人脸）

```python
# 训练跨模态生成模型
for brain_mri, face_image in paired_dataloader:
    # 提取脑特征（frozen encoder）
    with torch.no_grad():
        brain_feat = brain_encoder(brain_mri)

    # 生成人脸
    generated_face = generator(noise, brain_feat)

    # 损失
    reconstruction_loss = criterion(generated_face, face_image)
    perceptual_loss = perceptual_criterion(generated_face, face_image)

    loss = reconstruction_loss + lambda_p * perceptual_loss
    loss.backward()
```

#### 如果数据未配对

考虑使用CycleGAN或对比学习：

```python
# 对比学习：让配对的脑-人脸特征接近
brain_feat = brain_encoder(brain_mri)
face_feat = face_encoder(face_image)

contrastive_loss = contrastive_criterion(brain_feat, face_feat)
```

---

## 6. 常见问题

### Q1: 我的数据没有标签怎么办？

**答**: 有几种方案：

1. **使用重建任务**：让模型重建输入图像
   - 修改训练目标为reconstruction loss
   - encoder学习有用的表征

2. **自监督学习**：
   - 使用SimCLR等对比学习方法
   - 需要修改训练代码

3. **创建伪标签**：
   - 使用脑区分割的预训练模型生成标签
   - 然后用这些标签训练

### Q2: 显存不足怎么办？

```bash
# 方法1: 降低patch size（修改plans.json）
# 方法2: 减小batch size
# 方法3: 禁用mixed precision
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder --disable_amp

# 方法4: 使用梯度累积（需要修改训练器）
```

### Q3: 如何选择最佳checkpoint？

```bash
# 使用验证集性能最好的
checkpoint_best.pth  # 推荐

# 或使用最后的checkpoint
checkpoint_final.pth
```

### Q4: 特征维度太高怎么办？

```python
# 方法1: 使用PCA降维
from sklearn.decomposition import PCA

pca = PCA(n_components=256)
reduced_features = pca.fit_transform(features)

# 方法2: 训练一个projection head
projection = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128)
)
```

### Q5: 如何微调encoder？

```python
# 在跨模态任务上微调
brain_encoder.train()  # 设为训练模式

for brain_mri, face_image in dataloader:
    # 提取特征
    brain_feat = brain_encoder(brain_mri)

    # 生成
    generated = generator(brain_feat)

    # 反向传播（encoder也会更新）
    loss = criterion(generated, face_image)
    loss.backward()
    optimizer.step()
```

### Q6: 训练收敛慢或不收敛？

**可能原因**：
1. 学习率太高或太低
2. 数据归一化不正确
3. 数据量太少（考虑数据增强）
4. Mamba层的数值不稳定（尝试NoAMP版本）

```bash
# 使用NoAMP版本
# 修改代码使用 nnUNetTrainerUMambaEncNoAMP 作为基类
```

---

## 7. 进阶：自定义训练目标

如果你想使用重建任务或其他自定义目标，修改训练器：

```python
# umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerBrainEncoder.py

class nnUNetTrainerBrainEncoderReconstruction(nnUNetTrainerBrainEncoder):
    """
    使用重建任务训练encoder
    """

    def compute_loss(self, output, target):
        # 修改损失函数为reconstruction loss
        reconstruction_loss = F.mse_loss(output, target)

        # 可以添加其他正则化项
        # 例如：特征的稀疏性约束

        return reconstruction_loss
```

---

## 8. 总结工作流

```bash
# 完整工作流程
# 1. 准备数据
python prepare_brain_dataset.py

# 2. 预处理
nnUNetv2_plan_and_preprocess -d 800 --verify_dataset_integrity

# 3. 训练
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder

# 4. 提取特征
python extract_brain_features.py \\
    --model_folder nnUNet_results/Dataset800_BrainMRI/.../fold_all \\
    --image_folder /path/to/images \\
    --output_folder /path/to/features \\
    --pooling

# 5. 在跨模态生成任务中使用特征
# （在你的生成模型代码中加载和使用特征）
```

---

## 参考资源

- **U-Mamba论文**: [链接]
- **nnU-Net文档**: https://github.com/MIC-DKFZ/nnUNet
- **Mamba**: https://github.com/state-spaces/mamba

---

## 联系和反馈

如有问题，请检查：
1. 环境是否正确安装
2. 数据格式是否正确
3. 路径配置是否正确
4. 显存是否足够

祝训练顺利！🚀
