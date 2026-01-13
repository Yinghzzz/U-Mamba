# U-Mamba Brain Encoder 使用指南

> 使用U-Mamba训练3D脑MRI编码器，提取隐空间特征用于脑-人脸跨模态生成

## 🎯 项目目标

将U-Mamba作为**Brain Encoder**来提取3D脑MRI的隐空间特征向量，用于跨模态生成任务（例如：从大脑MRI生成对应的3D人脸）。

## 📦 你的数据

- **格式**: `.nii.gz` (NIfTI格式的医学影像)
- **维度**: 128×128×128 (3D体积数据)
- **内容**: 3D脑部MRI扫描

## 🚀 快速开始

### 方法1: 使用自动化脚本（推荐）

```bash
# 1. 首先编辑配置
vim quick_start_brain_encoder.sh
# 修改 SOURCE_IMAGE_DIR 为你的数据路径

# 2. 一键运行完整流程
./quick_start_brain_encoder.sh all

# 或者分步执行：
./quick_start_brain_encoder.sh setup       # 安装环境
./quick_start_brain_encoder.sh prepare     # 准备数据
./quick_start_brain_encoder.sh preprocess  # 预处理
./quick_start_brain_encoder.sh train       # 训练模型
./quick_start_brain_encoder.sh extract     # 提取特征
```

### 方法2: 手动执行（更灵活）

#### 步骤1: 安装环境

```bash
# 安装U-Mamba
cd umamba
pip install -e .

# 安装mamba-ssm（核心依赖）
pip install mamba-ssm

# 安装其他依赖
pip install SimpleITK batchgenerators
```

#### 步骤2: 准备数据

```bash
# 1. 编辑数据准备脚本
vim prepare_brain_dataset.py

# 修改以下配置：
# SOURCE_IMAGE_DIR = "/path/to/your/brain/images"  # 你的图像目录
# SOURCE_LABEL_DIR = None  # 如果有标签就填路径，没有就None
# DATASET_ID = 800  # 数据集编号
# DATASET_NAME = "BrainMRI"

# 2. 运行准备脚本
python prepare_brain_dataset.py
```

**输出结果**：
```
data/nnUNet_raw/Dataset800_BrainMRI/
├── imagesTr/
│   ├── brain_0001_0000.nii.gz
│   ├── brain_0002_0000.nii.gz
│   └── ...
├── labelsTr/  # 如果有标签
└── dataset.json
```

#### 步骤3: 预处理数据

```bash
# 验证数据完整性并生成训练计划
nnUNetv2_plan_and_preprocess -d 800 --verify_dataset_integrity
```

这一步会：
- ✅ 检查所有文件完整性
- 📊 分析数据集统计信息（spacing、强度分布等）
- 📝 生成训练计划（plans.json）
- 🔄 预处理数据（重采样、归一化）

#### 步骤4: 训练Brain Encoder

```bash
# 使用3D全分辨率配置训练
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder

# 参数说明：
# 800: 数据集ID
# 3d_fullres: 3D全分辨率配置（适合128³数据）
# all: 使用所有fold训练
# -tr nnUNetTrainerBrainEncoder: 使用自定义的Brain Encoder训练器
```

**训练选项**：
```bash
# 指定GPU
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder

# 从checkpoint继续训练
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder --c

# 如果显存不足
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder --disable_amp
```

**训练输出**：
```
nnUNet_results/
└── Dataset800_BrainMRI/
    └── nnUNetTrainerBrainEncoder__nnUNetPlans__3d_fullres/
        └── fold_all/
            ├── checkpoint_best.pth      # 最佳模型
            ├── checkpoint_final.pth     # 最终模型
            ├── training_log_*.txt       # 训练日志
            ├── progress.png             # 训练曲线
            └── plans.json               # 训练计划
```

#### 步骤5: 提取特征

```bash
# 批量提取特征
python extract_brain_features.py \
    --model_folder nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoder__nnUNetPlans__3d_fullres/fold_all \
    --image_folder /path/to/your/test/images \
    --output_folder /path/to/output/features \
    --checkpoint checkpoint_final.pth \
    --pooling \
    --pooling_method adaptive_avg \
    --device cuda
```

**输出特征**：
```
output/features/
├── brain_0001_0000_features.npz
├── brain_0002_0000_features.npz
└── ...

每个.npz文件包含：
- stage_0, stage_1, ..., stage_N: 多尺度特征
- pooled: 池化后的特征向量 [1, C]（如 [1, 512]）
- metadata: 元数据
```

## 📊 提取的特征说明

### 多尺度特征

U-Mamba encoder包含多个stage，每个stage提取不同尺度的特征：

```python
import numpy as np

# 加载特征
features = np.load('brain_0001_0000_features.npz')

# 查看所有可用的key
print(features.files)  # ['stage_0', 'stage_1', 'stage_2', 'stage_3', 'pooled', 'metadata']

# 不同stage的特征
stage_0 = features['stage_0']  # 浅层特征，空间分辨率高
stage_1 = features['stage_1']  # 中层特征
stage_2 = features['stage_2']  # 深层特征
stage_3 = features['stage_3']  # Bottleneck特征，语义最丰富

# 池化后的特征向量（最常用于跨模态生成）
pooled_feat = features['pooled']  # shape: [1, 512]（具体维度取决于模型配置）

print(f"Pooled特征维度: {pooled_feat.shape}")
```

### 特征维度说明

| 特征类型 | 形状 | 说明 | 用途 |
|---------|------|------|------|
| Stage 0 | `[1, 64, 64, 64, 64]` | 浅层特征，细节丰富 | 细粒度生成控制 |
| Stage 1 | `[1, 128, 32, 32, 32]` | 中层特征 | 中等语义特征 |
| Stage 2 | `[1, 256, 16, 16, 16]` | 深层特征 | 高级语义特征 |
| Stage 3 | `[1, 512, 8, 8, 8]` | Bottleneck特征 | 最抽象的表征 |
| **Pooled** | **`[1, 512]`** | **全局池化向量** | **跨模态生成的条件向量** |

## 🔗 集成到跨模态生成模型

### 方案1: 作为条件向量（最简单）

```python
import torch
import torch.nn as nn
import numpy as np

# 1. 加载脑特征
brain_feat = np.load('brain_features.npz')['pooled']  # [1, 512]
brain_feat = torch.from_numpy(brain_feat).float().cuda()

# 2. 定义跨模态生成器
class BrainToFaceGenerator(nn.Module):
    def __init__(self, brain_dim=512, latent_dim=128, output_size=256):
        super().__init__()

        # 脑特征投影
        self.brain_encoder = nn.Sequential(
            nn.Linear(brain_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # 生成器主体（例如使用DCGAN架构）
        self.generator = nn.Sequential(
            # 你的生成器架构
            # 输入: [B, latent_dim]
            # 输出: [B, 3, output_size, output_size] 人脸图像
            ...
        )

    def forward(self, brain_features, noise=None):
        # 编码脑特征
        encoded = self.brain_encoder(brain_features)  # [B, latent_dim]

        # 可选：与噪声向量结合
        if noise is not None:
            encoded = encoded + noise

        # 生成人脸
        face = self.generator(encoded)
        return face

# 3. 使用
generator = BrainToFaceGenerator(brain_dim=512).cuda()
generated_face = generator(brain_feat)

print(f"生成的人脸形状: {generated_face.shape}")  # [1, 3, 256, 256]
```

### 方案2: 多尺度特征融合（更精细）

```python
class MultiScaleBrainToFaceGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # 不同尺度特征的处理器
        self.stage3_proj = nn.Conv3d(512, 256, 1)  # bottleneck
        self.stage2_proj = nn.Conv3d(256, 128, 1)  # 中层
        self.stage1_proj = nn.Conv3d(128, 64, 1)   # 浅层

        # 上采样生成器
        self.decoder = nn.ModuleList([
            # 逐步上采样并融合多尺度特征
            ...
        ])

    def forward(self, multi_scale_features):
        # multi_scale_features: list of [stage_0, stage_1, stage_2, stage_3]

        # 从最深层开始，逐步上采样并融合浅层特征
        x = self.stage3_proj(multi_scale_features[3])

        for i, decoder_layer in enumerate(self.decoder):
            x = decoder_layer(x)
            # 在适当的层融合浅层特征
            if i < len(multi_scale_features) - 1:
                skip_feat = self.stage_projs[i](multi_scale_features[i])
                x = x + skip_feat

        return x
```

### 方案3: 训练策略

#### 有配对数据（脑-人脸配对）

```python
# 训练跨模态生成模型
for epoch in range(num_epochs):
    for brain_mri, face_image in paired_dataloader:
        # 提取脑特征（使用预训练的frozen encoder）
        with torch.no_grad():
            brain_feat = brain_encoder(brain_mri)['pooled']

        # 生成人脸
        generated_face = generator(brain_feat)

        # 计算损失
        reconstruction_loss = F.mse_loss(generated_face, face_image)
        perceptual_loss = vgg_loss(generated_face, face_image)  # 使用VGG感知损失

        loss = reconstruction_loss + 0.1 * perceptual_loss
        loss.backward()
        optimizer.step()
```

#### 无配对数据（未配对的脑和人脸）

```python
# 使用对比学习或CycleGAN
for epoch in range(num_epochs):
    # 1. 对比学习：让相关的脑-人脸特征接近
    brain_feat = brain_encoder(brain_mri)
    face_feat = face_encoder(face_image)

    # InfoNCE loss等对比学习损失
    contrastive_loss = info_nce_loss(brain_feat, face_feat)

    # 2. 或使用CycleGAN
    # Brain -> Face
    fake_face = brain_to_face_generator(brain_feat)

    # Face -> Brain
    reconstructed_brain = face_to_brain_generator(fake_face)

    # Cycle consistency loss
    cycle_loss = F.l1_loss(reconstructed_brain, brain_mri)
```

## 📈 关键特征

### ✅ 已实现功能

- ✅ 自动数据准备和格式转换
- ✅ 基于nnU-Net的自动预处理
- ✅ U-Mamba encoder训练（含Mamba SSM层）
- ✅ 多尺度特征提取
- ✅ 灵活的特征池化选项
- ✅ 批量特征提取脚本
- ✅ 完整的使用文档

### 🎯 适用场景

1. **跨模态生成**: 脑-人脸、脑-声音等
2. **脑特征分析**: 提取脑部表征用于下游任务
3. **多模态学习**: 将脑特征与其他模态对齐
4. **医学影像研究**: 脑MRI的深度学习应用

## 🔧 常见问题

### Q1: 我的数据没有标签怎么办？

**方案1**: 使用自监督学习
```python
# 修改训练目标为重建任务
class ReconstructionTrainer(nnUNetTrainerBrainEncoder):
    def compute_loss(self, output, target):
        # target就是输入图像本身
        return F.mse_loss(output, target)
```

**方案2**: 使用预训练模型生成伪标签
```bash
# 使用其他脑分割模型生成标签
# 然后用这些标签训练
```

### Q2: 显存不足

```bash
# 方法1: 禁用混合精度训练
nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder --disable_amp

# 方法2: 减小patch size（需要修改plans.json）

# 方法3: 使用梯度累积（需要修改训练器代码）
```

### Q3: 特征维度太大

```python
# 使用PCA降维
from sklearn.decomposition import PCA

features = np.load('features.npz')['pooled']  # [1, 512]
pca = PCA(n_components=128)
reduced_features = pca.fit_transform(features)  # [1, 128]
```

### Q4: 如何选择最佳checkpoint？

```bash
# 推荐使用验证集上性能最好的
checkpoint_best.pth

# 或最后保存的
checkpoint_final.pth
```

### Q5: 如何可视化特征？

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 1. 提取多个样本的特征
all_features = []
for feat_file in feature_files:
    feat = np.load(feat_file)['pooled']
    all_features.append(feat.flatten())

all_features = np.array(all_features)  # [N, 512]

# 2. t-SNE可视化
tsne = TSNE(n_components=2)
features_2d = tsne.fit_transform(all_features)

plt.scatter(features_2d[:, 0], features_2d[:, 1])
plt.title('Brain Features t-SNE')
plt.show()
```

## 📚 进阶主题

### 1. 微调Encoder

```python
# 在跨模态任务上微调encoder
brain_encoder.train()  # 解冻

for brain_mri, face_image in dataloader:
    # encoder参与梯度更新
    brain_feat = brain_encoder(brain_mri)
    generated = generator(brain_feat)
    loss = criterion(generated, face_image)
    loss.backward()  # encoder权重也会更新
    optimizer.step()
```

### 2. 自定义训练目标

```python
# 在nnUNetTrainerBrainEncoder.py中修改
class CustomBrainEncoder(nnUNetTrainerBrainEncoder):
    def compute_loss(self, output, target):
        # 自定义损失函数
        recon_loss = F.mse_loss(output, target)

        # 添加额外的正则化
        # 例如：特征稀疏性
        features = self.extract_encoder_features(target)
        sparsity_loss = torch.mean(torch.abs(features))

        return recon_loss + 0.01 * sparsity_loss
```

### 3. 数据增强

nnU-Net会自动应用数据增强，包括：
- 旋转、翻转
- 弹性变形
- 强度变换
- 高斯噪声

如需自定义增强，修改训练器的`get_training_transforms()`方法。

## 📖 完整文档

详细的使用指南请参考：
- **[BRAIN_ENCODER_GUIDE.md](BRAIN_ENCODER_GUIDE.md)** - 完整英文文档
- **代码文件**:
  - `prepare_brain_dataset.py` - 数据准备脚本
  - `extract_brain_features.py` - 特征提取脚本
  - `umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerBrainEncoder.py` - 训练器
  - `quick_start_brain_encoder.sh` - 快速启动脚本

## 🤝 反馈与支持

如遇问题，请检查：
1. ✅ 环境依赖是否正确安装
2. ✅ 数据路径配置是否正确
3. ✅ 数据格式是否符合要求
4. ✅ GPU显存是否充足

## 📝 引用

如果这个工作对你有帮助，请考虑引用：

```bibtex
@article{umamba,
  title={U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation},
  author={...},
  journal={...},
  year={2024}
}
```

---

**祝你的跨模态生成项目顺利！🎉**

如有任何问题，欢迎提issue或PR。
