# 自监督重建任务指南

## 🎯 为什么使用重建任务？

**当前分割任务的问题**：
- 简单的前景/背景分割（Dice 0.993+）
- Encoder只学到"是否是大脑组织"
- ❌ 缺乏大脑内部的精细结构表征

**重建任务的优势**：
- ✅ Encoder必须学习大脑的**所有细节**（皮层、白质、灰质、脑沟等）
- ✅ 自监督学习，不需要人工标注
- ✅ 提取的特征更适合跨模态生成任务

## 🚀 快速开始

### 步骤1: 停止当前训练

```bash
# 在训练终端按 Ctrl+C 停止当前分割训练
```

### 步骤2: 准备重建标签

```bash
cd ~/U-Mamba
git pull origin claude/umamba-brain-encoder-KRWeS

# 为重建任务准备标签（标签=原始图像）
python prepare_reconstruction_labels.py \
    --image_dir data/nnUNet_raw/Dataset800_BrainMRI/imagesTr \
    --label_dir data/nnUNet_raw/Dataset800_BrainMRI/labelsTr \
    --method copy
```

**说明**：
- `--method copy`: 直接复制图像作为标签（快速，推荐）
- `--method normalize`: 归一化后保存（可选）

### 步骤3: 重新预处理（可选）

如果你修改了标签，需要重新预处理：

```bash
# 删除旧的预处理数据
rm -rf data/nnUNet_preprocessed/Dataset800_BrainMRI

# 重新预处理
nnUNetv2_plan_and_preprocess -d 800 --verify_dataset_integrity
```

### 步骤4: 开始重建训练

#### 方法1: 基础重建（推荐）

```bash
CUDA_VISIBLE_DEVICES=4,5 nnUNetv2_train 800 3d_fullres all \
    -tr nnUNetTrainerBrainEncoderReconstruction \
    -num_gpus 2
```

#### 方法2: 高级重建（包含梯度损失）

```bash
CUDA_VISIBLE_DEVICES=4,5 nnUNetv2_train 800 3d_fullres all \
    -tr nnUNetTrainerBrainEncoderReconstructionAdvanced \
    -num_gpus 2
```

## 📊 监控重建训练

### 使用监控脚本

```bash
./monitor_training.sh
```

### 关键指标

在重建任务中，关注这些指标：

#### 1. **val_loss** (越低越好)
- 重建损失（MSE + L1）
- 目标: < 0.05
- 当前分割任务的loss约-0.96（不可比）

#### 2. **val_psnr** (越高越好) 🌟
- Peak Signal-to-Noise Ratio
- 范围: 0-50+ dB
- 评价标准：
  - `< 20 dB`: ❌ 差
  - `20-30 dB`: ⚠️ 一般
  - `30-40 dB`: ✅ 好
  - `> 40 dB`: 🌟 优秀

#### 3. **val_mae** (越低越好)
- Mean Absolute Error
- 目标: < 0.05
- 越小说明重建越精确

### 预期进展

```
Epoch 0:
  val_loss: 0.15
  val_psnr: 25.3 dB  ⚠️ (刚开始)
  val_mae: 0.082

Epoch 50:
  val_loss: 0.042
  val_psnr: 32.8 dB  ✅ (进步)
  val_mae: 0.038

Epoch 200:
  val_loss: 0.018
  val_psnr: 38.5 dB  🌟 (很好)
  val_mae: 0.015
```

## ⚖️ 两种任务对比

| 指标 | 分割任务 | 重建任务 |
|------|---------|---------|
| **任务** | 前景/背景分割 | 重建原始图像 |
| **主要指标** | Dice (0-1, ↑) | PSNR (dB, ↑) |
| **当前性能** | 0.993 (很高) | 待训练 |
| **Encoder学到的** | 是否是脑组织 | 脑内部精细结构 |
| **适用场景** | 脑区定位 | **跨模态生成** ✅ |
| **训练难度** | 简单 | 中等 |

## 🎯 训练建议

### 训练时长

- **最少**: 100 epochs (~3小时)
- **推荐**: 200-300 epochs (~6-9小时)
- **最多**: 500 epochs (~15小时)

### 何时停止

当以下条件满足时可以停止：

1. ✅ PSNR > 35 dB
2. ✅ MAE < 0.03
3. ✅ 连续50 epochs没有明显提升

## 🔧 高级选项

### 调整损失权重

编辑训练器，修改损失权重：

```python
# umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainerBrainEncoderReconstruction.py

def __init__(self, ...):
    super().__init__(...)

    self.mse_weight = 1.0      # MSE权重 (调整这个)
    self.l1_weight = 0.5       # L1权重 (调整这个)
```

**建议**：
- 如果重建太模糊：增加L1权重（如0.8）
- 如果重建不够平滑：增加MSE权重（如1.5）

### 使用高级重建器

高级重建器添加了**梯度损失**，可以更好地保持边缘细节：

```bash
# 使用高级重建器
CUDA_VISIBLE_DEVICES=4,5 nnUNetv2_train 800 3d_fullres all \
    -tr nnUNetTrainerBrainEncoderReconstructionAdvanced \
    -num_gpus 2
```

## 📈 训练后的使用

### 提取特征（方法相同）

重建训练后的Encoder提取特征的方法**完全相同**：

```bash
python extract_brain_features.py \
    --model_folder nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoderReconstruction__nnUNetPlans__3d_fullres/fold_all \
    --image_folder /path/to/images \
    --output_folder /path/to/features \
    --pooling
```

### 特征质量对比

| 训练任务 | 特征质量 | 适用场景 |
|---------|---------|---------|
| **分割任务** | 粗糙（只有前景/背景） | 脑区定位、粗略分类 |
| **重建任务** | 精细（完整的脑结构） | **跨模态生成、细粒度任务** ✅ |

## 💡 常见问题

### Q1: 可以同时训练两个模型吗？

可以！分别使用不同的数据集ID：

```bash
# 分割任务（Dataset800）
CUDA_VISIBLE_DEVICES=4,5 nnUNetv2_train 800 3d_fullres all -tr nnUNetTrainerBrainEncoder -num_gpus 2

# 重建任务（Dataset801）
CUDA_VISIBLE_DEVICES=6,7 nnUNetv2_train 801 3d_fullres all -tr nnUNetTrainerBrainEncoderReconstruction -num_gpus 2
```

### Q2: 重建任务比分割任务慢吗？

速度相似：
- 分割：~100s/epoch
- 重建：~100-120s/epoch

### Q3: 重建质量不好怎么办？

1. **检查数据**：确保标签就是原始图像
2. **调整损失权重**：增加L1权重
3. **增加训练时间**：重建任务可能需要更多epochs
4. **使用高级重建器**：包含梯度损失

### Q4: 分割任务已经训练了，是否浪费了？

不浪费！可以：
1. 保留分割模型作为baseline
2. 训练重建模型
3. 对比两者提取的特征质量
4. 选择更适合你任务的模型

### Q5: PSNR多少算好？

- **20-25 dB**: 基本可用
- **25-30 dB**: 一般
- **30-35 dB**: 好 ✅
- **35-40 dB**: 很好 🌟
- **> 40 dB**: 优秀（接近完美重建）

## 📝 总结

**推荐方案**：
1. ✅ 停止当前分割训练
2. ✅ 使用重建任务重新训练
3. ✅ 训练200-300 epochs
4. ✅ 提取特征用于跨模态生成

**预期结果**：
- Encoder学到更精细的大脑表征
- 特征包含脑内部结构信息
- 更适合脑-人脸跨模态生成任务

---

有问题随时问！🚀
