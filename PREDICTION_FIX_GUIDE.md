# 重建任务预测问题修复指南

## 🔴 问题诊断

你遇到的问题：
```
训练时: val_psnr 23.13 dB, val_mae 0.0114  ✓ 正常
预测时: PSNR 7.66 dB, MAE 0.2279           ✗ 异常（差10倍！）
```

### 问题根本原因

**`nnUNetv2_predict` 使用的是分割任务的预测流程，不适用于重建任务！**

具体问题：
1. **Softmax/Argmax处理**：nnUNet会对输出做argmax取最大值（用于分割），这会把连续值变成离散label
2. **Label映射**：输出会被映射回label ID（0, 1, 2...），破坏了重建图像的连续性
3. **后处理不匹配**：分割任务的后处理不适合重建任务

### 为什么训练指标正常？

训练时的验证指标是在**预处理后的数据空间**直接计算的，没有经过上述后处理，所以是正确的。

---

## ✅ 解决方案

我创建了两个专门用于重建任务的预测脚本。

### 方案1：使用预处理后的数据（推荐）⭐

**为什么推荐？**
- 使用与训练完全相同的数据空间
- 避免重复预处理可能引入的误差
- 评估结果最准确，与训练指标可比

#### 步骤：

**Step 1: 找到预处理后的数据**

```bash
# 预处理后的数据位置
PREPROCESSED_DIR="/home/huawei/public/home/langdy/U-Mamba/data/nnUNet_preprocessed/Dataset800_BrainMRI/nnUNetPlans_3d_fullres"

# 查看预处理数据
ls ${PREPROCESSED_DIR}/

# 应该看到类似这样的结构：
# dataset.json
# dataset_fingerprint.json
# gt_segmentations/  （这个文件夹的"分割"实际上是重建标签，即原始图像）
# *.npz              （预处理后的图像文件）
```

**Step 2: 运行预测（使用预处理数据）**

```bash
# 使用简单预测脚本
python simple_predict_reconstruction.py \
    --checkpoint /home/huawei/public/home/langdy/U-Mamba/data/nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoderReconstruction__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth \
    --input ${PREPROCESSED_DIR} \
    --output ./reconstructions_preprocessed \
    --use_preprocessed \
    --device cuda
```

**Step 3: 评估（使用预处理空间的原始图像）**

```bash
# 评估时，原始图像也应该使用预处理后的版本
# 预处理后的"标签"实际上就是原始图像的预处理版本

python evaluate_brain_encoder.py \
    --mode simple \
    --original_dir ${PREPROCESSED_DIR}/gt_segmentations \
    --predicted_dir ./reconstructions_preprocessed \
    --output_dir ./evaluation_preprocessed \
    --num_vis 20
```

**预期结果：**
- PSNR应该接近训练时的val_psnr（~23 dB）
- MAE应该接近训练时的val_mae（~0.011）

---

### 方案2：使用原始图像（不推荐，仅供参考）

如果你一定要使用原始图像格式（.nii.gz），可以用这个方法，但结果可能不如方案1准确。

```bash
# 使用原始图像
python simple_predict_reconstruction.py \
    --checkpoint /path/to/checkpoint_best.pth \
    --input /home/huawei/public/home/langdy/U-Mamba/data/nnUNet_raw/Dataset800_BrainMRI/imagesTr \
    --output ./reconstructions_raw \
    --device cuda
    # 注意：不加 --use_preprocessed 标志
```

**问题：**
- 脚本内部的简单预处理可能与nnUNet的预处理不完全一致
- 结果可能仍有偏差

---

## 📊 验证修复是否成功

### 检查1：预测输出是否为连续值

```python
import nibabel as nib
import numpy as np

# 加载一个预测结果
pred = nib.load('reconstructions_preprocessed/case_001.nii.gz').get_fdata()

print(f"Value range: [{pred.min():.4f}, {pred.max():.4f}]")
print(f"Unique values: {len(np.unique(pred))}")

# ✓ 正确：连续值，unique values很多（>1000）
# ✗ 错误：只有几个离散值（如0, 1, 2）
```

### 检查2：评估指标是否合理

运行评估后，检查指标：

```bash
cat evaluation_preprocessed/evaluation_results.json
```

**期望的指标范围：**
- PSNR: 20-25 dB（接近训练时的23 dB）
- SSIM: 0.7-0.9
- MAE: 0.01-0.02（接近训练时的0.011）

### 检查3：视觉对比

```bash
# 查看可视化结果
ls evaluation_preprocessed/visualizations/

# 应该看到原图和重建图很相似
# 差异热图应该显示较小的误差（冷色为主）
```

---

## 🔬 深入理解：数据空间问题

### nnUNet的数据处理流程

```
原始数据 (.nii.gz)
    ↓ [预处理]
    ├─ 重采样到统一spacing
    ├─ 裁剪/padding到合适大小
    ├─ 强度归一化 (Z-score)
    └─ 保存为.npz
预处理数据 (.npz)
    ↓ [训练]
训练和验证 ← 在这个空间计算指标
    ↓ [预测]
预测输出
    ↓ [后处理]
    ├─ 重采样回原始spacing
    ├─ 裁剪/padding回原始大小
    └─ 保存为.nii.gz
最终输出 (.nii.gz)
```

### 问题出在哪里？

**标准nnUNetv2_predict的流程：**
```python
# 1. 预处理输入
preprocessed = preprocess(raw_image)

# 2. 模型预测
output = model(preprocessed)

# 3. 【问题在这里！】分割后处理
output = softmax(output)           # 转为概率
output = argmax(output, axis=0)    # 取最大值索引 → 变成离散label！
output = map_to_labels(output)     # 映射到label ID (0, 1, 2, ...)

# 4. 后处理回原始空间
final = postprocess(output, original_properties)
```

**重建任务应该的流程：**
```python
# 1. 预处理输入
preprocessed = preprocess(raw_image)

# 2. 模型预测
output = model(preprocessed)

# 3. 【关键修改！】保持连续值
if output.shape[0] > 1:
    output = output.mean(axis=0)   # 如果多通道，取平均
# 不做argmax！直接使用连续值！

# 4. （可选）后处理回原始空间
final = postprocess(output, original_properties)
```

---

## 🛠️ 脚本说明

我创建了两个预测脚本：

### 1. `simple_predict_reconstruction.py` （推荐）⭐

**特点：**
- 简单直接，易于理解
- 直接加载模型和数据
- 不依赖nnUNet的复杂预测流程
- 支持预处理数据（.npz）和原始数据（.nii.gz）

**使用方法：**
```bash
python simple_predict_reconstruction.py \
    --checkpoint /path/to/checkpoint_best.pth \
    --input /path/to/preprocessed/data \
    --output ./output \
    --use_preprocessed \
    --device cuda
```

### 2. `predict_reconstruction.py` （高级）

**特点：**
- 基于nnUNetPredictor重写
- 支持更多nnUNet特性（TTA, 多fold ensemble等）
- 更复杂但功能更全

**使用方法：**
```bash
python predict_reconstruction.py \
    --checkpoint /path/to/checkpoint_best.pth \
    --input /path/to/input \
    --output ./output \
    --mode preprocessed \
    --device cuda
```

---

## 📋 完整的评估流程（推荐）

### Step 1: 准备环境

```bash
cd /home/huawei/public/home/langdy/U-Mamba

# 设置路径变量
CHECKPOINT="/home/huawei/public/home/langdy/U-Mamba/data/nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoderReconstruction__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth"
PREPROCESSED_DIR="/home/huawei/public/home/langdy/U-Mamba/data/nnUNet_preprocessed/Dataset800_BrainMRI/nnUNetPlans_3d_fullres"
```

### Step 2: 运行预测

```bash
# 使用预处理数据预测
CUDA_VISIBLE_DEVICES=0 python simple_predict_reconstruction.py \
    --checkpoint ${CHECKPOINT} \
    --input ${PREPROCESSED_DIR} \
    --output ./reconstructions \
    --use_preprocessed \
    --device cuda
```

### Step 3: 评估结果

```bash
# 评估重建质量
python evaluate_brain_encoder.py \
    --mode simple \
    --original_dir ${PREPROCESSED_DIR}/gt_segmentations \
    --predicted_dir ./reconstructions \
    --output_dir ./evaluation_results \
    --num_vis 20
```

### Step 4: 查看结果

```bash
# 查看定量结果
cat ./evaluation_results/evaluation_results.json

# 查看可视化
ls ./evaluation_results/visualizations/

# 查看指标分布
ls ./evaluation_results/metrics_distribution.png
```

---

## ❓ 常见问题

### Q1: 找不到预处理数据？

**检查路径：**
```bash
# nnUNet预处理数据的标准位置
ls $nnUNet_preprocessed/Dataset800_BrainMRI/

# 如果没有，检查环境变量
echo $nnUNet_preprocessed

# 或者查找
find /home/huawei/public/home/langdy/U-Mamba -name "nnUNet_preprocessed" -type d
```

### Q2: gt_segmentations文件夹是什么？

这个文件夹包含的是预处理后的"标签"，但对于重建任务，**标签就是原始图像本身**。

所以 `gt_segmentations/` 里的文件实际上就是预处理后的原始图像，可以用来做对比。

### Q3: 预处理数据格式是什么？

```python
import numpy as np

# 加载一个.npz文件
data = np.load('case_001.npz')

print(data.files)  # ['data', 'properties']

image = data['data']      # [C, D, H, W] - 预处理后的图像
props = data['properties'] # 包含原始spacing等信息
```

### Q4: 如果我想评估原始空间的结果怎么办？

需要将预测结果重采样回原始空间：

```python
# 这需要使用nnUNet的后处理功能
# 或者使用SimpleITK手动重采样

# 简单方法：保留当前评估（在预处理空间）
# 因为训练也是在预处理空间进行的，所以评估应该在同一空间
```

### Q5: 为什么不直接修改nnUNetv2_predict？

- nnUNetv2_predict是nnUNet框架的标准工具，专门设计用于分割任务
- 修改它需要改动框架核心代码，维护困难
- 创建独立的重建预测脚本更清晰、更易于理解和维护

---

## ✅ 成功检查清单

运行完整流程后，确认以下几点：

- [ ] 预测脚本成功运行，无错误
- [ ] 生成的重建图像是连续值（不是离散label）
- [ ] PSNR接近训练时的val_psnr（±2-3 dB是正常的）
- [ ] MAE接近训练时的val_mae
- [ ] 可视化结果显示重建图像与原图相似
- [ ] 差异热图显示误差较小

如果以上都满足，说明修复成功！ 🎉

---

## 📞 进一步帮助

如果仍有问题：

1. **检查错误日志**：完整的错误信息有助于诊断
2. **验证数据路径**：确保所有路径都正确
3. **检查checkpoint**：确认使用的是 `checkpoint_best.pth`
4. **尝试单个样本**：先测试一个文件，确认流程正确

---

**总结：使用预处理数据 + 新的预测脚本 = 正确的评估结果** ✓
