# 脑MRI重建模型评估指南

训练150个epochs后，你可以从多个维度评估模型性能。

## 🎯 评估方法总览

### 1️⃣ 查看训练曲线（最快速）
### 2️⃣ 定量评估重建质量（推荐）
### 3️⃣ 可视化重建结果
### 4️⃣ 提取和分析encoder特征

---

## 1️⃣ 查看训练曲线

### 查看progress.png
```bash
# 找到你的训练输出目录
RESULTS_DIR="/path/to/nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoderReconstruction__nnUNetPlans__3d_fullres/fold_all"

# 查看训练可视化
eog ${RESULTS_DIR}/progress.png
# 或者
display ${RESULTS_DIR}/progress.png
```

**查看要点**：
- ✅ `train_loss` 和 `val_loss` 应该持续下降
- ✅ `val_psnr` 应该持续上升（>30 dB 为好）
- ✅ `val_mae` 应该持续下降（<0.05 为好）
- ⚠️ 如果曲线平稳，说明模型已经收敛

### 查看训练日志
```bash
# 查看最近的训练日志
tail -n 100 ${RESULTS_DIR}/training_log*.txt

# 查找关键指标
grep -E "train_loss|val_loss|val_psnr|val_mae" ${RESULTS_DIR}/training_log*.txt | tail -n 20
```

**期望值**：
- PSNR: >30 dB（优秀），25-30 dB（良好）
- MAE: <0.05（优秀），0.05-0.1（良好）
- Loss: 持续下降并趋于稳定

---

## 2️⃣ 定量评估（推荐方法）

### 方法A：使用验证集快速评估

#### Step 1: 生成重建结果
```bash
# 设置路径
CHECKPOINT="${RESULTS_DIR}/checkpoint_best.pth"  # 使用最佳checkpoint
INPUT_DIR="/path/to/nnUNet_raw/Dataset800_BrainMRI/imagesTs"  # 测试集
OUTPUT_DIR="./evaluation_output"

# 运行预测（生成重建结果）
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
    -i ${INPUT_DIR} \
    -o ${OUTPUT_DIR}/predictions \
    -d 800 \
    -c 3d_fullres \
    -tr nnUNetTrainerBrainEncoderReconstruction \
    -chk ${CHECKPOINT} \
    -f all \
    --disable_tta
```

#### Step 2: 评估重建质量
```bash
# 使用评估脚本计算指标
python evaluate_brain_encoder.py \
    --mode simple \
    --original_dir ${INPUT_DIR} \
    --predicted_dir ${OUTPUT_DIR}/predictions \
    --output_dir ${OUTPUT_DIR}/evaluation \
    --num_vis 20
```

**输出内容**：
- ✅ 定量指标：PSNR, SSIM, MAE, MSE
- ✅ 统计信息：mean, std, median, min, max
- ✅ 可视化对比图：`evaluation/visualizations/`
- ✅ 指标分布图：`evaluation/metrics_distribution.png`
- ✅ JSON结果文件：`evaluation/evaluation_results.json`

### 方法B：使用训练集子集评估（如果没有测试集）

```bash
# 从训练集随机抽取一些样本
INPUT_DIR="/path/to/nnUNet_raw/Dataset800_BrainMRI/imagesTr"

# 其他步骤同上
```

---

## 3️⃣ 可视化重建结果

评估脚本会自动生成可视化结果在 `evaluation/visualizations/` 目录：

```bash
# 查看所有可视化结果
ls -lh ${OUTPUT_DIR}/evaluation/visualizations/

# 每个可视化包含：
# - 原始图像（3个正交平面：Axial, Coronal, Sagittal）
# - 重建图像
# - 差异热图（误差可视化）
```

**查看示例**：
```bash
# 在本地查看
eog ${OUTPUT_DIR}/evaluation/visualizations/sample_*.png

# 或者拷贝到本地电脑查看
# scp user@server:${OUTPUT_DIR}/evaluation/visualizations/*.png ./local_folder/
```

**评估要点**：
- ✅ 重建图应该与原图非常相似
- ✅ 差异热图应该显示很小的误差（颜色浅）
- ⚠️ 如果边缘模糊或细节丢失，可能需要更多训练或调整loss权重

---

## 4️⃣ 提取和分析Encoder特征

### 提取特征向量
```bash
# 使用之前创建的特征提取脚本
python extract_brain_features.py \
    --checkpoint ${CHECKPOINT} \
    --input_dir ${INPUT_DIR} \
    --output_dir ${OUTPUT_DIR}/features \
    --multi_scale \
    --pooling_method adaptive_avg \
    --batch_size 2
```

**输出**：
- 特征文件：`features/brain_features.npz`
- 包含内容：
  ```python
  data = np.load('brain_features.npz')
  features = data['features']      # [N, feature_dim]
  case_ids = data['case_ids']      # [N]
  feature_dim = data['feature_dim'] # 特征维度
  ```

### 分析特征质量

创建一个简单的特征分析脚本：

```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载特征
data = np.load('features/brain_features.npz')
features = data['features']

print(f"Feature shape: {features.shape}")
print(f"Feature dim: {data['feature_dim']}")

# 统计分析
print(f"\nFeature statistics:")
print(f"  Mean: {features.mean():.4f}")
print(f"  Std:  {features.std():.4f}")
print(f"  Min:  {features.min():.4f}")
print(f"  Max:  {features.max():.4f}")

# PCA可视化（降到2D）
pca = PCA(n_components=2)
features_2d = pca.fit_transform(features)

plt.figure(figsize=(10, 8))
plt.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Brain Features PCA Visualization')
plt.savefig('features_pca.png', dpi=150)
print(f"\nPCA variance ratio: {pca.explained_variance_ratio_}")

# 检查特征分布
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(features.flatten(), bins=100, edgecolor='black')
plt.xlabel('Feature value')
plt.ylabel('Frequency')
plt.title('Feature Distribution')

plt.subplot(1, 2, 2)
feature_norms = np.linalg.norm(features, axis=1)
plt.hist(feature_norms, bins=50, edgecolor='black')
plt.xlabel('Feature norm')
plt.ylabel('Frequency')
plt.title('Feature Norm Distribution')
plt.tight_layout()
plt.savefig('features_distribution.png', dpi=150)

print("\n✓ Feature analysis completed!")
```

---

## 📊 性能评估标准

### 重建质量指标

| 指标 | 优秀 | 良好 | 需改进 |
|------|------|------|--------|
| **PSNR** | >30 dB | 25-30 dB | <25 dB |
| **SSIM** | >0.9 | 0.8-0.9 | <0.8 |
| **MAE** | <0.05 | 0.05-0.1 | >0.1 |
| **MSE** | <0.01 | 0.01-0.05 | >0.05 |

### 特征质量指标

好的encoder特征应该：
- ✅ **表征丰富**：特征维度较高（如512-1024维）
- ✅ **分布合理**：不是全零或过度稀疏
- ✅ **区分性强**：不同样本的特征差异明显
- ✅ **稳定性好**：相似样本的特征相近

---

## 🔍 常见问题排查

### Q1: 重建质量不好怎么办？

**可能原因和解决方案**：

1. **训练不充分**
   ```bash
   # 继续训练更多epochs
   CUDA_VISIBLE_DEVICES=4,5 nnUNetv2_train 800 3d_fullres all \
       -tr nnUNetTrainerBrainEncoderReconstruction \
       -num_gpus 2 \
       --continue_training
   ```

2. **Loss权重不合适**
   - 修改 `nnUNetTrainerBrainEncoderReconstruction.py` 中的权重：
   ```python
   self.mse_weight = 1.0  # 调整MSE权重
   self.l1_weight = 0.5   # 调整L1权重
   ```

3. **需要更复杂的loss**
   - 使用高级版本：`nnUNetTrainerBrainEncoderReconstructionAdvanced`
   - 包含梯度loss，保持边缘细节

### Q2: 评估脚本报错？

**检查清单**：
- ✅ 确认checkpoint路径正确
- ✅ 确认输入目录包含 `*_0000.nii.gz` 文件
- ✅ 确认已安装依赖：`pip install scikit-image matplotlib nibabel`
- ✅ 确认GPU可用（如果使用GPU模式）

### Q3: 如何在跨模态任务中使用特征？

提取的encoder特征可以直接用于你的brain-to-face跨模态生成任务：

```python
import numpy as np

# 加载脑特征
brain_features = np.load('features/brain_features.npz')['features']

# 在你的跨模态模型中使用
# 例如：作为condition输入到生成器
def generate_face(brain_feature):
    # brain_feature: [1, feature_dim]
    # 你的生成模型
    face = your_generator(condition=brain_feature)
    return face
```

---

## 🎯 推荐评估流程（150 epochs后）

```bash
# 1. 查看训练曲线（30秒）
eog ${RESULTS_DIR}/progress.png

# 2. 如果曲线已经收敛，运行定量评估（5-10分钟）
python evaluate_brain_encoder.py \
    --mode simple \
    --original_dir /path/to/test_images \
    --predicted_dir ./predictions \
    --output_dir ./evaluation \
    --num_vis 20

# 3. 查看可视化结果（2分钟）
eog ./evaluation/visualizations/*.png

# 4. 检查定量指标（1分钟）
cat ./evaluation/evaluation_results.json

# 5. 如果满意，提取特征用于下游任务（5分钟）
python extract_brain_features.py \
    --checkpoint ${CHECKPOINT} \
    --input_dir /path/to/all_brains \
    --output_dir ./brain_features
```

**总时长：约15-20分钟**

---

## 💡 下一步建议

根据评估结果：

### 如果重建质量优秀（PSNR>30, SSIM>0.9）
✅ **提取特征用于跨模态任务**
```bash
python extract_brain_features.py --checkpoint checkpoint_best.pth ...
```

### 如果重建质量良好（PSNR 25-30）
○ **选项1**：继续训练50-100 epochs
```bash
nnUNetv2_train ... --continue_training
```

○ **选项2**：调整loss权重后重新训练
○ **选项3**：使用当前特征，在下游任务中评估效果

### 如果重建质量需要改进（PSNR<25）
✗ **需要调整训练策略**：
1. 检查数据预处理是否正确
2. 调整loss权重或使用更复杂的loss
3. 增加训练epochs
4. 调整学习率
5. 尝试使用`nnUNetTrainerBrainEncoderReconstructionAdvanced`

---

## 📝 快速命令参考

```bash
# 查看训练状态
tail -f ${RESULTS_DIR}/training_log*.txt

# 生成重建结果
nnUNetv2_predict -i INPUT -o OUTPUT -d 800 -c 3d_fullres \
    -tr nnUNetTrainerBrainEncoderReconstruction -chk checkpoint_best.pth -f all

# 评估重建质量
python evaluate_brain_encoder.py --mode simple \
    --original_dir INPUT --predicted_dir OUTPUT/predictions \
    --output_dir EVAL_OUTPUT

# 提取encoder特征
python extract_brain_features.py \
    --checkpoint checkpoint_best.pth \
    --input_dir INPUT --output_dir FEATURES

# 继续训练
nnUNetv2_train 800 3d_fullres all \
    -tr nnUNetTrainerBrainEncoderReconstruction \
    -num_gpus 2 --continue_training
```

---

**祝评估顺利！如有问题，请参考日志输出或查看详细错误信息。**
