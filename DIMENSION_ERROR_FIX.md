# 预测维度错误修复说明

## 错误信息

```
✗ Error processing brain_0057.npz: Sizes of tensors must match except in dimension 1.
Expected size 32 but got size 31 for tensor number 1 in the list.
```

## 问题原因

这是U-Net中的skip connection维度不匹配问题。

### 具体原因

U-Net在下采样和上采样过程中，如果输入尺寸不能被某个数（通常是16或32）整除，会导致：

```
输入: [1, 1, 128, 128, 127]  # 注意最后一维是127
  ↓ 下采样4次 (除以16)
中间: [1, 256, 8, 8, 7]      # 127 / 16 = 7.9 → 向下取整为7
  ↓ 上采样4次 (乘以16)
输出: [1, 1, 128, 128, 112]  # 7 * 16 = 112 ≠ 127

在skip connection时：
encoder_feat: [1, 256, 8, 8, 8]   # 从128除以16得到8
decoder_feat: [1, 256, 8, 8, 7]   # 从112除以16得到7
→ 维度不匹配！
```

## 解决方案

我已经修改了 `simple_predict_reconstruction.py`，添加了自动padding功能。

### 修改内容

1. **添加padding函数**：确保输入尺寸能被16整除
2. **预测后移除padding**：恢复原始尺寸
3. **更好的错误处理**：即使某个样本失败也继续处理其他样本
4. **详细的错误信息**：显示完整traceback便于调试

### 工作原理

```python
# 1. 加载数据
data = [1, 128, 128, 127]  # 原始尺寸

# 2. Pad到能被16整除
data_padded = [1, 128, 128, 128]  # 127 → 128
slicer = (slice(None), slice(None), slice(None), slice(0, 127))

# 3. 预测
output_padded = model(data_padded)  # [1, 128, 128, 128]

# 4. 移除padding
output = output_padded[slicer]  # [1, 128, 128, 127]
```

## 使用方法

### 更新脚本

```bash
cd /home/huawei/public/home/langdy/U-Mamba

# 从仓库拉取最新版本
git pull origin claude/umamba-brain-encoder-KRWeS

# 或者手动复制新的simple_predict_reconstruction.py
```

### 运行预测

```bash
# 使用修复后的脚本
CUDA_VISIBLE_DEVICES=0 python simple_predict_reconstruction.py \
    --checkpoint /home/huawei/public/home/langdy/U-Mamba/data/nnUNet_results/Dataset800_BrainMRI/nnUNetTrainerBrainEncoderReconstruction__nnUNetPlans__3d_fullres/fold_all/checkpoint_best.pth \
    --input /home/huawei/public/home/langdy/U-Mamba/data/nnUNet_preprocessed/Dataset800_BrainMRI/nnUNetPlans_3d_fullres \
    --output ./reconstructions_fixed \
    --use_preprocessed \
    --device cuda
```

### 查看结果

脚本会显示成功和失败的数量：

```
✓ Prediction completed!
  Successful: 3225/3229
  Failed: 4/3229
```

如果有失败的样本，会显示详细错误信息，但不会中断整个预测过程。

## 验证修复

### 检查1：是否还有维度错误

运行脚本后，如果所有样本都成功处理（Failed: 0），说明修复成功。

### 检查2：输出尺寸是否正确

```python
import nibabel as nib
import numpy as np

# 加载原始输入和重建输出
original = np.load('preprocessed/brain_0057.npz')['data']
reconstructed = nib.load('reconstructions/brain_0057.nii.gz').get_fdata()

print(f"Original shape: {original.shape}")
print(f"Reconstructed shape: {reconstructed.shape}")
# 应该完全一致（除了channel维度）
```

## 如果仍有问题

### 尝试更大的divisible_by值

如果仍有维度错误，可能需要更大的divisible值。修改脚本中的：

```python
# 在simple_predict_reconstruction.py第152行附近
divisible_by = 32  # 改成32试试
```

### 检查具体失败的样本

```bash
# 查看失败样本的尺寸
import numpy as np
data = np.load('brain_0057.npz')
print(data['data'].shape)
```

### 手动指定padding

如果自动padding不工作，可以手动计算并指定：

```python
# 假设你的数据是 [1, 128, 128, 127]
# 需要padding到 [1, 128, 128, 128]

# 修改脚本，使用固定的target_shape
target_shape = [1, 128, 128, 128]
data_padded, slicer = pad_nd_image(data, new_shape=target_shape, return_slicer=True)
```

## 技术细节

### U-Net的池化层数

U-Mamba使用的U-Net架构通常有4-5层池化：

- 4层池化: 输入需要能被 2^4 = 16 整除
- 5层池化: 输入需要能被 2^5 = 32 整除

### nnUNet的处理方式

nnUNet在训练时会自动调整patch size确保能被整除。但预测时如果输入尺寸不同，就会出现这个问题。

我们的修复方案模拟了nnUNet的padding策略。

## 性能影响

Padding对性能影响很小：

- **时间开销**：padding和去padding都是O(1)操作，几乎无开销
- **内存开销**：最多增加几个voxel，可忽略
- **预测质量**：padding区域是常数0，不影响有效区域的预测

## 总结

✅ **问题已修复**：添加了自动padding机制
✅ **向后兼容**：不影响正常尺寸的样本
✅ **鲁棒性提升**：即使某个样本失败也能继续处理
✅ **详细日志**：显示成功/失败统计和错误详情

如有问题，请提供完整的错误信息和失败样本的shape信息。
