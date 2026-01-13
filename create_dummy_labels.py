"""
为没有标签的脑MRI数据集创建虚拟标签
用于通过nnU-Net的数据验证
"""
import os
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm


def create_dummy_labels(image_dir, label_dir, method='foreground'):
    """
    创建虚拟标签文件

    参数:
        image_dir: 图像目录 (imagesTr)
        label_dir: 标签输出目录 (labelsTr)
        method: 标签生成方法
            - 'zeros': 全零标签（背景）
            - 'ones': 全一标签（前景）
            - 'foreground': 基于强度阈值的前景标签（推荐）
    """
    os.makedirs(label_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = sorted(Path(image_dir).glob("*_0000.nii.gz"))
    print(f"找到 {len(image_files)} 个图像文件")

    if len(image_files) == 0:
        print("错误: 没有找到图像文件")
        return

    print(f"使用方法: {method}")
    print("开始创建虚拟标签...")

    for img_path in tqdm(image_files, desc="创建标签"):
        # 读取图像
        img = sitk.ReadImage(str(img_path))
        img_array = sitk.GetArrayFromImage(img)

        # 生成标签
        if method == 'zeros':
            # 全零标签
            label_array = np.zeros_like(img_array, dtype=np.uint8)

        elif method == 'ones':
            # 全一标签
            label_array = np.ones_like(img_array, dtype=np.uint8)

        elif method == 'foreground':
            # 基于阈值的前景分割（简单但有效）
            # 假设背景是接近0的值
            threshold = np.percentile(img_array[img_array > 0], 10) if np.any(img_array > 0) else 0
            label_array = (img_array > threshold).astype(np.uint8)

        else:
            raise ValueError(f"未知的方法: {method}")

        # 创建标签图像
        label_img = sitk.GetImageFromArray(label_array)
        label_img.CopyInformation(img)  # 复制spacing、origin等信息

        # 保存标签
        # 从 brain_0001_0000.nii.gz 变成 brain_0001.nii.gz
        case_id = img_path.name.replace('_0000.nii.gz', '.nii.gz')
        label_path = Path(label_dir) / case_id

        sitk.WriteImage(label_img, str(label_path))

    print(f"\n✓ 完成! 创建了 {len(image_files)} 个标签文件")
    print(f"标签目录: {label_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='为脑MRI数据创建虚拟标签')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像目录 (imagesTr)')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='标签输出目录 (labelsTr)')
    parser.add_argument('--method', type=str, default='foreground',
                       choices=['zeros', 'ones', 'foreground'],
                       help='标签生成方法')

    args = parser.parse_args()

    create_dummy_labels(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        method=args.method
    )
