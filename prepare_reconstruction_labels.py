"""
为重建任务准备标签
标签就是原始图像本身（用于自监督学习）
"""
import os
import shutil
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm


def prepare_reconstruction_labels(image_dir, label_dir, method='copy'):
    """
    为重建任务准备标签

    参数:
        image_dir: 图像目录 (imagesTr)
        label_dir: 标签输出目录 (labelsTr)
        method:
            - 'copy': 直接复制图像作为标签（推荐，节省空间）
            - 'normalize': 归一化后再保存（可选）
    """
    os.makedirs(label_dir, exist_ok=True)

    # 获取所有图像文件
    image_files = sorted(Path(image_dir).glob("*_0000.nii.gz"))
    print(f"找到 {len(image_files)} 个图像文件")

    if len(image_files) == 0:
        print("错误: 没有找到图像文件")
        return

    print(f"使用方法: {method}")
    print("开始准备重建标签...")

    for img_path in tqdm(image_files, desc="准备标签"):
        # 生成标签文件名
        # 从 brain_0001_0000.nii.gz 变成 brain_0001.nii.gz
        case_id = img_path.name.replace('_0000.nii.gz', '.nii.gz')
        label_path = Path(label_dir) / case_id

        if method == 'copy':
            # 直接复制图像作为标签
            shutil.copy(str(img_path), str(label_path))

        elif method == 'normalize':
            # 读取图像，归一化后保存
            img = sitk.ReadImage(str(img_path))
            img_array = sitk.GetArrayFromImage(img)

            # 归一化到[0, 1]
            img_min = img_array.min()
            img_max = img_array.max()
            if img_max > img_min:
                img_array = (img_array - img_min) / (img_max - img_min)

            # 保存
            label_img = sitk.GetImageFromArray(img_array)
            label_img.CopyInformation(img)
            sitk.WriteImage(label_img, str(label_path))

        else:
            raise ValueError(f"未知的方法: {method}")

    print(f"\n✓ 完成! 创建了 {len(image_files)} 个重建标签")
    print(f"标签目录: {label_dir}")
    print("\n说明:")
    print("- 重建任务中，模型的目标是重建输入图像本身")
    print("- 因此标签就是原始图像的副本")
    print("- 这样encoder会学习到更精细的大脑内部结构")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='为重建任务准备标签（标签=图像）')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像目录 (imagesTr)')
    parser.add_argument('--label_dir', type=str, required=True,
                       help='标签输出目录 (labelsTr)')
    parser.add_argument('--method', type=str, default='copy',
                       choices=['copy', 'normalize'],
                       help='标签生成方法')

    args = parser.parse_args()

    prepare_reconstruction_labels(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        method=args.method
    )
