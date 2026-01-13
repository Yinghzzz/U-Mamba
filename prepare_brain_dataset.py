"""
数据集准备脚本：将你的3D脑MRI数据转换为nnU-Net格式
"""
import os
import shutil
from pathlib import Path
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw

def prepare_brain_dataset(
    source_image_dir: str,
    source_label_dir: str = None,
    dataset_id: int = 800,
    dataset_name: str = "BrainMRI"
):
    """
    准备脑MRI数据集

    参数:
        source_image_dir: 原始图像目录（包含.nii.gz文件）
        source_label_dir: 原始标签目录（可选，如果没有标签可以设为None）
        dataset_id: 数据集ID（建议使用800+）
        dataset_name: 数据集名称
    """
    # 1. 创建目标目录
    foldername = f"Dataset{dataset_id:03d}_{dataset_name}"
    out_base = os.path.join(nnUNet_raw, foldername)
    imagestr = os.path.join(out_base, "imagesTr")
    labelstr = os.path.join(out_base, "labelsTr")

    os.makedirs(imagestr, exist_ok=True)
    os.makedirs(labelstr, exist_ok=True)

    print(f"创建数据集目录: {out_base}")

    # 2. 复制和重命名图像文件
    source_images = sorted(Path(source_image_dir).glob("*.nii.gz"))
    print(f"找到 {len(source_images)} 个图像文件")

    for idx, img_path in enumerate(source_images, start=1):
        # 生成标准化的文件名
        case_id = f"brain_{idx:04d}"
        target_img = os.path.join(imagestr, f"{case_id}_0000.nii.gz")

        # 复制文件
        shutil.copy(str(img_path), target_img)

        if idx % 10 == 0:
            print(f"已处理 {idx}/{len(source_images)} 个图像")

    # 3. 复制标签文件（如果有）
    has_labels = False
    if source_label_dir and os.path.exists(source_label_dir):
        source_labels = sorted(Path(source_label_dir).glob("*.nii.gz"))
        print(f"找到 {len(source_labels)} 个标签文件")

        if len(source_labels) == len(source_images):
            for idx, label_path in enumerate(source_labels, start=1):
                case_id = f"brain_{idx:04d}"
                target_label = os.path.join(labelstr, f"{case_id}.nii.gz")
                shutil.copy(str(label_path), target_label)
            has_labels = True
            print("标签文件复制完成")
        else:
            print(f"警告: 标签数量({len(source_labels)})与图像数量({len(source_images)})不匹配")

    # 4. 创建dataset.json
    if has_labels:
        # 如果有标签，假设是脑区分割任务
        labels = {
            'background': 0,
            'brain': 1,  # 根据你的实际标签调整
        }
    else:
        # 如果没有标签，使用虚拟标签（用于自监督学习）
        labels = {
            'background': 0,
            'foreground': 1,
        }
        print("警告: 未找到标签，将创建虚拟标签配置")
        print("建议: 考虑使用重建任务或对比学习来训练encoder")

    generate_dataset_json(
        output_folder=out_base,
        channel_names={0: 'MRI'},  # 如果有多模态，可以添加更多通道
        labels=labels,
        num_training_cases=len(source_images),
        file_ending='.nii.gz',
        dataset_name=foldername,
        description=f'3D Brain MRI dataset for cross-modal generation (brain-to-face)',
        reference='Your research project',
    )

    print(f"\n数据集准备完成!")
    print(f"数据集位置: {out_base}")
    print(f"训练样本数: {len(source_images)}")
    print(f"下一步: 运行预处理 - nnUNetv2_plan_and_preprocess -d {dataset_id}")

    return out_base


if __name__ == '__main__':
    # ========== 配置区域 - 请修改这些路径 ==========

    # 你的原始数据路径
    SOURCE_IMAGE_DIR = "/path/to/your/brain/images"  # 修改为你的图像目录
    SOURCE_LABEL_DIR = None  # 如果有标签，修改为标签目录；否则保持None

    # 数据集配置
    DATASET_ID = 800  # 你的数据集ID
    DATASET_NAME = "BrainMRI"  # 数据集名称

    # =============================================

    prepare_brain_dataset(
        source_image_dir=SOURCE_IMAGE_DIR,
        source_label_dir=SOURCE_LABEL_DIR,
        dataset_id=DATASET_ID,
        dataset_name=DATASET_NAME
    )
