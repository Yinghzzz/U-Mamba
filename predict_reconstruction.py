#!/usr/bin/env python3
"""
重建任务的预测脚本
专门用于nnUNetTrainerBrainEncoderReconstruction

与标准nnUNetv2_predict的区别：
1. 不使用softmax/argmax（保持连续值）
2. 直接输出重建结果（不做分割后处理）
3. 正确处理多通道输出（取平均）
"""

import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import argparse
from typing import Union, Tuple
import json

from batchgenerators.utilities.file_and_folder_operations import load_json, join, isfile, maybe_mkdir_p, subfiles
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


class ReconstructionPredictor(nnUNetPredictor):
    """
    重建任务的预测器
    重写关键方法以适配连续值输出
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("=" * 80)
        print("Reconstruction Predictor Initialized")
        print("This predictor is designed for reconstruction tasks.")
        print("Output will be continuous values (not segmentation labels).")
        print("=" * 80)

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                  segmentation_previous_stage: np.ndarray = None,
                                  output_file_truncated: str = None,
                                  save_or_return_probabilities: bool = False):
        """
        重写预测方法以处理重建任务
        """
        # 调用父类方法获取预测结果
        # 注意：这会得到网络的原始输出
        ppa = self.configuration_manager.patch_size
        if 'network_input_must_be_divisible_by' in self.plans_manager.plans['configurations'][self.configuration]:
            current_spacing = self.configuration_manager.spacing if 'spacing' in self.configuration_manager.__dict__.keys() else image_properties['spacing']
            ppa = self.plans_manager.plans['configurations'][self.configuration]['network_input_must_be_divisible_by']

        data, slicer = pad_nd_image(input_image, ppa, 'constant', {'constant_values': 0}, True, None)

        # 预测
        predicted_logits = self.predict_logits_from_preprocessed_data(data).cpu()

        # 移除padding
        predicted_logits = predicted_logits[tuple([slice(None), *slicer])]

        # 关键修改：对于重建任务，直接使用输出（不做softmax/argmax）
        # 如果是多通道输出，取平均
        if predicted_logits.shape[0] > 1:
            print(f"  Multi-channel output detected ({predicted_logits.shape[0]} channels), taking mean...")
            reconstruction = predicted_logits.mean(0, keepdim=True)
        else:
            reconstruction = predicted_logits

        # 返回连续值（不做argmax）
        return reconstruction.numpy()[0]  # [H, W, D]

    def predict_from_files(self, list_of_lists_or_source_folder: Union[str, list],
                           output_folder_or_list_of_truncated_output_files: Union[str, list, None],
                           save_probabilities: bool = False, overwrite: bool = True,
                           num_processes_preprocessing: int = 4, num_processes_segmentation_export: int = 4,
                           folder_with_segs_from_prev_stage: str = None, num_parts: int = 1, part_id: int = 0):
        """
        重写预测入口方法
        """
        # 参数处理
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        else:
            output_folder = None

        # 获取输入文件列表
        if isinstance(list_of_lists_or_source_folder, str):
            folder = list_of_lists_or_source_folder
            case_ids = [i[:-len('_0000.nii.gz')] if i.endswith('_0000.nii.gz') else i[:-len('.nii.gz')]
                       for i in subfiles(folder, suffix='.nii.gz', join=False)]
            input_files = [[join(folder, i + '_0000.nii.gz')] for i in case_ids]
        else:
            input_files = list_of_lists_or_source_folder
            case_ids = [Path(i[0]).stem.replace('_0000', '') for i in input_files]

        # 创建输出目录
        if output_folder is not None:
            maybe_mkdir_p(output_folder)

        # 预测每个case
        print(f"\nPredicting {len(case_ids)} cases...")
        print("=" * 80)

        for idx, (case_id, input_file_list) in enumerate(tqdm(list(zip(case_ids, input_files)),
                                                               desc="Processing cases")):
            if output_folder is not None:
                output_file = join(output_folder, case_id + '.nii.gz')
            else:
                output_file = output_folder_or_list_of_truncated_output_files[idx]

            # 跳过已存在的文件
            if isfile(output_file) and not overwrite:
                continue

            # 预测单个case
            self.predict_from_list_of_npy_arrays(
                [input_file_list],
                [output_file],
                [None],  # segs_from_prev_stage
                [None],  # properties
                num_processes_segmentation_export
            )

        print("=" * 80)
        print("Prediction completed!")


def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False, shape_must_be_divisible_by=None):
    """
    从nnUNet复制的padding函数
    """
    if kwargs is None:
        kwargs = {}

    if new_shape is not None:
        old_shape = np.array(image.shape)
        new_shape = np.array(new_shape)
    elif shape_must_be_divisible_by is not None:
        old_shape = np.array(image.shape)
        new_shape = old_shape + (shape_must_be_divisible_by - old_shape % shape_must_be_divisible_by) % shape_must_be_divisible_by
    else:
        return image

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference - pad_below

    pad_list = [[int(pad_below[i]), int(pad_above[i])] for i in range(len(old_shape))]

    res = np.pad(image, pad_list, mode, **kwargs)

    if return_slicer:
        slicer = tuple([slice(int(pad_below[i]), int(pad_below[i] + old_shape[i])) for i in range(len(old_shape))])
        return res, slicer
    else:
        return res


def simple_reconstruction_predict(
    checkpoint_path: str,
    input_folder: str,
    output_folder: str,
    device: str = 'cuda',
    use_folds: tuple = (0, 1, 2, 3, 4),
    use_mirroring: bool = False,
    verbose: bool = True
):
    """
    简化的重建预测函数

    参数:
        checkpoint_path: checkpoint文件路径或包含checkpoint的文件夹
        input_folder: 输入图像文件夹（预处理后的或原始的）
        output_folder: 输出文件夹
        device: 'cuda' or 'cpu'
        use_folds: 使用哪些fold（对于fold_all训练的模型可以只用(None,)）
        use_mirroring: 是否使用test-time augmentation镜像
    """
    from nnunetv2.utilities.plans_handling import PlansManager
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class

    print("\n" + "=" * 80)
    print("Reconstruction Prediction")
    print("=" * 80)

    # 初始化预测器
    predictor = ReconstructionPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=use_mirroring,
        perform_everything_on_device=True,
        device=torch.device(device),
        verbose=verbose,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # 确定checkpoint文件
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_dir():
        # 如果是文件夹，查找checkpoint_best.pth
        checkpoint_file = checkpoint_file / "checkpoint_best.pth"
        if not checkpoint_file.exists():
            checkpoint_file = Path(checkpoint_path) / "checkpoint_final.pth"

    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    print(f"\nLoading checkpoint: {checkpoint_file}")

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))

    # 从checkpoint获取配置
    if 'trainer_name' in checkpoint.keys():
        trainer_name = checkpoint['trainer_name']
    else:
        trainer_name = 'nnUNetTrainerBrainEncoderReconstruction'

    # 初始化预测器（从checkpoint文件夹）
    predictor.initialize_from_trained_model_folder(
        str(checkpoint_file.parent.parent),  # 模型文件夹
        use_folds=use_folds,
        checkpoint_name=checkpoint_file.name.replace('.pth', '')
    )

    print(f"Predictor initialized successfully")
    print(f"  Trainer: {trainer_name}")
    print(f"  Configuration: {predictor.configuration}")
    print(f"  Device: {device}")
    print(f"  Use mirroring: {use_mirroring}")

    # 创建输出目录
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 执行预测
    print(f"\nInput folder: {input_folder}")
    print(f"Output folder: {output_folder}")

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_folder,
        output_folder_or_list_of_truncated_output_files=output_folder,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=4,
        num_processes_segmentation_export=4
    )

    print("\n" + "=" * 80)
    print("✓ Prediction completed successfully!")
    print("=" * 80)


def predict_from_preprocessed(
    checkpoint_path: str,
    preprocessed_folder: str,
    output_folder: str,
    dataset_id: int = 800,
    device: str = 'cuda',
    batch_size: int = 1
):
    """
    直接从预处理后的数据预测（更准确）

    这个方法直接使用nnUNet预处理后的.npz文件，
    避免了重复预处理可能引入的误差
    """
    print("\n" + "=" * 80)
    print("Reconstruction Prediction from Preprocessed Data")
    print("=" * 80)

    # 加载checkpoint和模型
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.is_dir():
        checkpoint_file = checkpoint_file / "checkpoint_best.pth"

    print(f"Loading checkpoint: {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location='cpu')

    # 获取模型配置
    plans = checkpoint['init_args']['plans']
    configuration = checkpoint['init_args']['configuration']

    # 实例化trainer以获取网络
    from umamba.nnunetv2.training.nnUNetTrainer.nnUNetTrainerBrainEncoderReconstruction import nnUNetTrainerBrainEncoderReconstruction

    # 创建临时trainer来加载模型
    trainer = nnUNetTrainerBrainEncoderReconstruction(
        plans=plans,
        configuration=configuration,
        fold=0,  # 不重要，只是为了初始化
        dataset_json=checkpoint['init_args']['dataset_json'],
        unpack_dataset=False,
        device=torch.device(device)
    )

    # 初始化网络
    trainer.initialize()

    # 加载权重
    trainer.network.load_state_dict(checkpoint['network_weights'])
    trainer.network.eval()

    print(f"Model loaded successfully")
    print(f"  Configuration: {configuration}")
    print(f"  Device: {device}")

    # 查找预处理后的文件
    preprocessed_path = Path(preprocessed_folder)
    npz_files = sorted(list(preprocessed_path.glob("*.npz")))

    if len(npz_files) == 0:
        # 尝试在子文件夹中查找
        npz_files = sorted(list(preprocessed_path.glob("**/*.npz")))

    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz files found in {preprocessed_folder}")

    print(f"\nFound {len(npz_files)} preprocessed files")

    # 创建输出目录
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 预测每个文件
    with torch.no_grad():
        for npz_file in tqdm(npz_files, desc="Predicting"):
            # 加载预处理数据
            data_dict = np.load(npz_file)
            data = data_dict['data']  # [C, H, W, D]

            # 转为tensor
            data_tensor = torch.from_numpy(data).float().unsqueeze(0).to(device)  # [1, C, H, W, D]

            # 预测
            output = trainer.network(data_tensor)  # [1, C_out, H, W, D]

            # 处理输出
            output = output.squeeze(0).cpu().numpy()  # [C_out, H, W, D]

            # 如果是多通道，取平均
            if output.shape[0] > 1:
                output = output.mean(axis=0, keepdims=True)  # [1, H, W, D]

            # 保存结果
            output_file = output_path / npz_file.name.replace('.npz', '.nii.gz')

            # 使用原始的properties
            if 'properties' in data_dict.files:
                properties = data_dict['properties'].item()
            else:
                properties = None

            # 保存为NIfTI
            # 注意：这里保存的是预处理空间的重建结果
            # 如果需要转回原始空间，需要使用nnUNet的后处理
            nii_img = nib.Nifti1Image(output[0], affine=np.eye(4))
            nib.save(nii_img, str(output_file))

    print("\n" + "=" * 80)
    print("✓ Prediction completed successfully!")
    print(f"✓ Results saved to: {output_folder}")
    print("=" * 80)
    print("\n⚠ Note: Results are in preprocessed space.")
    print("For comparison with original images, they need to be resampled back.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict reconstructions using trained U-Mamba encoder")

    parser.add_argument("--checkpoint", "-chk", type=str, required=True,
                       help="Path to checkpoint file or folder containing checkpoint")
    parser.add_argument("--input", "-i", type=str, required=True,
                       help="Input folder with images")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output folder for predictions")
    parser.add_argument("--mode", type=str, choices=['standard', 'preprocessed'], default='standard',
                       help="Prediction mode: 'standard' (from raw images) or 'preprocessed' (from .npz)")
    parser.add_argument("--device", type=str, default='cuda',
                       help="Device: 'cuda' or 'cpu'")
    parser.add_argument("--dataset_id", "-d", type=int, default=800,
                       help="Dataset ID (for finding preprocessed data)")
    parser.add_argument("--use_mirroring", action='store_true',
                       help="Use test-time augmentation (mirroring)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for prediction")

    args = parser.parse_args()

    if args.mode == 'standard':
        simple_reconstruction_predict(
            checkpoint_path=args.checkpoint,
            input_folder=args.input,
            output_folder=args.output,
            device=args.device,
            use_folds=(None,),  # 使用fold_all训练的模型
            use_mirroring=args.use_mirroring
        )
    else:  # preprocessed
        predict_from_preprocessed(
            checkpoint_path=args.checkpoint,
            preprocessed_folder=args.input,
            output_folder=args.output,
            dataset_id=args.dataset_id,
            device=args.device,
            batch_size=args.batch_size
        )
