"""
自监督重建训练器：使用U-Mamba进行脑MRI重建
让encoder学习更精细的大脑表征
"""
import torch
import torch.nn.functional as F
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBrainEncoder import nnUNetTrainerBrainEncoder
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


class nnUNetTrainerBrainEncoderReconstruction(nnUNetTrainerBrainEncoder):
    """
    自监督重建训练器

    特点:
    1. 使用重建损失（MSE + L1）而不是分割损失
    2. 目标是重建输入图像本身
    3. 让encoder学习更丰富的大脑内部结构表征
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # 重建任务不需要深度监督，禁用它
        self.enable_deep_supervision = False

        # 重建任务的权重
        self.mse_weight = 1.0      # MSE损失权重
        self.l1_weight = 0.5       # L1损失权重（鼓励稀疏性）
        self.perceptual_weight = 0.0  # 感知损失权重（可选）

        # 用于保存最佳模型的指标
        self.current_val_mae = float('inf')
        self.current_val_psnr = 0.0

    def _build_loss(self):
        """
        构建重建损失函数
        使用MSE + L1的组合
        """
        # 不使用深度监督包装，直接返回损失函数
        # 因为重建任务中，深度监督会让中间层也尝试重建，可能不太合适
        if self.enable_deep_supervision:
            # 如果启用深度监督，包装损失
            loss = DeepSupervisionWrapper(
                self._reconstruction_loss,
                weight_factors=None  # 使用默认权重
            )
        else:
            loss = self._reconstruction_loss

        return loss

    def _reconstruction_loss(self, output, target):
        """
        重建损失：MSE + L1

        参数:
            output: 模型输出 [B, C, D, H, W] - C可能>1（多类别输出）
            target: 目标图像（输入图像本身）[B, 1, D, H, W]

        返回:
            loss: 标量损失
        """
        # 如果网络输出多通道，取第一个通道用于重建
        # 或者对所有通道求平均
        if output.shape[1] > 1:
            # 方法1: 只用第一个通道
            # output = output[:, 0:1, ...]

            # 方法2: 对所有通道求平均（更稳健）
            output = torch.mean(output, dim=1, keepdim=True)

        # 确保target也是正确的形状
        if target.shape[1] != output.shape[1]:
            # target可能是[B, C, D, H, W]格式，取第一个通道
            target = target[:, 0:1, ...] if target.shape[1] > 0 else target

        # MSE损失
        mse_loss = F.mse_loss(output, target)

        # L1损失
        l1_loss = F.l1_loss(output, target)

        # 组合损失
        total_loss = self.mse_weight * mse_loss + self.l1_weight * l1_loss

        return total_loss

    def train_step(self, batch: dict) -> dict:
        """
        训练步骤
        重写以使用图像本身作为目标
        """
        data = batch['data']
        target = batch['target']  # 在重建任务中，target就是data本身

        # 前向传播
        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        # 计算输出
        output = self.network(data)

        # 计算损失
        l = self.loss(output, target)

        # 反向传播
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()

        return {'loss': l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        """
        验证步骤
        """
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            output = self.network(data)
            l = self.loss(output, target)

        # 计算额外的指标：PSNR和SSIM（可选）
        with torch.no_grad():
            # PSNR (Peak Signal-to-Noise Ratio)
            mse = F.mse_loss(output, target)
            psnr = 20 * torch.log10(target.max() / torch.sqrt(mse))

            # MAE (Mean Absolute Error)
            mae = F.l1_loss(output, target)

        return {
            'loss': l.detach().cpu().numpy(),
            'psnr': psnr.detach().cpu().numpy(),
            'mae': mae.detach().cpu().numpy()
        }

    def on_train_epoch_end(self, train_outputs: list):
        """
        训练epoch结束时的处理
        """
        # 计算平均训练损失
        losses = [o['loss'] for o in train_outputs]
        mean_loss = sum(losses) / len(losses)

        # 记录到logger（用于绘图）
        if 'train_losses' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['train_losses'] = []
        self.logger.my_fantastic_logging['train_losses'].append(mean_loss)

        self.print_to_log_file(f"train_loss {mean_loss:.4f}")

    def on_validation_epoch_end(self, val_outputs: list):
        """
        验证epoch结束时的处理
        """
        # 计算平均验证指标
        losses = [o['loss'] for o in val_outputs]
        psnrs = [o['psnr'] for o in val_outputs]
        maes = [o['mae'] for o in val_outputs]

        mean_loss = sum(losses) / len(losses)
        mean_psnr = sum(psnrs) / len(psnrs)
        mean_mae = sum(maes) / len(maes)

        # 记录到logger（用于绘图）
        if 'val_losses' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['val_losses'] = []
        if 'val_psnr' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['val_psnr'] = []
        if 'val_mae' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['val_mae'] = []

        self.logger.my_fantastic_logging['val_losses'].append(mean_loss)
        self.logger.my_fantastic_logging['val_psnr'].append(mean_psnr)
        self.logger.my_fantastic_logging['val_mae'].append(mean_mae)

        # 为了兼容绘图函数，也添加一个伪Dice指标
        if 'dice_per_class_or_region' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['dice_per_class_or_region'] = []
        # 使用归一化的PSNR作为伪Dice（用于绘图显示）
        pseudo_dice = min(mean_psnr / 50.0, 1.0)  # 将PSNR归一化到[0,1]
        self.logger.my_fantastic_logging['dice_per_class_or_region'].append([pseudo_dice])

        # 计算EMA用于保存最佳模型
        if 'ema_fg_dice' not in self.logger.my_fantastic_logging:
            self.logger.my_fantastic_logging['ema_fg_dice'] = []
        if len(self.logger.my_fantastic_logging['ema_fg_dice']) == 0:
            self.logger.my_fantastic_logging['ema_fg_dice'].append(-mean_mae)
        else:
            # 指数移动平均
            ema_alpha = 0.9
            prev_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            new_ema = ema_alpha * prev_ema + (1 - ema_alpha) * (-mean_mae)
            self.logger.my_fantastic_logging['ema_fg_dice'].append(new_ema)

        self.print_to_log_file(f"val_loss {mean_loss:.4f}")
        self.print_to_log_file(f"val_psnr {mean_psnr:.2f} dB")
        self.print_to_log_file(f"val_mae {mean_mae:.4f}")

        # 保存当前的MAE用于后续的最佳模型判断
        self.current_val_mae = mean_mae
        self.current_val_psnr = mean_psnr

    def on_epoch_end(self):
        """
        Epoch结束时的处理，包括最佳模型保存
        """
        from time import time
        from os.path import join

        self.print_to_log_file(
            f"Epoch time: {time() - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1]:.2f} s"
        )

        # 定期保存checkpoint
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, 'checkpoint_latest.pth'))

        # 保存最佳模型（基于MAE，越小越好）
        if self._best_ema is None or self.current_val_mae < self._best_ema:
            self._best_ema = self.current_val_mae
            self.print_to_log_file(f"Yayy! New best MAE: {self._best_ema:.4f}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        # 绘制训练曲线（如果需要）
        if self.local_rank == 0:
            try:
                self.logger.plot_progress_png(self.output_folder)
            except:
                pass  # 绘图失败不影响训练

        self.current_epoch += 1


class nnUNetTrainerBrainEncoderReconstructionAdvanced(nnUNetTrainerBrainEncoderReconstruction):
    """
    高级重建训练器
    添加了更多正则化和约束
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # 确保深度监督被禁用
        self.enable_deep_supervision = False

        # 高级损失权重
        self.mse_weight = 1.0
        self.l1_weight = 0.5
        self.gradient_weight = 0.1  # 梯度损失（保持边缘）
        self.feature_weight = 0.05   # 特征损失（encoder特征的一致性）

    def _reconstruction_loss(self, output, target):
        """
        高级重建损失
        """
        # 基础重建损失
        mse_loss = F.mse_loss(output, target)
        l1_loss = F.l1_loss(output, target)

        # 梯度损失（保持边缘细节）
        grad_loss = self._gradient_loss(output, target)

        # 组合
        total_loss = (
            self.mse_weight * mse_loss +
            self.l1_weight * l1_loss +
            self.gradient_weight * grad_loss
        )

        return total_loss

    def _gradient_loss(self, output, target):
        """
        梯度损失：保持图像的边缘和细节
        """
        # 计算x方向梯度
        grad_output_x = output[:, :, :, :, 1:] - output[:, :, :, :, :-1]
        grad_target_x = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]

        # 计算y方向梯度
        grad_output_y = output[:, :, :, 1:, :] - output[:, :, :, :-1, :]
        grad_target_y = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]

        # 计算z方向梯度
        grad_output_z = output[:, :, 1:, :, :] - output[:, :, :-1, :, :]
        grad_target_z = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]

        # 梯度损失
        loss_x = F.l1_loss(grad_output_x, grad_target_x)
        loss_y = F.l1_loss(grad_output_y, grad_target_y)
        loss_z = F.l1_loss(grad_output_z, grad_target_z)

        return (loss_x + loss_y + loss_z) / 3.0


# ========== 使用说明 ==========
"""
# 1. 准备数据（不需要标签，会自动使用图像作为标签）
python prepare_brain_dataset.py

# 2. 预处理
nnUNetv2_plan_and_preprocess -d 800 --verify_dataset_integrity

# 3. 训练（基础重建）
CUDA_VISIBLE_DEVICES=4,5 nnUNetv2_train 800 3d_fullres all \\
    -tr nnUNetTrainerBrainEncoderReconstruction \\
    -num_gpus 2

# 4. 训练（高级重建，包含梯度损失）
CUDA_VISIBLE_DEVICES=4,5 nnUNetv2_train 800 3d_fullres all \\
    -tr nnUNetTrainerBrainEncoderReconstructionAdvanced \\
    -num_gpus 2

# 监控训练
# 关注指标：
# - val_loss: 越低越好
# - val_psnr: 越高越好（通常>30 dB为好）
# - val_mae: 越低越好（<0.05为好）
"""
