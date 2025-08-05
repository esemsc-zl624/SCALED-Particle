import random
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scaled.pipelines.pipeline_ddim_scaled_particle_fluid import (
    SCALEDParticleFluidPipeline,
)
import torch
from scaled.model.unets.unet_3ds import UNet3DsModel
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(
        self,
        denoising_unet: UNet3DsModel,
    ):
        super().__init__()
        self.denoising_unet = denoising_unet

    def forward(
        self,
        noisy_latents,
        timesteps,
    ):
        model_pred = self.denoising_unet(
            noisy_latents,
            timesteps,
        ).sample
        return model_pred


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)
    snr = (alpha / sigma) ** 2
    return snr


def log_validation(
    denoising_unet,
    depth,
    height,
    width,
    scheduler,
    accelerator,
    generator=None,
    valid_dataset=None,
):
    logger.info("Running validation... ")
    if generator is None:
        generator = torch.manual_seed(42)

    dataset_len = len(valid_dataset)
    sample_idx = random.randint(0, dataset_len)
    ori_data, gt_result = valid_dataset[sample_idx]

    previous_value = ori_data[:3].unsqueeze(0).to("cuda")
    next_value = gt_result[:3].unsqueeze(0).to("cuda")
    control_value = ori_data[3:].unsqueeze(0).to("cuda")
    background_value = control_value.clone().bool()
    back_data = next_value.clone()
    back_data[:, :, 1:-1] = 1
    back_data[:, 0:1][background_value] = 0
    back_data[:, 1:2][background_value] = 0
    back_data[:, 2:3][background_value] = 0

    pipe = SCALEDUrbanFlowPipeline(
        denoising_unet,
        scheduler=scheduler,
    )
    pipe = pipe.to(accelerator.device)

    pred = pipe(
        previous_value,
        back_data,
        num_inference_steps=100,
        guidance_scale=0,
        depth=depth,
        height=height,
        width=width,
        generator=generator,
        return_dict=False,
    )

    results = {}
    # results["WithoutBackground"] = {
    #     "prediction_flow": pred.detach().cpu().numpy()[0],
    #     "gt_flow": next_value.detach().cpu().numpy()[0],
    #     "original_flow": previous_value.detach().cpu().numpy()[0],
    # }
    results["WithoutBackground"] = {
        "position_mask": pred.detach().cpu().numpy()[0],
        "future_position_mask": next_value.detach().cpu().numpy()[0],
        "original_position_mask": previous_value.detach().cpu().numpy()[0],
    }

    del pipe
    return results


def dilate_mask_square_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    3D方形膨胀：对bool类型mask进行立方体卷积，半径为radius。
    """
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    kernel = torch.ones(
        (1, 1, 2 * radius + 1, 2 * radius + 1, 2 * radius + 1), device=mask.device
    )
    out = F.conv3d(mask_f, kernel, padding=radius)
    return (out > 0).squeeze(0).squeeze(0).bool()  # 返回布尔类型 (D, H, W)


def apply_mask_on_velocity(pred, current_data, dilation_radius=5):
    """
    对预测结果进行 Top-K 粒子筛选，并根据 mask 修改速度通道。
    仅在 current_data 的 mask 膨胀区域内执行 Top-K。

    参数:
        pred: 模型预测结果，形状为 (b, 7, D, H, W)
        current_data: 当前输入数据，形状同 pred
        dilation_radius: 膨胀半径（默认5）

    返回:
        pred: 经过掩膜和速度处理后的预测结果
    """

    batch_size = pred.shape[0]

    for b in range(batch_size):
        # --- Step 1: 获取当前帧的 ground truth mask，并计算粒子个数 ---
        current_mask = current_data[b, -1]  # (D, H, W)，值为 -1 或 1
        current_mask = (current_mask + 1) / 2  # 映射为 [0, 1]
        num_particle = int(torch.sum(current_mask).item())

        # --- Step 2: 计算膨胀 mask ---
        dilated_mask = dilate_mask_square_3d(current_mask.bool(), radius=1)

        # --- Step 3: 获取模型预测的 mask，并仅在膨胀区域内做 Top-K ---
        pred_mask = pred[b, -1]
        min_val = pred_mask.min()
        max_val = pred_mask.max()
        normalized_pred_mask = (pred_mask - min_val) / (max_val - min_val)  # [0, 1]

        # 屏蔽掉非膨胀区域
        masked_pred = normalized_pred_mask.clone()
        masked_pred[~dilated_mask] = 0  # 确保这些位置不会被 topk 选中

        flat_pred = masked_pred.view(-1)
        topk_indices = torch.topk(flat_pred, num_particle).indices

        new_mask = torch.zeros_like(pred_mask)
        new_mask.view(-1)[topk_indices] = 1
        new_mask = (new_mask * 2) - 1  # 映射为 [-1, 1]
        pred[b, -1] = new_mask

        # --- Step 4: 根据 mask 修改速度通道 ---
        binary_mask = ((new_mask + 1) / 2).bool()  # (D, H, W)

        for c in range(3):
            pred[b, c][binary_mask] = pred[b, c][binary_mask]  # 粒子区域保持原预测
            current_channel = current_data[b, c]
            current_particle_mask = ((current_data[b, -1] + 1) / 2).bool()
            background_mask = ~current_particle_mask

            if torch.sum(background_mask) > 0:
                base_velocity = torch.sum(
                    current_channel * background_mask
                ) / torch.sum(background_mask)
            else:
                base_velocity = 0.0

            pred[b, c][~binary_mask] = base_velocity

        return pred


def log_validation_particle_fluid(
    denoising_unet,
    depth,
    height,
    width,
    scheduler,
    accelerator,
    generator=None,
    valid_dataset=None,
    num_inference_steps=100,
):
    logger.info("Running validation... ")
    if generator is None:
        generator = torch.manual_seed(42)

    dataset_len = len(valid_dataset)
    sample_idx = random.randint(0, dataset_len)
    current_data, future_data = valid_dataset[sample_idx]

    current_data = current_data.unsqueeze(0).to("cuda")
    future_data = future_data.unsqueeze(0).to("cuda")

    pipe = SCALEDParticleFluidPipeline(
        denoising_unet,
        scheduler,
    )
    pipe = pipe.to(accelerator.device)

    pred = pipe(
        current_data,
        num_inference_steps=num_inference_steps,
        guidance_scale=0,
        depth=depth,
        height=height,
        width=width,
        generator=generator,
        return_dict=False,
    )

    pred = apply_mask_on_velocity(pred, current_data, dilation_radius=5)

    results = {
        "sample_index": sample_idx,
        "pred_future_velocity": pred.detach().cpu().numpy()[0],
        "gt_future_velocity": future_data.detach().cpu().numpy()[0],
        "current_velocity": current_data.detach().cpu().numpy()[0],
    }

    del pipe
    return results


def visualize_with_diff(data_pre, data_gt, data_ori, filename):
    depth = data_pre.shape[1]

    # Create a figure with a larger size and higher resolution
    fig = plt.figure(figsize=(24, 42), dpi=300)  # Taller figure for 7 rows
    gs = gridspec.GridSpec(
        8, 4, width_ratios=[1, 1, 1, 0.1]
    )  # 7 rows: 6 channels + 1 diff

    # Plot first 6 channels
    for i in range(6):
        ax1 = plt.subplot(gs[i, 0])
        im1 = ax1.imshow(data_pre[i, depth // 2], vmin=-1, vmax=1)
        ax1.set_title(f"Prediction [{i}]")
        ax1.axis("off")

        ax2 = plt.subplot(gs[i, 1])
        im2 = ax2.imshow(data_gt[i, depth // 2], vmin=-1, vmax=1)
        ax2.set_title(f"Ground Truth [{i}]")
        ax2.axis("off")

        ax3 = plt.subplot(gs[i, 2])
        im3 = ax3.imshow(data_ori[i, depth // 2], vmin=-1, vmax=1)
        ax3.set_title(f"Current Velocity [{i}]")
        ax3.axis("off")

        # Add colorbars (reuse same axis to avoid too many legends)
        cbar_ax = plt.subplot(gs[i, 3])
        fig.colorbar(im1, cax=cbar_ax)

    # Position Mask
    ax4 = plt.subplot(gs[6, 0])
    im4 = ax4.imshow(data_pre[-1, depth // 2], vmin=-1, vmax=1)
    ax4.set_title("Prediction Position Mask")
    ax4.axis("off")

    ax5 = plt.subplot(gs[6, 1])
    im5 = ax5.imshow(data_gt[-1, depth // 2], vmin=-1, vmax=1)
    ax5.set_title("Ground Truth Position Mask")
    ax5.axis("off")

    ax6 = plt.subplot(gs[6, 2])
    im6 = ax6.imshow(data_ori[-1, depth // 2], vmin=-1, vmax=1)
    ax6.set_title("Current Position Mask")
    ax6.axis("off")

    cbar_ax_diff = plt.subplot(gs[6, 3])
    fig.colorbar(im4, cax=cbar_ax_diff)  # Use one colorbar for all diffs

    # Now plot the differences in the 7th row using channel 0
    diff_pre_gt = np.abs(data_pre - data_gt)
    diff_pre_ori = np.abs(data_pre - data_ori)
    diff_gt_ori = np.abs(data_gt - data_ori)

    ax7 = plt.subplot(gs[7, 0])
    im7 = ax7.imshow(diff_pre_gt[0, depth // 2], vmin=0, vmax=1)
    ax7.set_title("Diff Pre-GT")
    ax7.axis("off")

    ax8 = plt.subplot(gs[7, 1])
    im8 = ax8.imshow(diff_pre_ori[0, depth // 2], vmin=0, vmax=1)
    ax8.set_title("Diff Pre-Ori")
    ax8.axis("off")

    ax9 = plt.subplot(gs[7, 2])
    im9 = ax9.imshow(diff_gt_ori[0, depth // 2], vmin=0, vmax=1)
    ax9.set_title("Diff GT-Ori")
    ax9.axis("off")

    cbar_ax_diff = plt.subplot(gs[7, 3])
    fig.colorbar(im7, cax=cbar_ax_diff)  # Use one colorbar for all diffs

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the figure with higher resolution
    plt.savefig(filename, dpi=300)
    plt.close(fig)
