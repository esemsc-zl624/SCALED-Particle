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

from scaled.tools.inference.inference import apply_mask_on_velocity, get_boundary_condition


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
    num_inference_steps=100,
    dilation_radius=5,
):
    logger.info("Running validation... ")
    if generator is None:
        generator = torch.manual_seed(42)

    dataset_len = len(valid_dataset)
    sample_idx = random.randint(0, dataset_len)
    current_data, future_data = valid_dataset[sample_idx]

    current_data = current_data.unsqueeze(0).to("cuda")
    future_data = future_data.unsqueeze(0).to("cuda")

    boundary_condition, boundary_mask = get_boundary_condition(
        current_data, future_data
    )

    current_data = torch.cat([current_data, boundary_condition], dim=1)

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

    pred = apply_mask_on_velocity(pred, current_data, dilation_radius=dilation_radius)
    pred = torch.where(boundary_mask, future_data, pred)

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
    im4 = ax4.imshow(data_pre[7, depth // 2], vmin=-1, vmax=1)
    ax4.set_title("Prediction Position Mask")
    ax4.axis("off")

    ax5 = plt.subplot(gs[6, 1])
    im5 = ax5.imshow(data_gt[7, depth // 2], vmin=-1, vmax=1)
    ax5.set_title("Ground Truth Position Mask")
    ax5.axis("off")

    ax6 = plt.subplot(gs[6, 2])
    im6 = ax6.imshow(data_ori[7, depth // 2], vmin=-1, vmax=1)
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
