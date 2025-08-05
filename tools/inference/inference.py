import os
import sys

sys.path.append(
    "/scratch_dgxl/zl624/workspace/SCALED-Scalable-Generative-Foundational-Model-for-Computational-Physics-main/tools/trainning"
)

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scaled.model.unets.unet_3ds import UNet3DsModel
from scaled.dataset.particle_fluid_dataset import ParticleFluidDataset
from scaled.pipelines.pipeline_ddim_scaled_particle_fluid import (
    SCALEDParticleFluidPipeline,
)
from diffusers import DDIMScheduler
import torch.nn as nn


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


def dilate_mask_square_3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    3D方形膨胀：对bool类型mask进行立方体卷积，半径为radius。
    """
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    kernel = torch.ones(
        (1, 1, 2 * radius + 1, 2 * radius + 1, 2 * radius + 1), device=mask.device
    )
    out = torch.nn.functional.conv3d(mask_f, kernel, padding=radius)
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


def visualize_particles(points, data_type, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    plt.scatter(points[:, 0], points[:, 1], s=1, c=data_type, alpha=0.5)
    plt.xlim(0.1, 0.9)
    plt.ylim(0.1, 0.9)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


def main():
    # meta_path = "data/SFC/WaterRamps/metadata.json"
    # weight_path = "outputs/unet1d_regression_experiment/model-20000.pth"
    # sample_list = [f"sample_{i:05d}" for i in range(2)]

    save_dir = "result_particle/8channel/npy/pred"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    denoising_unet = UNet3DsModel(
        in_channels=16,
        out_channels=8,
        down_block_types=("DownBlock3D", "DownBlock3D", "DownBlock3D", "DownBlock3D"),
        up_block_types=("UpBlock3D", "UpBlock3D", "UpBlock3D", "UpBlock3D"),
        block_out_channels=(64, 128, 192, 256),
        add_attention=False,
    )
    denoising_unet.requires_grad_(False)
    weight = torch.load("exp_output/fludized_bed/mask/denoising_unet-100000.pth", map_location="cpu")
    denoising_unet.load_state_dict(weight, strict=False)

    net = Net(denoising_unet).to(device)

    val_dataset = ParticleFluidDataset(
        data_dir="data/couple_spout_3D",
        skip_timestep=1,
        time_steps_list=[i for i in range(200, 250)],
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        drop_last=True,
    )

    noise_scheduler_kwargs = {
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "linear",
        "steps_offset": 1,
        "clip_sample": False,
    }

    scheduler = DDIMScheduler(**noise_scheduler_kwargs)

    pipe = SCALEDParticleFluidPipeline(
        net.denoising_unet,
        scheduler,
    )

    num_inference_steps = 100
    depth = 256
    height = 64
    width = 64
    generator = torch.Generator(device=device)

    weight_dtype = "fp32"

    current_data = next(iter(val_dataloader))[0].to(device)
    os.makedirs(save_dir, exist_ok=True)
    for step in tqdm(range(1, 49)):

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
        pred = apply_mask_on_velocity(pred, current_data, dilation_radius=3)

        # 更新current_data
        current_data = pred

        # 保存每一步的mask
        position_mask = pred.cpu().numpy()[0, :, :, :, :]
        np.save(os.path.join(save_dir, f"prediction_{step:03d}.npy"), position_mask)

if __name__ == "__main__":
    main()
