import os
import torch
import numpy as np
from tqdm import tqdm
from scaled.model.scaled_particle import Net
from scaled.model.unets.unet_3ds import UNet3DsModel
from scaled.dataset.particle_fluid_dataset import ParticleFluidDataset
from scaled.pipelines.pipeline_ddim_scaled_particle_fluid import (
    SCALEDParticleFluidPipeline,
)
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
import argparse


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
        current_mask = current_data[b, 7]  # (D, H, W)，值为 -1 或 1
        current_mask = (current_mask + 1) / 2  # 映射为 [0, 1]
        num_particle = int(torch.sum(current_mask).item())

        # --- Step 2: 计算膨胀 mask ---
        dilated_mask = dilate_mask_square_3d(current_mask.bool(), radius=1)

        # --- Step 3: 获取模型预测的 mask，并仅在膨胀区域内做 Top-K ---
        pred_mask = pred[b, 7]
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
        new_mask = (new_mask * 2) - 1  # 映射为 [7, 1]
        pred[b, 7] = new_mask

        # --- Step 4: 根据 mask 修改速度通道 ---
        binary_mask = ((new_mask + 1) / 2).bool()  # (D, H, W)

        for c in range(3):
            pred[b, c][binary_mask] = pred[b, c][binary_mask]  # 粒子区域保持原预测
            current_channel = current_data[b, c]
            current_particle_mask = ((current_data[b, 7] + 1) / 2).bool()
            background_mask = ~current_particle_mask

            if torch.sum(background_mask) > 0:
                base_velocity = torch.sum(
                    current_channel * background_mask
                ) / torch.sum(background_mask)
            else:
                base_velocity = 0.0

            pred[b, c][~binary_mask] = base_velocity

        return pred


def main(cfg, weight_path):

    save_dir = os.path.join(cfg.output_dir, cfg.exp_name, "rollout", "npy")
    os.makedirs(save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    denoising_unet = UNet3DsModel(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        down_block_types=cfg.model.down_block_types,
        up_block_types=cfg.model.up_block_types,
        block_out_channels=cfg.model.block_out_channels,
        add_attention=cfg.model.add_attention,
    )
    denoising_unet.requires_grad_(False)
    weight = torch.load(weight_path, map_location="cpu")
    denoising_unet.load_state_dict(weight, strict=False)

    net = Net(denoising_unet).to(device)

    val_dataset = ParticleFluidDataset(
        data_dir=cfg.dataset.dataset_path,
        skip_timestep=cfg.dataset.skip_timestep,
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
        "num_train_timesteps": cfg.noise_scheduler_kwargs.num_train_timesteps,
        "beta_start": cfg.noise_scheduler_kwargs.beta_start,
        "beta_end": cfg.noise_scheduler_kwargs.beta_end,
        "beta_schedule": cfg.noise_scheduler_kwargs.beta_schedule,
        "steps_offset": cfg.noise_scheduler_kwargs.steps_offset,
        "clip_sample": cfg.noise_scheduler_kwargs.clip_sample,
        "prediction_type": cfg.noise_scheduler_kwargs.prediction_type,
    }

    scheduler = DDIMScheduler(**noise_scheduler_kwargs)

    pipe = SCALEDParticleFluidPipeline(
        net.denoising_unet,
        scheduler,
    )

    num_inference_steps = cfg.inference.num_inference_steps
    generator = torch.Generator(device=device)

    current_data = next(iter(val_dataloader))[0].to(device)

    os.makedirs(save_dir, exist_ok=True)
    for step in tqdm(range(1, 49)):

        # boundary condition
        boundary_mask = torch.zeros_like(current_data, dtype=torch.bool)

        boundary_mask[:, :, 0, :, :] = True  # front
        boundary_mask[:, :, 7, :, :] = True  # back
        boundary_mask[:, :, :, 0, :] = True  # top
        boundary_mask[:, :, :, 7, :] = True  # bottom
        boundary_mask[:, :, :, :, 0] = True  # left
        boundary_mask[:, :, :, :, 7] = True  # right

        boundary_condition = current_data * boundary_mask

        current_data = torch.cat([current_data, boundary_condition], dim=1)

        pred = pipe(
            current_data,
            num_inference_steps=num_inference_steps,
            guidance_scale=0,
            depth=cfg.dataset.depth,
            height=cfg.dataset.height,
            width=cfg.dataset.width,
            generator=generator,
            return_dict=False,
        )
        pred = apply_mask_on_velocity(
            pred, current_data, dilation_radius=cfg.inference.dilation_radius
        )

        # 更新current_data
        current_data = pred

        # 保存每一步的result
        pred_numpy = pred.cpu().numpy()[0, :, :, :, :]
        np.save(os.path.join(save_dir, f"pred_{step:03d}.npy"), pred_numpy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--weight_path", type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config, args.weight_path)
