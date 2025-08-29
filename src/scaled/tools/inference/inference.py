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
    3D Square Dilate:
    Dilate a bool mask with a cubic kernel of radius.
    """
    mask_f = mask.float().unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
    kernel = torch.ones(
        (1, 1, 2 * radius + 1, 2 * radius + 1, 2 * radius + 1), device=mask.device
    )
    out = torch.nn.functional.conv3d(mask_f, kernel, padding=radius)
    return (out > 0).squeeze(0).squeeze(0).bool()  # return bool type (D, H, W)


def apply_mask_on_velocity(pred, current_data, dilation_radius=5):
    """
    Apply Top-K particle selection on the prediction results, and modify the velocity channel based on the mask.
    Only perform Top-K within the dilated region of current_data.

    Args:
        pred: model prediction results, shape (b, 7, D, H, W)
        current_data: current input data, shape same as pred
        dilation_radius: dilation radius (default 5)

    Returns:
        pred: prediction results after mask and velocity processing
    """

    batch_size = pred.shape[0]

    for b in range(batch_size):
        # --- Step 1: get the ground truth mask of current frame, and calculate the number of particles ---
        current_mask = current_data[b, 7]  # (D, H, W), value is -1 or 1
        current_mask = (current_mask + 1) / 2  # map to [0, 1]
        num_particle = int(torch.sum(current_mask).item())

        # --- Step 2: calculate dilated mask ---
        dilated_mask = dilate_mask_square_3d(
            current_mask.bool(), radius=dilation_radius
        )

        # --- Step 3: get the model prediction mask, and only perform Top-K within the dilated region ---
        pred_mask = pred[b, 7]
        min_val = pred_mask.min()
        max_val = pred_mask.max()
        normalized_pred_mask = (pred_mask - min_val) / (max_val - min_val)  # [0, 1]

        # mask out the non-dilated region
        masked_pred = normalized_pred_mask.clone()
        masked_pred[~dilated_mask] = 0  # ensure these positions are not selected by topk

        flat_pred = masked_pred.view(-1)
        topk_indices = torch.topk(flat_pred, num_particle).indices

        new_mask = torch.zeros_like(pred_mask)
        new_mask.view(-1)[topk_indices] = 1
        new_mask = (new_mask * 2) - 1  # map to [-1, 1]
        pred[b, 7] = new_mask

        # --- Step 4: modify the velocity channel based on the mask ---
        binary_mask = ((new_mask + 1) / 2).bool()  # (D, H, W)

        for c in range(3):
            pred[b, c][binary_mask] = pred[b, c][binary_mask]  # particle region keep original prediction
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


def get_boundary_condition(future_data, halo=4):
    """
    Get the boundary condition of the future data.
    The boundary mask is a bool mask that is True for the boundary region.

    Args:
        future_data: future data, shape (b, 8, D, H, W)
        halo: halo size (default 4)

    Returns:
        boundary_condition: boundary condition, shape (b, 8, D, H, W)
        boundary_mask: boundary mask, shape (b, 8, D, H, W)
    """
    boundary_mask = torch.zeros_like(future_data, dtype=torch.bool)

    boundary_mask[:, :, :halo, :, :] = True  # front
    boundary_mask[:, :, -halo:, :, :] = True  # back
    boundary_mask[:, :, :, :halo, :] = True  # top
    boundary_mask[:, :, :, -halo:, :] = True  # bottom
    boundary_mask[:, :, :, :, :halo] = True  # left
    boundary_mask[:, :, :, :, -halo:] = True  # right

    boundary_condition = future_data * boundary_mask

    return boundary_condition, boundary_mask


def find_latest_ckpt(ckpt_dir):
    """
    Find the latest checkpoint file in the checkpoint directory.

    Args:
        ckpt_dir: checkpoint directory

    Returns:
        latest_ckpt: path to the latest checkpoint file
    """

    # filter out all checkpoint files
    ckpt_files = [
        f
        for f in os.listdir(ckpt_dir)
        if f.startswith("denoising_unet-") and f.endswith(".pth")
    ]
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")

    # sort by training step
    def get_step(f):
        return int(f.split("-")[1].split(".")[0])

    ckpt_files.sort(key=get_step)
    latest_ckpt = ckpt_files[-1]
    return os.path.join(ckpt_dir, latest_ckpt)


def main(cfg, weight_path, inference_type):
    """
    Main function for inference.

    Args:
        cfg: config
        weight_path: path to the checkpoint file
        inference_type: type of inference

    Returns:
        None
    """

    if inference_type == "long_rollout":
        time_steps_list = [i for i in range(1, 249)]
    elif inference_type == "debug_rollout":
        time_steps_list = [i for i in range(200, 209)]
    else:
        time_steps_list = [i for i in range(200, 249)]

    save_dir = os.path.join(cfg.output_dir, cfg.exp_name, inference_type, "npy")
    os.makedirs(save_dir, exist_ok=True)

    if weight_path is None:
        ckpt_dir = os.path.join(cfg.output_dir, cfg.exp_name, "ckpt")
        weight_path = find_latest_ckpt(ckpt_dir)

    print(f"Using model checkpoint: {weight_path}")

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
        time_steps_list=time_steps_list,
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

    os.makedirs(save_dir, exist_ok=True)
    pred = None
    for step, data in enumerate(tqdm(val_dataloader, total=len(val_dataloader))):
        # get current data:
        # 1. for onestep, use the current data
        # 2. for rollout, use the previous output as current data
        if inference_type == "onestep" or pred is None:
            current_data = data[0].to(device)
        else:
            current_data = pred

        # get boundary condition
        future_data = data[1].to(device)
        boundary_condition, boundary_mask = get_boundary_condition(future_data)

        # get latent input
        latent_input = torch.cat([current_data, boundary_condition], dim=1)

        # get sfc condition
        if net.denoising_unet.config.in_channels == 25:
            morton_sfc = torch.load("data/morton_sfc.pt")
            position_mask = (current_data[:, 7:8] + 1) / 2
            sfc_condition = (
                morton_sfc.to(device=position_mask.device, dtype=position_mask.dtype)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand_as(position_mask)
            )
            sfc_condition = sfc_condition * position_mask
            sfc_condition = sfc_condition * 2 - 1
            latent_input = torch.cat([latent_input, sfc_condition], dim=1)

        # inference
        pred = pipe(
            latent_input,
            num_inference_steps=num_inference_steps,
            guidance_scale=0,
            depth=cfg.dataset.depth,
            height=cfg.dataset.height,
            width=cfg.dataset.width,
            generator=generator,
            return_dict=False,
        )
        pred = apply_mask_on_velocity(
            pred, latent_input, dilation_radius=cfg.inference.dilation_radius
        )
        pred = torch.where(boundary_mask, future_data, pred)  # [B, 8, D, H, W]

        # save result
        pred_numpy = pred.cpu().numpy()[0, :, :, :, :]
        np.save(os.path.join(save_dir, f"pred_{step+1:03d}.npy"), pred_numpy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--weight_path", type=str)
    parser.add_argument(
        "--inference_type",
        type=str,
        default="rollout",
        choices=["rollout", "onestep", "long_rollout", "debug_rollout"],
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    main(config, args.weight_path, args.inference_type)
