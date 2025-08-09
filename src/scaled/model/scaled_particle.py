import torch.nn as nn
from scaled.model.unets.unet_3ds import UNet3DsModel


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
