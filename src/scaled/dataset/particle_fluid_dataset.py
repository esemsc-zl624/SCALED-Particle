import os
from torch.utils.data import Dataset
import numpy as np
import torch
import random
import json


class ParticleFluidDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        time_steps_list=None,
        skip_timestep=1,
    ):
        self.data_dir = data_dir
        self.skip_timestep = skip_timestep
        self.data_list = time_steps_list

    def __len__(self):
        return len(self.data_list) - self.skip_timestep

    def get_data(self, time_step):

        up = np.load(
            os.path.join(self.data_dir, f"up{time_step}000.npy"), mmap_mode="r"
        )
        vp = np.load(
            os.path.join(self.data_dir, f"vp{time_step}000.npy"), mmap_mode="r"
        )
        wp = np.load(
            os.path.join(self.data_dir, f"wp{time_step}000.npy"), mmap_mode="r"
        )
        u = np.load(os.path.join(self.data_dir, f"u{time_step}000.npy"), mmap_mode="r")
        v = np.load(os.path.join(self.data_dir, f"v{time_step}000.npy"), mmap_mode="r")
        w = np.load(os.path.join(self.data_dir, f"w{time_step}000.npy"), mmap_mode="r")
        C = np.load(os.path.join(self.data_dir, f"C{time_step}000.npy"))

        if (
            u.shape != v.shape
            or u.shape != w.shape
            or u.shape != C.shape
            or u.shape != up.shape
            or u.shape != vp.shape
            or u.shape != wp.shape
        ):
            print(f"Shape mismatch found at timestep {time_step}:")
            print(f"  u shape: {u.shape}")
            print(f"  v shape: {v.shape}")
            print(f"  w shape: {w.shape}")
            print(f"  up shape: {up.shape}")
            print(f"  vp shape: {vp.shape}")
            print(f"  wp shape: {wp.shape}")
            print(f"  C shape: {C.shape}")

        # mask the data where the velocity is 0
        mask = np.all(np.array([up, vp, wp]) != 0, axis=0).astype(np.float32)

        result = np.stack([up, vp, wp, u, v, w, C, mask], axis=0)[:, 0:256, 0:64, 0:64]
        return result

    # resize the value to -1--1
    def normalize(self, data):
        data[0] = (data[0] + 5) / 15
        data[1] = (data[1] + 4) / 14
        data[2] = (data[2] + 7) / 16
        data[3] = (data[3] + 5) / 10
        data[4] = (data[4] + 5) / 10
        data[5] = (data[5] + 8) / 30
        data[6] = data[6] / 0.6
        data = data * 2 - 1
        return data

    def __getitem__(self, idx):
        time_step = self.data_list[idx]
        ori_data = self.get_data(time_step)
        future_data = self.get_data(time_step + self.skip_timestep)
        ori_data = self.normalize(ori_data)
        future_data = self.normalize(future_data)
        return torch.from_numpy(ori_data).float(), torch.from_numpy(future_data).float()
