# input is a folder of T npy files, each file is in shape (8, D, H, W)
# we need to extract the -1 channel as mask
# visualize the mask over T timesteps
# output is a folder of png files and a gif file

import numpy as np
import imageio
import os
from tqdm import tqdm
import sys
import argparse


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 获取按顺序排列的 T 个 .npy 文件名
    npy_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    T = len(npy_files)


    position_frames = []
    velocity_frames = []

    for fname in tqdm(npy_files, desc="Processing slices"):

        fname = os.path.join(input_dir, fname)

        # ================================================
        # visualize the position mask over T timesteps
        # ================================================

        volume = np.load(fname)[-1]  # channel -1 is position mask
        volume_bool = volume > 0
        mid_slice = volume_bool[volume.shape[0] // 2]

        # 转为灰度图像（0 或 255）
        img = (mid_slice * 255).astype(np.uint8)

        position_frames.append(img)

        # ================================================
        # visualize the velocity over T timesteps
        # ================================================

        volume = np.load(fname)[3]  # channel 3 is flow velocity
        mid_slice = volume[volume.shape[0] // 2]

        # 将值归一化到 [0, 255]，适用于浮点值（-1 到 1）
        norm_img = ((mid_slice + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        velocity_frames.append(norm_img)

    output_position_gif = os.path.join(output_dir, "position.gif")
    output_velocity_gif = os.path.join(output_dir, "velocity.gif")
    # 保存为 GIF
    imageio.mimsave(output_position_gif, position_frames, duration=500)
    imageio.mimsave(output_velocity_gif, velocity_frames, duration=500)

    print(f"GIF saved to {output_position_gif} and {output_velocity_gif}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
