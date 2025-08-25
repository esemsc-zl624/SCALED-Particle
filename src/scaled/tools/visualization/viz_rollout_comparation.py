import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import glob
import os
from PIL import Image
from tqdm import tqdm
import argparse


def visualize_onestep_comparasion(data_pred, data_gt, png_path, step=None, slice="xy"):
    """
    Visualize one-step comparison of predicted vs. ground-truth particle states.

    Args:
        data_pred: (8, D, H, W)
        data_gt: (8, D, H, W)
        png_path: str, path to save figure
        slice: str, one of {"xy", "xz", "yz"}
    """
    depth, height, width = data_pred.shape[1], data_pred.shape[2], data_pred.shape[3]
    channel_names = [
        "Particle Velocity (X)",
        "Particle Velocity (Y)",
        "Particle Velocity (Z)",
        "Flow Velocity (X)",
        "Flow Velocity (Y)",
        "Flow Velocity (Z)",
        "Particle Position",
    ]

    # Slicing
    if slice == "xy":
        slicer = lambda arr: arr[:, depth // 2, :, :]
    elif slice == "xz":
        slicer = lambda arr: arr[:, :, height // 2, :]
    elif slice == "yz":
        slicer = lambda arr: arr[:, :, :, width // 2]
    else:
        raise ValueError(f"Invalid slice: {slice}. Must be one of 'xy','xz','yz'.")

    pred_slice = slicer(data_pred)
    gt_slice = slicer(data_gt)

    # Rotate xz / yz 90° CCW
    def prepare_img(img, plane):
        if plane == "xy":
            return img
        elif plane in ["xz", "yz"]:
            img = img.transpose(0, 2, 1)
            return np.rot90(img, k=1, axes=(1, 2))
        return img

    pred_slice = prepare_img(pred_slice, slice)
    gt_slice = prepare_img(gt_slice, slice)

    if slice == "xy":
        fig = plt.figure(figsize=(12, 2))
        title_y = 0.7
    else:
        fig = plt.figure(figsize=(12, 3))
        title_y = 0.9

    outer = fig.add_gridspec(1, 7, wspace=0.5)

    for i in range(7):
        inner = outer[i].subgridspec(1, 2, wspace=0.1)

        ax_pred = fig.add_subplot(inner[0])
        ax_pred.imshow(pred_slice[i], vmin=-1, vmax=1)
        ax_pred.axis("off")

        ax_gt = fig.add_subplot(inner[1])
        ax_gt.imshow(gt_slice[i], vmin=-1, vmax=1)
        ax_gt.axis("off")

        ax_title = fig.add_subplot(outer[i])
        ax_title.set_title(channel_names[i], fontsize=10, pad=10, y=title_y)
        ax_title.axis("off")

    if step is not None:
        fig.suptitle(f"Plane: {slice}    Step: {step}", fontsize=16)

    # Adjust layout to avoid overlap
    plt.savefig(png_path, dpi=300)
    plt.close(fig)


def pngs_to_gif(png_dir, save_path, slice="xy"):
    """
    Convert a directory of PNG images to a GIF
    Args:
        png_dir: str, the path to the directory of PNG images
        save_path: str, the path to save the GIF
    Returns:
        None
    """

    png_files = sorted(glob.glob(os.path.join(png_dir, f"*_{slice}.png")))
    images = [Image.open(file) for file in png_files]
    images[0].save(
        save_path, save_all=True, append_images=images[1:], duration=1000, loop=0
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_input_dir", type=str, default="data/evaluation_npy_step200-250"
    )
    parser.add_argument("--pred_input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()

    data_pred_dir = args.pred_input_dir
    data_gt_dir = args.gt_input_dir

    npy_files = sorted(glob.glob(os.path.join(data_pred_dir, "pred_*.npy")))
    num_files = len(npy_files)

    rollout_dir = args.output_dir

    png_dir = os.path.join(rollout_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    for slice in ["xy", "xz", "yz"]:
        for i in tqdm(range(1, num_files + 1)):
            data_pred = np.load(
                os.path.join(data_pred_dir, f"pred_{i:03d}.npy")
            )  # (8, D, H, W)
            data_gt = np.load(
                os.path.join(data_gt_dir, f"gt_{i:03d}.npy")
            )  # (8, D, H, W)
            png_path = os.path.join(png_dir, f"comparison_{i:03d}_{slice}.png")

            visualize_onestep_comparasion(
                data_pred, data_gt, png_path, step=i, slice=slice
            )

        gif_path = os.path.join(rollout_dir, f"comparison_{slice}.gif")
        pngs_to_gif(png_dir, gif_path, slice=slice)
