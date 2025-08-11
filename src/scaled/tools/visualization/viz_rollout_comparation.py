import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
import glob
import os
from PIL import Image
from tqdm import tqdm
import argparse

# 输入的data_pre和data_gt的shape是 (T, 8, D, H, W)
# 需要画出每一个时间步的，7个通道的对比，图片grid是 2行7列
# 将所有时间步的png拼接成一个gif

# 得从validation dataset get_item 得到的才是 (-1,1) 的值
# 写一个脚本，将validation dataset 的值转换为 (-1,1) 的值，保存为npy
# 或者直接normalize npy


def visualize_onestep_comparasion(data_pred, data_gt, png_path, step=None, slice="xy"):
    """
    Visualize the rollout comparison of one timestep of data_pred and data_gt
    Args:
        data_pred: (8, D, H, W)
        data_gt: (8, D, H, W)
        png_path: str, the path to save the figure
        slice: str, one of {"xy", "xz", "yz"} — which plane to visualize
    """
    depth = data_pred.shape[1]  # D
    height = data_pred.shape[2]  # H
    width = data_pred.shape[3]  # W

    rows, cols = 2, 7
    if slice == "xy":
        fig = plt.figure(figsize=(2 * cols, 2 * rows), dpi=100)
    else:
        fig = plt.figure(figsize=(2 * cols, 1 * rows), dpi=100)

    gs = gridspec.GridSpec(rows, cols)

    channel_names = [
        "Particle Velocity (X)",
        "Particle Velocity (Y)",
        "Particle Velocity (Z)",
        "Flow Velocity (X)",
        "Flow Velocity (Y)",
        "Flow Velocity (Z)",
        "Particle Position",
    ]

    # choose index and slicing
    if slice == "xy":  # fixed z
        slicer = lambda arr: arr[:, depth // 2, :, :]
    elif slice == "xz":  # fixed y
        slicer = lambda arr: arr[:, :, height // 2, :]
    elif slice == "yz":  # fixed x
        slicer = lambda arr: arr[:, :, :, width // 2]
    else:
        raise ValueError(f"Invalid slice: {slice}. Must be one of 'xy','xz','yz'.")

    pred_slice = slicer(data_pred)
    gt_slice = slicer(data_gt)

    # for xz / yz, we may need to transpose for imshow
    def prepare_img(img, plane):
        if plane == "xy":
            return img
        elif plane == "xz":
            return img.transpose(0, 2, 1)  # swap z and x to match imshow
        elif plane == "yz":
            return img.transpose(0, 2, 1)  # swap z and y
        return img

    pred_slice = prepare_img(pred_slice, slice)
    gt_slice = prepare_img(gt_slice, slice)

    # Plot first 6 channels
    for i in range(6):
        ax1 = plt.subplot(gs[0, i])
        ax1.imshow(pred_slice[i], vmin=-1, vmax=1)
        ax1.set_title(f"{channel_names[i]}")
        ax1.axis("off")

        ax2 = plt.subplot(gs[1, i])
        ax2.imshow(gt_slice[i], vmin=-1, vmax=1)
        ax2.axis("off")

    # Position mask (channel 7)
    ax4 = plt.subplot(gs[0, 6])
    ax4.imshow(pred_slice[7], vmin=-1, vmax=1)
    ax4.set_title(f"{channel_names[6]}")
    ax4.axis("off")

    ax5 = plt.subplot(gs[1, 6])
    ax5.imshow(gt_slice[7], vmin=-1, vmax=1)
    ax5.axis("off")

    if step is not None:
        fig.suptitle(f"Slice: {slice}    Step: {step}", fontsize=16)

    # Adjust layout to avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save the figure with higher resolution
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

    eval_dir = args.output_dir

    png_dir = os.path.join(eval_dir, "png")
    os.makedirs(png_dir, exist_ok=True)

    for slice in ["xy", "xz", "yz"]:
        for i in tqdm(range(1, 48)):
            data_pred = np.load(
                os.path.join(data_pred_dir, f"pred_{i:03d}.npy")
            )  # (8, D, H, W)
            data_gt = np.load(
                os.path.join(data_gt_dir, f"gt_{i:03d}.npy")
            )  # (8, D, H, W)
            png_path = os.path.join(png_dir, f"comparasion_{i:03d}_{slice}.png")

            visualize_onestep_comparasion(
                data_pred, data_gt, png_path, step=i, slice=slice
            )

        gif_path = os.path.join(eval_dir, f"comparasion_{slice}.gif")
        pngs_to_gif(png_dir, gif_path, slice=slice)
