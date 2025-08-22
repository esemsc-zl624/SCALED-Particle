import numpy as np
import argparse
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt


def numpy2csv(numpy_list, csv_file):
    # 转成 DataFrame
    df = pd.DataFrame(numpy_list, columns=["error"])

    # 计算累积误差
    df["cumulative"] = df["error"].cumsum()

    # 计算累积平均
    df["cumulative_avg"] = df["error"].expanding().mean()

    # 保存到 CSV
    df.to_csv(csv_file, index=False)


def plot_error_csv(csv_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(8, 5))
    plt.plot(df["error"], label="MSE per step")
    # plt.plot(df["cumulative"], label="Cumulative MSE")
    plt.plot(df["cumulative_avg"], label="Cumulative Average MSE")
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.title("MSE and Cumulative Average MSE")
    plt.legend()
    plt.grid(True)

    save_dir = os.path.dirname(csv_file)
    save_path = os.path.join(save_dir, "mse.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_cumulative_error(csv_file):
    df = pd.read_csv(csv_file)

    plt.figure(figsize=(8,5))
    plt.plot(df["cumulative"], label="Cumulative MSE", color="orange")
    plt.xlabel("Step")
    plt.ylabel("Cumulative MSE")
    plt.title("Cumulative Average MSE")
    plt.legend()
    plt.grid(True)

    save_dir = os.path.dirname(csv_file)
    save_path = os.path.join(save_dir, "cumulative_mse.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_mse(gt_file, pred_file, time_idx=None):
    """
    计算单个时间步或整个样本的 one-step MSE（最后一个通道）
    - time_idx: None 表示整个样本，否则取该时间步
    """
    gt = np.load(gt_file)  # shape: 8*256*64*64
    pred = np.load(pred_file)

    gt_last = gt[-1]
    pred_last = pred[-1]

    mse = np.mean((pred_last - gt_last) ** 2)
    return mse


def calculate_mse_dir(gt_dir, rollout_pred_dir):
    """
    对整个目录的 npy 文件计算 rollout MSE
    文件名规则：
        gt_{i:03d}.npy
        pred_{i:03d}.npy
    """
    rollout_mse_list = []
    num_files = len(os.listdir(rollout_pred_dir))

    for step in tqdm(range(1, num_files + 1)):
        gt_file = os.path.join(gt_dir, f"gt_{step:03d}.npy")
        pred_file = os.path.join(rollout_pred_dir, f"pred_{step:03d}.npy")

        mse = calculate_mse(gt_file, pred_file, time_idx=step)
        rollout_mse_list.append(mse)

    return np.mean(rollout_mse_list), rollout_mse_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute one-step and rollout MSE for particle predictions."
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        help="Directory containing groundtruth .npy files",
        default="/scratch_dgxl/zl624/workspace/SCALED-Particle/data/evaluation_npy_step200-250",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        help="Directory containing predicted .npy files",
        required=True,
    )

    args = parser.parse_args()

    avg_mse, mse_list = calculate_mse_dir(args.gt_dir, args.pred_dir)

    print(f"MSE: {avg_mse:.6e}")

    parent_dir = os.path.dirname(args.pred_dir)
    mse_csv_file = os.path.join(parent_dir, "mse.csv")
    numpy2csv(mse_list, mse_csv_file)
    plot_error_csv(mse_csv_file)
    plot_cumulative_error(mse_csv_file)