import numpy as np
import argparse
from tqdm import tqdm
import os


def calculate_one_step_mse(gt_file, pred_file, time_idx=None):
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


def calculate_rollout_mse(gt_dir, rollout_pred_dir):
    """
    对整个目录的 npy 文件计算 rollout MSE
    文件名规则：
        gt_{i:03d}.npy
        pred_{i:03d}.npy
    """
    rollout_mse_list = []
    num_files = len(os.listdir(rollout_pred_dir))

    for step in tqdm(range(1, num_files+1)):
        gt_file = os.path.join(gt_dir, f"gt_{step:03d}.npy")
        pred_file = os.path.join(rollout_pred_dir, f"pred_{step:03d}.npy")

        mse = calculate_one_step_mse(gt_file, pred_file, time_idx=step)
        rollout_mse_list.append(mse)

    return np.mean(rollout_mse_list)


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
        "--rollout_pred_dir",
        type=str,
        help="Directory containing predicted .npy files",
    )
    parser.add_argument(
        "--one_step_pred_dir",
        type=str,
        help="Directory containing predicted .npy files",
    )

    args = parser.parse_args()

    rollout_mse = calculate_rollout_mse(args.gt_dir, args.rollout_pred_dir)
    # avg_onestep_mse = calculate_rollout_mse(args.gt_dir, args.one_step_pred_dir)

    print(f"Rollout MSE: {rollout_mse:.6e}")
    # print(f"Avg one-step MSE: {avg_onestep_mse:.6e}")
