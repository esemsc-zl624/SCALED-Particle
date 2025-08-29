import numpy as np
import argparse
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def save_metrics_csv(metrics_dict, csv_file):
    """
    metrics_dict: dict[str, list]，每个 key 对应一个指标列表
    csv_file: 保存的 CSV 路径
    """
    df = pd.DataFrame(metrics_dict)
    # 计算累积和累积平均
    for metric in metrics_dict.keys():
        df[f"{metric}_cumulative"] = df[metric].cumsum()
        df[f"{metric}_cumulative_avg"] = df[metric].expanding().mean()
    df.to_csv(csv_file, index=False)

def plot_metrics(csv_file):
    df = pd.read_csv(csv_file)
    metrics = ["mse", "accuracy", "precision", "recall", "f1", "iou"]

    plt.figure(figsize=(12, 8))
    for metric in metrics:
        plt.plot(df[metric], label=f"{metric} per step")
        plt.plot(df[f"{metric}_cumulative_avg"], linestyle='--', label=f"Cumulative Avg {metric}")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Metrics per Step and Cumulative Average")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(os.path.dirname(csv_file), "metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def calculate_metrics(gt_file, pred_file):
    gt = np.load(gt_file)
    pred = np.load(pred_file)

    gt_pos_mask = ((gt[-1] + 1) / 2).astype(np.uint8)
    pred_pos_mask = ((pred[-1] + 1) / 2).astype(np.uint8)

    gt_vel = gt[3:6]
    pred_vel = pred[3:6]

    gt_flat = gt_pos_mask.flatten()
    pred_flat = pred_pos_mask.flatten()

    # mse_pos = np.mean((pred_pos_mask - gt_pos_mask) ** 2)
    mse_vel = np.mean((pred_vel - gt_vel) ** 2)

    accuracy = accuracy_score(gt_flat, pred_flat)
    precision = precision_score(gt_flat, pred_flat, zero_division=0)
    recall = recall_score(gt_flat, pred_flat, zero_division=0)
    f1 = f1_score(gt_flat, pred_flat, zero_division=0)

    return mse_vel, accuracy, precision, recall, f1

def calculate_metrics_dir(gt_dir, pred_dir):
    metrics_lists = {
        "mse": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    num_files = len(os.listdir(pred_dir))
    for step in tqdm(range(1, num_files+1)):
        gt_file = os.path.join(gt_dir, f"gt_{step:03d}.npy")
        pred_file = os.path.join(pred_dir, f"pred_{step:03d}.npy")

        mse, acc, prec, rec, f1_val = calculate_metrics(gt_file, pred_file)
        metrics_lists["mse"].append(mse)
        metrics_lists["accuracy"].append(acc)
        metrics_lists["precision"].append(prec)
        metrics_lists["recall"].append(rec)
        metrics_lists["f1"].append(f1_val)

    return metrics_lists

def plot_metrics_subplots(csv_file):
    """
    每个指标单独一个 subplot，显示原始值和累积平均
    布局：2行3列
    第一行：recall, precision, f1, mse
    """
    df = pd.read_csv(csv_file)
    metrics_order = ["recall", "precision", "f1", "mse"]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharex=True)
    axes = axes.flatten()

    for i, metric in enumerate(metrics_order):
        ax = axes[i]
        ax.plot(df[metric], label=f"{metric} per step")
        ax.plot(df[f"{metric}_cumulative_avg"], label=f"Cumulative Avg {metric}")
        ax.set_xlabel("Step")
        ax.set_ylabel(metric)
        ax.grid(True)
        ax.legend()


    fig.suptitle("Metrics per Step and Cumulative Average", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    save_path = os.path.join(os.path.dirname(csv_file), "metrics_subplots.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt_dir",
        type=str,
        help="Directory containing groundtruth .npy files",
        default="data/evaluation_npy_step200-250",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        help="Directory containing predicted .npy files",
        required=True,
    )
    args = parser.parse_args()

    parent_dir = os.path.dirname(args.pred_dir)

    metrics_lists = calculate_metrics_dir(args.gt_dir, args.pred_dir)

    metrics_csv_file = os.path.join(parent_dir, "metrics.csv")
    save_metrics_csv(metrics_lists, metrics_csv_file)
    plot_metrics_subplots(metrics_csv_file) 
