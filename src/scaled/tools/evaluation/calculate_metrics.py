import numpy as np
import argparse
from tqdm import tqdm
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def save_metrics_csv(metrics_dict, csv_file):
    """
    Save the metrics to a CSV file.

    Args:
        metrics_dict: dict[str, list], each key corresponds to a list of metrics
        csv_file: path to save the CSV file

    Returns:
        None
    """
    df = pd.DataFrame(metrics_dict)
    # calculate cumulative and cumulative average
    for metric in metrics_dict.keys():
        df[f"{metric}_cumulative"] = df[metric].cumsum()
        df[f"{metric}_cumulative_avg"] = df[metric].expanding().mean()
    df.to_csv(csv_file, index=False)


def plot_metrics(csv_file):
    """
    Plot the metrics per step and cumulative average.

    Args:
        csv_file: path to the CSV file

    Returns:
        None
    """
    df = pd.read_csv(csv_file)
    metrics = ["mse", "accuracy", "precision", "recall", "f1", "iou"]

    plt.figure(figsize=(12, 8))
    for metric in metrics:
        plt.plot(df[metric], label=f"{metric} per step")
        plt.plot(
            df[f"{metric}_cumulative_avg"],
            linestyle="--",
            label=f"Cumulative Avg {metric}",
        )
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Metrics per Step and Cumulative Average")
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(os.path.dirname(csv_file), "metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_metrics(gt_file, pred_file):
    """
    Calculate the metrics.

    Args:
        gt_file: path to the ground truth file
        pred_file: path to the predicted file

    Returns:
        mse_vel: mean squared error of velocity
        accuracy: accuracy
        precision: precision
        recall: recall
        f1: f1 score
    """
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
    """
    Calculate the metrics for all the files in the directory.

    Args:
        gt_dir: path to the ground truth directory
        pred_dir: path to the predicted directory

    Returns:
        metrics_lists: dict[str, list], each key corresponds to a list of metrics
    """

    metrics_lists = {
        "mse": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    num_files = len(os.listdir(pred_dir))
    for step in tqdm(range(1, num_files + 1)):
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
    Plot the metrics per step and cumulative average.

    Args:
        csv_file: path to the CSV file

    Returns:
        None
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
