import os
import matplotlib.pyplot as plt
import pandas as pd


def plot_multiple_models(csv_files, model_names=None, save_dir=None):
    """
    Args:
        csv_files: list of csv file paths
        model_names: list of model names for legend, default: use file basename
        save_dir: directory to save plots, default: same as first csv
    """
    if model_names is None:
        model_names = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]

    if save_dir is None:
        save_dir = os.path.dirname(csv_files[0])
    os.makedirs(save_dir, exist_ok=True)

    # Plot MSE per step
    plt.figure(figsize=(8, 5))
    for csv_file, name in zip(csv_files, model_names):
        df = pd.read_csv(csv_file)
        plt.plot(df["error"], label=name)
    plt.xlabel("Step")
    plt.ylabel("MSE")
    plt.title("MSE per step")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(save_dir, "mse_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot Cumulative MSE
    plt.figure(figsize=(8, 5))
    for csv_file, name in zip(csv_files, model_names):
        df = pd.read_csv(csv_file)
        plt.plot(df["cumulative_avg"], label=name)
    plt.xlabel("Step")
    plt.ylabel("Cumulative MSE")
    plt.title("Cumulative Average MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(save_dir, "cumulative_mse_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


csv_files = [
    "exp/01_conv/rollout/mse.csv",
    "exp/02_attn/rollout/mse.csv",
    "exp/03_conv_sfc/rollout/mse.csv",
    "exp/04_attn_sfc/rollout/mse.csv",
]
model_names = ["SCALED(conv)", "SCALED(attn)", "SCALED(conv)+SFC", "SCALED(attn)+SFC"]
plot_multiple_models(csv_files, model_names)
