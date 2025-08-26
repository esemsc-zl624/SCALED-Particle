import pandas as pd
import os
import argparse


def main(args):

    all_results = []
    onestep_results = []
    rollout_results = []

    for exp_dir in args.exp_dirs:
        onestep_results.append(os.path.join(exp_dir, "onestep", "metrics.csv"))
        rollout_results.append(os.path.join(exp_dir, "rollout", "metrics.csv"))

    for file_path in onestep_results:
        df = pd.read_csv(file_path)

        result_row = {
            "mode": "onestep",
            "file_path": file_path,
            "recall": df["recall"].mean(),
            "precision": df["precision"].mean(),
            "f1_score": df["f1"].mean(),
            "mse": df["mse"].mean(),
        }
        all_results.append(result_row)

    for file_path in rollout_results:
        df = pd.read_csv(file_path)

        result_row = {
            "mode": "rollout",
            "file_path": file_path,
            "recall": df["recall"].mean(),
            "precision": df["precision"].mean(),
            "f1_score": df["f1"].mean(),
            "mse": df["mse"].mean(),
        }
        all_results.append(result_row)

    result_df = pd.DataFrame(all_results)

    numeric_columns = ["recall", "precision", "f1_score", "mse"]
    for col in numeric_columns:
        result_df[col] = result_df[col].apply(lambda x: f"{x:.3e}")

    result_df.to_csv(args.result_save_path, index=False)

    print(result_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dirs",
        default=["exp/01_conv", "exp/02_attn", "exp/03_conv_sfc", "exp/04_attn_sfc"],
    )
    parser.add_argument("--result_save_path", default="result/summary.csv")

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.result_save_path), exist_ok=True)

    main(args)
