import pandas as pd
import os

# onestep_results = ["exp/01_conv/onestep/metrics.csv", "exp/02_attn/onestep/metrics.csv", "exp/03_conv_sfc/onestep/metrics.csv", "exp/04_attn_sfc/onestep/metrics.csv"]
# rollout_results = ["exp/01_conv/rollout/metrics.csv", "exp/02_attn/rollout/metrics.csv", "exp/03_conv_sfc/rollout/metrics.csv", "exp/04_attn_sfc/rollout/metrics.csv"]
onestep_results = ["exp/01_conv/onestep/metrics.csv", "exp/02_attn/onestep/metrics.csv"]
rollout_results = ["exp/01_conv/rollout/metrics.csv", "exp/02_attn/rollout/metrics.csv"]

os.makedirs("result", exist_ok=True)
result_dir = "result/summary.csv"

# 创建空的DataFrame来存储所有结果
all_results = []

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

numeric_columns = ['recall', 'precision', 'f1_score', 'mse']
for col in numeric_columns:
    result_df[col] = result_df[col].apply(lambda x: f"{x:.3e}")

result_df.to_csv(result_dir, index=False)

print(result_df)
