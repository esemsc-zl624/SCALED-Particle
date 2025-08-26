import os
import pandas as pd
import matplotlib.pyplot as plt

# ================ Configuration ================
file_path = "data/02_attn_loss.csv"  # Change to your file path
window = 20      # Moving Average window
save_path = "result/loss_curve.png"
# ==============================================

# Read CSV
if not os.path.exists(file_path):
    raise FileNotFoundError(f"CSV not found: {file_path}")
df = pd.read_csv(file_path)

# Automatically detect Step column (fallback to index)
step_candidates = [c for c in df.columns if c.lower() == "step" or "step" in c.lower()]
if step_candidates:
    step_col = step_candidates[0]
else:
    # If no Step column is found, use the index as step
    df = df.reset_index().rename(columns={"index": "Step"})
    step_col = "Step"

# Automatically detect loss column (prefer train_loss, otherwise any column containing 'loss', excluding __min/__max)
loss_candidates = [
    c for c in df.columns
    if ("loss" in c.lower()) and ("__min" not in c.lower()) and ("__max" not in c.lower())
]
preferred = [c for c in loss_candidates if "train" in c.lower()]
if preferred:
    loss_col = preferred[0]
elif loss_candidates:
    loss_col = loss_candidates[0]
else:
    raise ValueError("No loss-like column found. Please set loss_col manually.")

# Extract and clean data
steps = pd.to_numeric(df[step_col], errors="coerce")
loss = pd.to_numeric(df[loss_col], errors="coerce")
mask = steps.notna() & loss.notna()
steps = steps[mask]
loss = loss[mask]

# Compute smoothing (Moving Average)
ma = loss.rolling(window=window, min_periods=1).mean()

# Plot: raw (light) + MA (darker)
plt.figure(figsize=(12,6))
plt.plot(steps, loss, alpha=0.25, label="Raw Loss")               # raw loss
plt.plot(steps, ma, linewidth=1.6, label=f"MA (window={window})") # moving average

plt.xlabel("Step")
plt.ylabel("L1 Loss")
plt.title(f"Loss curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and show
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved figure to {save_path}")
