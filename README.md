# SCALED-X: Extending SCALED to Multi-Physics Problems

**SCALED-X** is a diffusion-based generative framework for particle-based physical simulation. It leverages structured representations, space-filling curve positional encodings, and attention mechanisms to achieve efficient and scalable simulation without explicit graph construction. SCALED-X enables accurate one-step predictions and stable multi-step rollouts for complex particle–fluid interactions.

# Installation

## 1. Using uv (Recommended)

```bash
pip install uv

cd <your_path_to>/SCALED-Particle
uv sync
source .venv/bin/activate
uv pip install -e .
```

## 2. Using Conda

```bash
conda create -n scaled python=3.10 -y
conda activate scaled

cd <your_path_to>/SCALED-Particle
pip install -e .
```

# Dataset and Model Checkpoint

The dataset and pretrained model checkpoints can be downloaded from [OneDrive](https://imperiallondon-my.sharepoint.com/:f:/g/personal/zl624_ic_ac_uk/EoGYl49nPsBPl0fTYsmrCfwBleRQzOaegHZ74jFS2Tpzzg?e=p6KiTp).

**Note:** Please place the dataset under `data/couple_spout_3D`, since many scripts assume this as the default path.

# Model Training

## 1. Prepare Your Experiment Directory

```
your_exp/
├── ckpt/
├── config/
│   └── config.yaml
└── logs/
```

## 2. Run Training Script

Modify the path to your config file (`your_exp/config/config.yaml`) inside `script/train.sh`.

```bash
bash script/train.sh
```

# Model Evaluation

## 1. Prepare Validation Dataset

After preparing `data/couple_spout_3D`, run the first cell in `/root/autodl-tmp/SCALED-Particle/data_processing.ipynb`: **"Generate Validation .npy files"**

## 2. Run Evaluation Script

Two scripts are provided for evaluation:

* `script/eval.sh`: Evaluates a single model by computing one-step and rollout metrics (Recall, Precision, F1 Score, MSE), and visualizes rollout results as GIFs.
* `script/benchmarking.sh`: Compares metrics across multiple models and saves results to `result/summary.csv`.

Simply update the `EXP_DIR` in the corresponding script.

# Model Inference

Run inference with **SCALED-X** using a config file and pretrained weights:

## Usage

```bash
python src/scaled/tools/inference/inference.py --config <path_to_config.yaml> \
              --weight_path <path_to_weights.pth> \
              --inference_type <rollout|onestep|long_rollout|debug_rollout>
```

### Arguments

* `--config` : Path to YAML config file. Inference-related and dataset-related parameters can be modified inside.
* `--weight_path` : Path to pretrained weights checkpoint. If not provided, the script will automatically load the latest checkpoint from `<your_path_to>/<your_exp>/ckpt`.
* `--inference_type` : Inference mode (default: `rollout`):
  * `rollout` – Standard multi-step simulation
  * `onestep` – Single-step prediction
  * `long_rollout` – Long-horizon rollout
  * `debug_rollout` – Debugging rollout

## Output

Results for each time step will be saved as `.npy` files under:

```
<your_path_to>/<your_exp>/<inference_type>/npy
```

These `.npy` files are used for subsequent visualization.

