# uv
# source .venv/bin/activate

# conda
# source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh
source /root/miniconda3/etc/profile.d/conda.sh
conda activate scaled

EXP_DIR=("exp/01_conv" "exp/02_attn" "exp/03_conv_sfc" "exp/04_attn_sfc")

for dir in "${EXP_DIR[@]}"; do
    # onestep performance on 200-249 steps
    python src/scaled/tools/inference/inference.py --config "$dir/config/config.yaml" --inference_type onestep
    python src/scaled/tools/evaluation/calculate_metrics.py --gt_dir "data/evaluation_npy_step200-250" --pred_dir "$dir/onestep/npy"

    # rollout performance on 200-249 steps
    python src/scaled/tools/inference/inference.py --config "$dir/config/config.yaml" --inference_type rollout
    python src/scaled/tools/evaluation/calculate_metrics.py --gt_dir "data/evaluation_npy_step200-250" --pred_dir "$dir/rollout/npy"
    python src/scaled/tools/visualization/viz_rollout_comparation.py --pred_input_dir "$dir/rollout/npy" --output_dir "$dir/rollout"
done

python src/scaled/tools/evaluation/collect_result.py
