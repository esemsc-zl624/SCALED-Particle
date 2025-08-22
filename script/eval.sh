EXP_DIR=exp/exp4_gtbc

# uv
# source /scratch_dgxl/zl624/workspace/SCALED-Particle/.venv/bin/activate

# conda
source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh && conda activate scaled

# onestep performance on 200-249 steps
# python src/scaled/tools/inference/inference.py --config $EXP_DIR/config/exp4.yaml --inference_type onestep
# python src/scaled/tools/evaluation/calculate_mse.py --pred_dir $EXP_DIR/onestep/npy

# rollout performance on 200-249 steps
# python src/scaled/tools/inference/inference.py --config $EXP_DIR/config/exp4.yaml --inference_type rollout
python src/scaled/tools/evaluation/calculate_mse.py --pred_dir $EXP_DIR/rollout/npy

# rollout performance on whole dataset (long rollout step=250)
# python src/scaled/tools/inference/inference.py --config $EXP_DIR/config/exp4.yaml --inference_type long_rollout

# python src/scaled/tools/visualization/viz_rollout_comparation.py --pred_input_dir $EXP_DIR/rollout/npy --output_dir $EXP_DIR/rollout

