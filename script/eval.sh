EXP_DIR=exp/01_conv
# EXP_DIR=exp/02_attn
# EXP_DIR=exp/03_conv_sfc
# EXP_DIR=exp/04_attn_sfc

# uv
# source /scratch_dgxl/zl624/workspace/SCALED-Particle/.venv/bin/activate

# conda
# source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh && conda activate scaled
source /root/miniconda3/etc/profile.d/conda.sh
conda activate scaled

# onestep performance on 200-249 steps
python src/scaled/tools/inference/inference.py --config $EXP_DIR/config/config.yaml --inference_type onestep
python src/scaled/tools/evaluation/calculate_metrics.py --pred_dir $EXP_DIR/onestep/npy

# rollout performance on 200-249 steps
python src/scaled/tools/inference/inference.py --config $EXP_DIR/config/config.yaml --inference_type rollout
python src/scaled/tools/evaluation/calculate_metrics.py --pred_dir $EXP_DIR/rollout/npy
python src/scaled/tools/visualization/viz_rollout_comparation.py --pred_input_dir $EXP_DIR/rollout/npy --output_dir $EXP_DIR/rollout

# debug
# python src/scaled/tools/inference/inference.py --config $EXP_DIR/config/debug.yaml --inference_type debug_rollout
# python src/scaled/tools/evaluation/calculate_metrics.py --pred_dir $EXP_DIR/debug_rollout/npy
# python src/scaled/tools/visualization/viz_rollout_comparation.py --pred_input_dir $EXP_DIR/debug_rollout/npy --output_dir $EXP_DIR/debug_rollout

# rollout performance on whole dataset (long rollout step=250)
# python src/scaled/tools/inference/inference.py --config $EXP_DIR/config/config.yaml --inference_type long_rollout
# python src/scaled/tools/evaluation/calculate_metrics.py --gt_dir data/evaluation_npy_step1-250 --pred_dir $EXP_DIR/long_rollout/npy
# python src/scaled/tools/visualization/viz_rollout_comparation.py --pred_input_dir $EXP_DIR/long_rollout/npy --output_dir $EXP_DIR/long_rollout


