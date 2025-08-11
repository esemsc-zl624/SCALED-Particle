EXP_DIR=exp/exp2_boundary

# uv
# source /scratch_dgxl/zl624/workspace/SCALED-Particle/.venv/bin/activate

# conda
source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh && conda activate scaled

# python tools/inference/inference.py --config exp/exp2_boundary/config/exp2.yaml --weight_path exp/exp2_boundary/ckpt/denoising_unet-190000.pth

python tools/visualization/viz_rollout_comparation.py --pred_input_dir $EXP_DIR/inference/npy --output_dir $EXP_DIR/inference/rollout