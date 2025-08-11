EXP_DIR=exp/exp2_boundary

# uv
# source /scratch_dgxl/zl624/workspace/SCALED-Particle/.venv/bin/activate

# conda
# source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh && conda activate scaled

python src/scaled/tools/inference/inference.py --config $EXP_DIR/config/exp2.yaml --weight_path $EXP_DIR/ckpt/denoising_unet-190000.pth

python src/scaled/tools/visualization/viz_rollout_comparation.py --pred_input_dir $EXP_DIR/rollout/npy --output_dir $EXP_DIR/rollout