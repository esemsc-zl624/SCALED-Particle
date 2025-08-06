EXP_DIR=exp/exp1_add_attention

python tools/inference/inference.py --config $EXP_DIR/config/exp1.yaml --weight_path $EXP_DIR/ckpt/denoising_unet-100000.pth
python tools/visualization/viz_rollout.py --input_dir $EXP_DIR/inference/npy --output_dir $EXP_DIR/inference/rollout