srun --pty -p dgxl_irp --gres=gpu:1 --cpus-per-task=12 --mem=16G --qos=dgxl_irp_low bash

# uv
# source /scratch_dgxl/zl624/workspace/SCALED-Particle/.venv/bin/activate

# conda
source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh  && conda activate scaled
