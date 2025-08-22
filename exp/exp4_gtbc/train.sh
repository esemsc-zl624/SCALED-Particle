#!/bin/bash

#SBATCH --job-name=exp4_gtbc
#SBATCH --partition=dgxl_irp
#SBATCH --qos=dgxl_irp_low

#SBATCH -e exp/exp4_gtbc/logs/%j-exp4_gtbc.err              # File to redirect stderr
#SBATCH -o exp/exp4_gtbc/logs/%j-exp4_gtbc.out              # File to redirect stdout
#SBATCH --mem=10GB                   # Memory per processor
#SBATCH --time=24:00:00              # The walltime
#SBATCH --nodes=1                    # Run all processes on a single node
#SBATCH --ntasks-per-node=1          # number of MP tasks
#SBATCH --cpus-per-task=12           # CPUs per task
#SBATCH --gres=gpu:2                 # Number of GPUs



# uv
# source /scratch_dgxl/zl624/workspace/SCALED-Particle/.venv/bin/activate

# conda
source /scratch_dgxl/zl624/miniconda3/etc/profile.d/conda.sh && conda activate scaled

accelerate launch src/scaled/tools/trainning/train.py --config exp/exp4_gtbc/config/exp4.yaml