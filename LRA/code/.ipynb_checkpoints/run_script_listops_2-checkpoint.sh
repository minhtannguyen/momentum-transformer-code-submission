#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 10.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-5-elu-w0-w1-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 10.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-5-elu-elu-flip-w0-w1-seed-0-final --seed 0


wait
echo "Continue"
