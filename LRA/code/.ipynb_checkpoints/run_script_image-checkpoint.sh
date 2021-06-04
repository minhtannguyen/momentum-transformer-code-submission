#!/bin/bash

CUDA_VISIBLE_DEVICES=7 python3 run_tasks.py --model "softmax" --task image --use_wandb --project_name 'lra' --job_name image-10m-softmax-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "linear" --task image --use_wandb --project_name 'lra' --job_name image-10m-linear-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "sparselowrank" --task image --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name image-10m-sparselowrank-diag-size-5-seed-0-final --seed 0 

wait
echo "Continue"


CUDA_VISIBLE_DEVICES=7 python3 run_tasks.py --model "softmax" --task image --use_wandb --project_name 'lra' --job_name image-10m-softmax-seed-0-final --seed 0 --skip_train 1
