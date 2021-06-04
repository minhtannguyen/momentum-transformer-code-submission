#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1-elu-elu-flip-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1-elu-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-elu-flip-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-scalarw-seed-0-final --seed 0 &


wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 10 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-10-elu-elu-flip-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 10 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-10-elu-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-elu-flip-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-scalarw-seed-0-final --seed 0 &


wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 30 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-30-elu-elu-flip-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 30 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-30-elu-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-elu-flip-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-scalarw-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 50 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-50-elu-elu-flip-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 50 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-50-elu-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 60 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-60-elu-elu-flip-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 60 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-60-elu-scalarw-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 100 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-100-elu-elu-flip-scalarw-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 100 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-100-elu-scalarw-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 200 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-200-elu-elu-flip-scalarw-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 200 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-200-elu-scalarw-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 400 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-400-elu-elu-flip-scalarw-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 400 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-400-elu-scalarw-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 800 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-800-elu-elu-flip-scalarw-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 800 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-800-elu-scalarw-seed-0-final --seed 0 &

wait
echo "Continue"




