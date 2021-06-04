#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 1.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 3.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 5.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 1.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 3.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 5.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 1.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 3.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 5.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-5-elu-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 1.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-5-elu-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-5-elu-wscalar-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 3.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-5-elu-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-5-elu-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 5.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-5-elu-convex-seed-0-final --seed 0 &


wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-20-elu-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 20 --sparse_ratio 1.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-20-elu-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-20-elu-wscalar-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 20 --sparse_ratio 3.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-20-elu-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-20-elu-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 20 --sparse_ratio 5.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-20-elu-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 40 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-40-elu-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 40 --sparse_ratio 1.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-40-elu-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 40 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-40-elu-wscalar-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 40 --sparse_ratio 3.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-40-elu-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 40 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-40-elu-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 40 --sparse_ratio 5.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-40-elu-convex-seed-0-final --seed 0 &
