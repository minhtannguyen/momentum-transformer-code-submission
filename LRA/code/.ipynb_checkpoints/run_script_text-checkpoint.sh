#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "softmax" --task text --use_wandb --project_name 'lra' --job_name text-softmax-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "linear" --task text --use_wandb --project_name 'lra' --job_name text-linear-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 7.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-elu-elu-flip-lowrank-only-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 10 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-10-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 10 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-10-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 10 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-10-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 10 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-10-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 10 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-10-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 10 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-10-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 10 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-10-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-elu-elu-flip-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 20 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-20-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 30 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-30-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 30 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-30-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 30 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-30-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 30 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-30-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 30 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-30-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 30 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-30-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 30 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-30-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-elu-flip-onsparse-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 40 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-40-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 50 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-50-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 50 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-50-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 50 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-50-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 50 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-50-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 50 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-50-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 50 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-50-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 50 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-50-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 60 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-60-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 60 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-60-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 60 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-60-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 60 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-60-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 60 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-60-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 60 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-60-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 60 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-60-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 70 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-70-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 70 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-70-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 70 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-70-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 70 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-70-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 70 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-70-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 70 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-70-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 70 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-70-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 80 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-80-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 80 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-80-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 80 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-80-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 80 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-80-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 80 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-80-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 80 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-80-elu-elu-flip-convex-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 80 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-80-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 90 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-90-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 90 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-90-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 90 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-90-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 90 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-90-elu-elu-flip-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 90 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-90-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 90 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-90-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 90 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-90-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 100 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-100-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 100 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-100-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 100 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-100-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 100 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-100-elu-elu-flip-onsparse-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 100 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-100-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 100 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-100-elu-elu-flip-convex-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 100 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-100-seed-0-final --seed 0 &


wait
echo "Continue"




