#!/bin/bash

CUDA_VISIBLE_DEVICES=4,7 python3 run_tasks.py --model "softmax" --task text --use_wandb --project_name 'lra' --job_name text-softmax-seed-1-final --seed 1 

CUDA_VISIBLE_DEVICES=4,7 python3 run_tasks.py --model "linear" --task text --use_wandb --project_name 'lra' --job_name text-linear-seed-1-final --seed 1

CUDA_VISIBLE_DEVICES=4,7 python3 run_tasks.py --model "softmax" --task text --use_wandb --project_name 'lra' --job_name text-softmax-seed-2-final --seed 2 

CUDA_VISIBLE_DEVICES=4,7 python3 run_tasks.py --model "linear" --task text --use_wandb --project_name 'lra' --job_name text-linear-seed-2-final --seed 2




