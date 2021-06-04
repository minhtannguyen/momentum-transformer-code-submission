#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python3 run_tasks.py --model "momentum" --task retrieval --mu 0.7 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-0-7-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=5 python3 run_tasks.py --model "momentum" --task retrieval --mu 0.8 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-0-8-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=7 python3 run_tasks.py --model "momentum" --task retrieval --mu 0.5 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-0-5-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=7 python3 run_tasks.py --model "momentum" --task retrieval --mu 0.4 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-0-4-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=1 python3 run_tasks.py --model "momentum" --task retrieval --mu 0.6 --stepsize 0.99 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-0-6-stepsize-0-99-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=1 python3 run_tasks.py --model "momentum" --task retrieval --mu 0.6 --stepsize 1.5 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-0-6-stepsize-1-5-seed-0-final --seed 0 &
