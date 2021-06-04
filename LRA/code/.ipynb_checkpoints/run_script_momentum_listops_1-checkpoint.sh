#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.1 --stepsize 0.1 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-1-stepsize-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.1 --stepsize 0.3 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-1-stepsize-0-3-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.1 --stepsize 0.6 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-1-stepsize-0-6-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.1 --stepsize 0.9 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-1-stepsize-0-9-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.1 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-1-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.1 --stepsize 2.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-1-stepsize-2-0-seed-0-final --seed 0 &

wait
echo "Continue"


CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.3 --stepsize 0.1 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-3-stepsize-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.3 --stepsize 0.3 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-3-stepsize-0-3-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.3 --stepsize 0.6 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-3-stepsize-0-6-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.3 --stepsize 0.9 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-3-stepsize-0-9-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.3 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-3-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.3 --stepsize 2.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-3-stepsize-2-0-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.6 --stepsize 0.1 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-6-stepsize-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.6 --stepsize 0.3 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-6-stepsize-0-3-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.6 --stepsize 0.6 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-6-stepsize-0-6-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.6 --stepsize 0.9 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-6-stepsize-0-9-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.6 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-6-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-6-stepsize-2-0-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.9 --stepsize 0.1 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-9-stepsize-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.9 --stepsize 0.3 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-9-stepsize-0-3-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.9 --stepsize 0.6 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-9-stepsize-0-6-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.9 --stepsize 0.9 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-9-stepsize-0-9-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.9 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-9-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 0.9 --stepsize 2.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-9-stepsize-2-0-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 1.0 --stepsize 0.1 --use_wandb --project_name 'lra' --job_name text-momentum-mu-1-0-stepsize-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 1.0 --stepsize 0.3 --use_wandb --project_name 'lra' --job_name text-momentum-mu-1-0-stepsize-0-3-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 1.0 --stepsize 0.6 --use_wandb --project_name 'lra' --job_name text-momentum-mu-1-0-stepsize-0-6-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 1.0 --stepsize 0.9 --use_wandb --project_name 'lra' --job_name text-momentum-mu-1-0-stepsize-0-9-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 1.0 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-1-0-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task text --mu 1.0 --stepsize 2.0 --use_wandb --project_name 'lra' --job_name text-momentum-mu-1-0-stepsize-2-0-seed-0-final --seed 0 &

wait
echo "Continue"
