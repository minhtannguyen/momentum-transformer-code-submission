#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.1 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.2 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-2-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.3 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-3-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.4 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-4-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.5 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-5-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.6 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-6-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.7 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-7-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.8 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-8-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.9 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-9-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 1.0 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-1-0-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 2.0 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-2-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.01 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-01-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.001 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-001-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.99 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-99-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.999 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-999-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.1 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-1-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.2 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-2-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.3 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-3-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.4 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-4-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.5 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-5-delta-0-1-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.6 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-6-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.7 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-7-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.8 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-8-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.9 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-9-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 1.0 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-1-0-delta-0-1-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 2.0 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-2-0-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.01 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-01-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.001 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-001-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.99 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-99-delta-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.999 --res_delta 0.1 --use_wandb --project_name 'lra' --job_name text-adaptive-momentum-mu-0-6-stepsize-2-0-rstep-0-999-delta-0-1-seed-0-final --seed 0 &

wait
echo "Continue"