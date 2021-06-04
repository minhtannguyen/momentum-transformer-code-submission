#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task retrieval --mu 1.0 --stepsize 0.1 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-1-0-stepsize-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task retrieval --mu 1.0 --stepsize 0.3 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-1-0-stepsize-0-3-seed-0-final --seed 0 &

wait
echo "Continue"


CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task retrieval --mu 1.0 --stepsize 0.6 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-1-0-stepsize-0-6-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task retrieval --mu 1.0 --stepsize 0.9 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-1-0-stepsize-0-9-seed-0-final --seed 0 &

wait
echo "Continue"


CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task retrieval --mu 1.0 --stepsize 1.0 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-1-0-stepsize-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "momentum" --task retrieval --mu 1.0 --stepsize 2.0 --use_wandb --project_name 'lra' --job_name retrieval-momentum-mu-1-0-stepsize-2-0-seed-0-final --seed 0 &

wait
echo "Continue"
