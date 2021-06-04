#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.1 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-1-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.2 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-2-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.3 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-3-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.4 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-4-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.5 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-5-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.6 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-6-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.7 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-7-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.8 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-8-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.9 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-9-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 1.0 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-1-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 2.0 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-2-0-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.01 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-01-seed-0-final --seed 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.99 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-99-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.999 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-999-seed-0-final --seed 0 &

CUDA_VISIBLE_DEVICES=6 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.001 --res_delta 0.0001 --use_wandb --project_name 'lra' --job_name image-adaptive-momentum-mu-0-9-stepsize-0-9-rstep-0-001-seed-0-final --seed 0 &

wait
echo "Continue"

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.1 --res_stepsize 0.1 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-1-rstep-0-1-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.1 --res_stepsize 0.3 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-1-rstep-0-3-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.1 --res_stepsize 0.6 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-1-rstep-0-6-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.1 --res_stepsize 0.9 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-1-rstep-0-9-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.1 --res_stepsize 1.0 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-1-rstep-1-0-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.1 --res_stepsize 2.0 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-1-rstep-2-0-seed-0-final --seed 0 &

# wait
# echo "Continue"


# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.3 --res_stepsize 0.1 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-3-rstep-0-1-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.3 --res_stepsize 0.3 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-3-rstep-0-3-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.3 --res_stepsize 0.6 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-3-rstep-0-6-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.3 --res_stepsize 0.9 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-3-rstep-0-9-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.3 --res_stepsize 1.0 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-3-rstep-1-0-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.3 --res_stepsize 2.0 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-3-rstep-2-0-seed-0-final --seed 0 &

# wait
# echo "Continue"


# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.6 --res_stepsize 0.1 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-6-rstep-0-1-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.6 --res_stepsize 0.3 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-6-rstep-0-3-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.6 --res_stepsize 0.6 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-6-rstep-0-6-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.6 --res_stepsize 0.9 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-6-rstep-0-9-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.6 --res_stepsize 1.0 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-6-rstep-1-0-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.6 --res_stepsize 2.0 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-6-rstep-2-0-seed-0-final --seed 0 &

# wait
# echo "Continue"


# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.9 --res_stepsize 0.1 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-9-rstep-0-1-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.9 --res_stepsize 0.3 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-9-rstep-0-3-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.9 --res_stepsize 0.6 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-9-rstep-0-6-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.9 --res_stepsize 0.9 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-9-rstep-0-9-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.9 --res_stepsize 1.0 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-9-rstep-1-0-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=6 python3 run_tasks_momentum.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_mu 0.9 --res_stepsize 2.0 --use_wandb --project_name 'lra' --job_name image-momentum-momentum-mu-0-9-stepsize-0-9-rmu-0-9-rstep-2-0-seed-0-final --seed 0 &

# wait
# echo "Continue"
