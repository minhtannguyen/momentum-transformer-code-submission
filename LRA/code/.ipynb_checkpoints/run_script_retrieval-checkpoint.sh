#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 10.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-w0-w1-head-seed-0-final --seed 0


# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-elu-flip-wscalar-onlowrank-bs-16-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 8.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-elu-flip-ws-al1-bl2-onlowrank-bs-16-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 9.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-elu-flip-s-w-al1-bl2-onlowrank-bs-16-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 1.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-wscalar-onlowrank-bs-16-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "softmax" --task retrieval --use_wandb --project_name 'lra' --job_name retrieval-softmax-bs-16-seed-0-final --seed 0 

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "linear" --task retrieval --use_wandb --project_name 'lra' --job_name retrieval-linear-bs-16-seed-0-final --seed 0 


# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-wscalar-onlowrank-bs-16-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 30 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-30-elu-elu-flip-wscalar-onlowrank-bs-16-seed-0-final --seed 0




# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 3.5 --kernels 'elu' --use_wandb --project_name 'debug' --job_name debug --seed 0 



# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 3.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-onsparse-seed-0-final --seed 0 

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-onlowrank-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 5.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-convex-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-wscalar-onsparse-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 1.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-wscalar-onlowrank-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-wscalar-convex-seed-0-final --seed 0


# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 3.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-onsparse-seed-0-final --seed 0 

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-onlowrank-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 5.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-convex-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 1.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-wscalar-convex-seed-0-final --seed 0


# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-seed-0-final --seed 0 &













# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 1 --sparse_ratio 7.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-elu-elu-flip-lowrank-only-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 1 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-1-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 1 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-1-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 1 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-1-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 1 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-1-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 1 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-1-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 1 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-1-elu-elu-flip-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 1 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-1-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-elu-elu-flip-convex-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-5-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 10 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-10-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 10 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-10-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 10 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-10-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 10 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-10-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 10 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-10-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 10 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-10-elu-elu-flip-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 10 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-10-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-elu-elu-flip-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 20 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-20-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 30 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-30-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 30 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-30-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 30 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-30-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 30 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-30-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 30 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-30-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 30 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-30-elu-elu-flip-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 30 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-30-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 40 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-40-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 40 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-40-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 40 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-40-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 40 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-40-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 40 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-40-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 40 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-40-elu-elu-flip-convex-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 40 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-40-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 50 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-50-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 50 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-50-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 50 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-50-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 50 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-50-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 50 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-50-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 50 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-50-elu-elu-flip-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 50 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-50-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 60 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-60-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 60 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-60-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 60 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-60-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 60 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-60-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 60 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-60-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 60 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-60-elu-elu-flip-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 60 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-60-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 70 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-70-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 70 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-70-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 70 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-70-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 70 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-70-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 70 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-70-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 70 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-70-elu-elu-flip-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 70 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-70-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 80 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-80-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 80 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-80-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 80 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-80-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 80 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-80-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 80 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-80-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 80 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-80-elu-elu-flip-convex-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 80 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-80-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 90 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-90-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 90 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-90-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 90 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-90-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 90 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-90-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 90 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-90-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 90 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-90-elu-elu-flip-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 90 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-90-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 100 --sparse_ratio 0.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-100-elu-elu-flip-wscalar-onsparse-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 100 --sparse_ratio 1.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-100-elu-elu-flip-wscalar-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 100 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-100-elu-elu-flip-wscalar-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 100 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-100-elu-elu-flip-onsparse-seed-0-final --seed 0 &

# wait
# echo "Continue"

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 100 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-100-elu-elu-flip-onlowrank-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=1,2,3 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 100 --sparse_ratio 5.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-100-elu-elu-flip-convex-seed-0-final --seed 0 &

# CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 100 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-diag-size-100-seed-0-final --seed 0 &


# wait
# echo "Continue"




