CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "softmax" --task listops


python3 run_tasks.py --model $1 --task listops
python3 run_tasks.py --model $1 --task text
python3 run_tasks.py --model $1 --task retrieval
python3 run_tasks.py --model $1 --task image
python3 run_tasks.py --model $1 --task pathfinder32-curv_contour_length_14


CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "softmax" --task text

CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "linear" --task text

CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "performer-256" --task text

CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "linformer-256" --task text

CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "reformer-2" --task text

CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "nystrom-64" --task text

CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "nystrom-32" --task text

CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "nystrom-128" --task text

CUDA_VISIBLE_DEVICES=2,3 python3 run_tasks.py --model "nystrom-256" --task text




dict_keys(['softmax', 'nystrom-32', 'nystrom-64', 'nystrom-128', 'nystrom-256', 'linformer-256', 'reformer-2', 'performer-256', 'linear'])


CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-elu-flip-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=5 python3 run_tasks.py --model "softmax" --task text --use_wandb --project_name 'lra' --job_name text-softmax-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=6 python3 run_tasks.py --model "linear" --task text --use_wandb --project_name 'lra' --job_name text-linear-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 1024 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-1024-seed-0-final --seed 0


CUDA_VISIBLE_DEVICES=7 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 7.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name retrieval-sparselowrank-elu-elu-flip-lowrankonly-seed-0-final --seed 0



CUDA_VISIBLE_DEVICES=3 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 8.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name pathfinder32-sparselowrank-diag-size-5-elu-elu-flip-ws-al1-bl2-onlowrank-bs-16-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=5 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 9.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name pathfinder32-sparselowrank-diag-size-5-elu-elu-flip-s-w-al1-bl2-onlowrank-bs-16-seed-0-final --seed 0



CUDA_VISIBLE_DEVICES=6 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 3.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name pathfinder32-sparselowrank-diag-size-5-elu-elu-flip-onsparse-seed-0-final --seed 0 

CUDA_VISIBLE_DEVICES=7 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name pathfinder32-sparselowrank-diag-size-5-elu-elu-flip-onlowrank-seed-0-final --seed 0 


CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "softmax" --task image --use_wandb --project_name 'debug' --job_name cifar-softmax-seed-0-final --seed 0 


CUDA_VISIBLE_DEVICES=6 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 10.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name pathfinder32-sparselowrank-diag-size-5-elu-w0-w1-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=4 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 11.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name pathfinder32-sparselowrank-diag-size-5-elu-w1-w0-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 60 --sparse_ratio 11.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-60-elu-w1-w0-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 40 --sparse_ratio 11.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-40-elu-elu-flip-w1-w0-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 40 --sparse_ratio 13.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name listops-sparselowrank-diag-size-40-elu-elu-flip-w1-w0-a1-b0-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=0 python3 cifar10.py


CUDA_VISIBLE_DEVICES=6 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name pathfinder32-sparselowrank-diag-size-5-seed-0-final --seed 0



CUDA_VISIBLE_DEVICES=5 python3 run_tasks_test.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' --job_name listops-test --seed 0

CUDA_VISIBLE_DEVICES=5 python3 run_tasks_test.py --model "softmax" --task text --job_name listops-test --seed 0


CUDA_VISIBLE_DEVICES=4,5,6,7 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name text-sparselowrank-diag-size-5-elu-onlowrank-seed-1-final --seed 1 &


CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 15.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'lra' --job_name pathfinder32-sparselowrank-diag-size-5-elu-elu-flip-wscalar-convex-b0-seed-0-final --seed 0 &


CUDA_VISIBLE_DEVICES=5 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'lra' --job_name pathfinder32-sparselowrank-diag-size-5-elu-wscalar-convex-seed-0-final --seed 0


CUDA_VISIBLE_DEVICES=5 python3 run_tasks.py --model "linear" --task text --use_wandb --project_name 'debug' --job_name debug --seed 0



CUDA_VISIBLE_DEVICES=5 python3 run_tasks.py --model "momentum" --task text --mu 0.9 --stepsize 0.4 --use_wandb --project_name 'lra' --job_name text-momentum-mu-0-9-stepsize-0-4-seed-0-final --seed 0

CUDA_VISIBLE_DEVICES=5 python3 run_tasks.py --model "momentum" --task text --mu 0.9 --stepsize 0.4 --use_wandb --project_name 'debug' --job_name debug --seed 0


CUDA_VISIBLE_DEVICES=5 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.9 --stepsize 0.4 --res_stepsize 0.1 --res_delta 0.0001 --use_wandb --project_name 'debug' --job_name debug --seed 0


with open(f"image.{component}.10m.pickle", "wb") as f: pickle.dump(ds_list, f)

"res_stepsize": 1.0,
"res_delta": 0.0001