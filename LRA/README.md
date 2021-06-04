
## LRA Benchmark

To prepare the datasets, one would need
```
tensorboard>=2.3.0, tensorflow>=2.3.1, tensorflow-datasets>=4.0.1
```

To prepare the datasets, one would need to download the source code from [LRA repo](https://github.com/google-research/long-range-arena) and place `long-range-arena` folder in folder `LRA/datasets/` and also download [lra_release.gz](https://storage.googleapis.com/long-range-arena/lra_release.gz) released by LRA repo and place the unzipped folder in folder `LRA/datasets/`. The directory structure would be
```
LRA/datasets/long-range-arena
LRA/datasets/lra_release
```
Then, run `sh create_datasets.sh` and it will create train, dev, and test dataset pickle files for each task.

To run the LRA tasks, one would need
```
pytorch==1.7.1, transformers==3.3.1, performer-pytorch
```

Run following commands to reproduce results in Table 1 in our paper.

Listops
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "softmax" --task listops --use_wandb --project_name 'fmmformer' --job_name listops-softmax --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "linear" --task listops --use_wandb --project_name 'fmmformer' --job_name listops-linear --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name listops-fmmformer-band-5 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 11.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name listops-fmmformer-1-kernel-band-5 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task listops --diag_size 5 --sparse_ratio 13.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'fmmformer' --job_name listops-fmmformer-2-kernel-band-5 --seed 0
```

Text
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "softmax" --task text --use_wandb --project_name 'fmmformer' --job_name text-softmax --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "linear" --task text --use_wandb --project_name 'fmmformer' --job_name text-linear --seed 0 
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name text-fmmformer-band-5 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name text-fmmformer-1-kernel-band-5 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task text --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'fmmformer' --job_name text-fmmformer-2-kernel-band-5 --seed 0
```

Retrieval
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "softmax" --task retrieval --use_wandb --project_name 'fmmformer' --job_name retrieval-softmax --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "linear" --task retrieval --use_wandb --project_name 'fmmformer' --job_name retrieval-linear --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name retrieval-fmmformer-band-5 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 18.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name retrieval-fmmformer-1-kernel-band-5 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task retrieval --diag_size 5 --sparse_ratio 19.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'fmmformer' --job_name retrieval-fmmformer-2-kernel-band-5 --seed 0
```

Image
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "softmax" --task image --use_wandb --project_name 'fmmformer' --job_name image-softmax --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "linear" --task image --use_wandb --project_name 'fmmformer' --job_name image-linear --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task image --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name image-fmmformer-band-5 --seed 0 
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task image --diag_size 5 --sparse_ratio 4.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name image-fmmformer-1-kernel-band-5 --seed 0 
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task image --diag_size 5 --sparse_ratio 17.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'fmmformer' --job_name image-fmmformer-2-kernel-band-5 --seed 0
```

Pathfinder
```
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "softmax" --task pathfinder32-curv_contour_length_14 --use_wandb --project_name 'fmmformer' --job_name pathfinder-softmax --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "linear" --task pathfinder32-curv_contour_length_14 --use_wandb --project_name 'fmmformer' --job_name pathfinder-linear --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 6.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name pathfinder-fmmformer-band-5 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 2.5 --kernels 'elu' --use_wandb --project_name 'fmmformer' --job_name pathfinder-fmmformer-1-kernel-band-5 --seed 0
CUDA_VISIBLE_DEVICES=0,1 python3 run_tasks.py --model "sparselowrank" --task pathfinder32-curv_contour_length_14 --diag_size 5 --sparse_ratio 15.5 --kernels 'elu' 'elu_flip' --use_wandb --project_name 'fmmformer' --job_name pathfinder-fmmformer-2-kernel-band-5 --seed 0
```
