
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

Run following commands to reproduce results in Table 5 in our paper.

Listops
```
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "softmax" --task listops --use_wandb --project_name 'momentum-transformer' --job_name listops-softmax --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "linear" --task listops --use_wandb --project_name 'momentum-transformer' --job_name listops-linear --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "momentum" --task listops --mu 0.1 --stepsize 0.6 --use_wandb --project_name 'momentum-transformer' --job_name listops-momentum --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks_adaptive.py --model "momentum" --task listops --mu 0.1 --stepsize 0.6 --res_stepsize 0.4 --res_delta 0.0001 --use_wandb --project_name 'momentum-transformer' --job_name listops-adaptive-momentum --seed 0 
```

Text
```
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "softmax" --task text --use_wandb --project_name 'momentum-transformer' --job_name text-softmax --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "linear" --task text --use_wandb --project_name 'momentum-transformer' --job_name text-linear --seed 0 
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --use_wandb --project_name 'momentum-transformer' --job_name text-momentum --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks_adaptive.py --model "momentum" --task text --mu 0.6 --stepsize 2.0 --res_stepsize 0.001 --res_delta 0.0001 --use_wandb --project_name 'momentum-transformer' --job_name text-adaptive-momentum --seed 0
```

Retrieval
```
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "softmax" --task retrieval --use_wandb --project_name 'momentum-transformer' --job_name retrieval-softmax --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "linear" --task retrieval --use_wandb --project_name 'momentum-transformer' --job_name retrieval-linear --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "momentum" --task retrieval --mu 0.6 --stepsize 1.0 --use_wandb --project_name 'momentum-transformer' --job_name retrieval-momentum --seed 0 
CUDA_VISIBLE_DEVICES=0 python3 run_tasks_adaptive.py --model "momentum" --task retrieval --mu 0.6 --stepsize 1.0 --res_stepsize 0.5 --res_delta 0.0001 --use_wandb --project_name 'momentum-transformer' --job_name retrieval-adaptive-momentum --seed 0
```

Image
```
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "softmax" --task image --use_wandb --project_name 'momentum-transformer' --job_name image-softmax --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "linear" --task image --use_wandb --project_name 'momentum-transformer' --job_name image-linear --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --use_wandb --project_name 'momentum-transformer' --job_name image-momentum --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks_adaptive.py --model "momentum" --task image --mu 0.9 --stepsize 0.9 --res_stepsize 0.001 --res_delta 0.0001 --use_wandb --project_name 'momentum-transformer' --job_name image-adaptive-momentum --seed 0
```

Pathfinder
```
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "softmax" --task pathfinder32-curv_contour_length_14 --use_wandb --project_name 'momentum-transformer' --job_name pathfinder-softmax --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "linear" --task pathfinder32-curv_contour_length_14 --use_wandb --project_name 'momentum-transformer' --job_name pathfinder-linear --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks.py --model "momentum" --task pathfinder32-curv_contour_length_14 --mu 0.3 --stepsize 0.1 --use_wandb --project_name 'momentum-transformer' --job_name pathfinder-momentum --seed 0
CUDA_VISIBLE_DEVICES=0 python3 run_tasks_adaptive.py --model "momentum" --task pathfinder32-curv_contour_length_14 --mu 0.3 --stepsize 0.1 --res_stepsize 0.8 --res_delta 0.0001 --use_wandb --project_name 'momentum-transformer' --job_name pathfinder-adaptive --seed 0
```
