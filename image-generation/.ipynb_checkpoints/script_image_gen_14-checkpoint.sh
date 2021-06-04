#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_causal_adaptive_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_adaptivemomentum_wang_causal_mu_0.1_step_0.9_rstep_0.1_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --res_stepsize 0.1 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 0 --gpu-id 0 &

python main_causal_adaptive_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_adaptivemomentum_wang_causal_mu_0.1_step_0.9_rstep_0.3_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --res_stepsize 0.3 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_causal_adaptive_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_adaptivemomentum_wang_causal_mu_0.1_step_0.9_rstep_0.6_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --res_stepsize 0.6 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 0 --gpu-id 1 &

python main_causal_adaptive_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_adaptivemomentum_wang_causal_mu_0.1_step_0.9_rstep_0.9_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --res_stepsize 0.9 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 0 --gpu-id 1 &

# GPU 4

python main_causal_adaptive_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_adaptivemomentum_wang_causal_mu_0.1_step_0.9_rstep_2.0_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --res_stepsize 2.0 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 0 --gpu-id 4 &

python main_causal_adaptive_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_adaptivemomentum_wang_causal_mu_0.1_step_0.9_rstep_0.01_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --res_stepsize 0.01 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 0 --gpu-id 2 &

python main_causal_adaptive_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_adaptivemomentum_wang_causal_mu_0.1_step_0.9_rstep_1.0_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --res_stepsize 1.0 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 0 --gpu-id 3 &

wait
echo "Done"
