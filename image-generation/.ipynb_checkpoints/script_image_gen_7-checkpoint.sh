#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 3

python main.py --dataset cifar10 --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_linear_causal_final_seed_0" --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_ncmomentum_causal_final_step_0.7_delta_0.0001_seed_0" --stepsize 0.7 --delta 0.0001 --manualSeed 0 --gpu-id 3 &

python main.py --dataset cifar10 --attention_type full --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 1 --iterations 5000000 --evaluate_frequency 50000 --save_frequency 50000 --save_to "/tanData/momentum_transformer/image_generation/cifar_full_softfmax_final_seed_0" --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_final_mu_0.2_step_0.7_delta_0.0001_seed_0" --mu 0.2 --stepsize 0.7 --delta 0.0001 --manualSeed 0 --gpu-id 1 &


wait
echo "Done"
