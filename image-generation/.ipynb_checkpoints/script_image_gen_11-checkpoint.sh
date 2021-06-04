#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Momentum

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_final_mu_0.1_step_0.9_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_final_mu_0.1_step_0.9_delta_0.0001_seed_1" --mu 0.1 --stepsize 0.9 --delta 0.0001 --manualSeed 1 --gpu-id 1 &

# python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_final_mu_0.1_step_0.9_delta_0.0001_seed_2" --mu 0.1 --stepsize 0.9 --delta 0.0001 --manualSeed 2 --gpu-id 1 &

# Softmax

python main.py --dataset cifar10 --attention_type full --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 1 --iterations 5000000 --evaluate_frequency 50000 --save_frequency 50000 --save_to "/tanData/momentum_transformer/image_generation/cifar_full_softfmax_final_seed_0" --manualSeed 0 --gpu-id 2 &

python main.py --dataset cifar10 --attention_type full --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 1 --iterations 5000000 --evaluate_frequency 50000 --save_frequency 50000 --save_to "/tanData/momentum_transformer/image_generation/cifar_full_softfmax_final_seed_1" --manualSeed 1 --gpu-id 3 &

# python main.py --dataset cifar10 --attention_type full --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 1 --iterations 5000000 --evaluate_frequency 50000 --save_frequency 50000 --save_to "/tanData/momentum_transformer/image_generation/cifar_full_softfmax_final_seed_2" --manualSeed 2 --gpu-id 2 &

# Linear

python main.py --dataset cifar10 --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_linear_causal_final_seed_0" --manualSeed 0 --gpu-id 0 &

python main.py --dataset cifar10 --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_linear_causal_final_seed_1" --manualSeed 1 --gpu-id 0 &

# python main.py --dataset cifar10 --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_linear_causal_final_seed_2" --manualSeed 2 --gpu-id 3 &






wait
echo "Done"
