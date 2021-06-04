#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 2

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_mu_0.1_step_0.6_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.6 --delta 0.0001 --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_mu_0.3_step_0.6_delta_0.0001_seed_0" --mu 0.3 --stepsize 0.6 --delta 0.0001 --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_mu_0.6_step_0.6_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.6 --delta 0.0001 --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_mu_0.9_step_0.6_delta_0.0001_seed_0" --mu 0.9 --stepsize 0.6 --delta 0.0001 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_mu_0.1_step_0.9_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_mu_0.3_step_0.9_delta_0.0001_seed_0" --mu 0.3 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_mu_0.6_step_0.9_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_mu_0.9_step_0.9_delta_0.0001_seed_0" --mu 0.9 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 3 &


wait
echo "Done"
