#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 1

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.1_step_0.1_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.2_step_0.1_delta_0.0001_seed_0" --mu 0.2 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.3_step_0.1_delta_0.0001_seed_0" --mu 0.3 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.4_step_0.1_delta_0.0001_seed_0" --mu 0.4 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.5_step_0.1_delta_0.0001_seed_0" --mu 0.5 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.6_step_0.1_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.1_step_0.1_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.8_step_0.1_delta_0.0001_seed_0" --mu 0.8 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.9_step_0.1_delta_0.0001_seed_0" --mu 0.9 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

# python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.99_step_0.1_delta_0.0001_seed_0" --mu 0.99 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 0 &

# python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.999_step_0.1_delta_0.0001_seed_0" --mu 0.999 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 2 &

# python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_mu_0.01_step_0.1_delta_0.0001_seed_0" --mu 0.01 --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 2 &

wait
echo "Done"
