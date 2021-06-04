#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_1.0_delta_0.0001_seed_0" --stepsize 1.0 --delta 0.0001 --manualSeed 0 --gpu-id 0 &

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_2.0_delta_0.0001_seed_0" --stepsize 2.0 --delta 0.0001 --manualSeed 0 --gpu-id 0 &

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.7_delta_0.0001_seed_0" --stepsize 0.7 --delta 0.0001 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_4.0_delta_0.0001_seed_0" --stepsize 4.0 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.9_delta_0.0001_seed_0" --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.4_delta_0.0001_seed_0" --stepsize 0.4 --delta 0.0001 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.6_delta_0.0001_seed_0" --stepsize 0.6 --delta 0.0001 --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.3_delta_0.0001_seed_0" --stepsize 0.3 --delta 0.0001 --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.2_delta_0.0001_seed_0" --stepsize 0.2 --delta 0.0001 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.1_delta_0.0001_seed_0" --stepsize 0.1 --delta 0.0001 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.01_delta_0.0001_seed_0" --stepsize 0.01 --delta 0.0001 --manualSeed 0 --gpu-id 3 &

# GPU 4

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.99_delta_0.0001_seed_0" --stepsize 0.99 --delta 0.0001 --manualSeed 0 --gpu-id 4 &

# GPU 5

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.8_delta_0.0001_seed_0" --stepsize 0.8 --delta 0.0001 --manualSeed 0 --gpu-id 5 &

# GPU 6

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_0.5_delta_0.0001_seed_0" --stepsize 0.5 --delta 0.0001 --manualSeed 0 --gpu-id 6 &

# GPU 7

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_10.0_delta_0.0001_seed_0" --stepsize 10.0 --delta 0.0001 --manualSeed 0 --gpu-id 7 &

wait
echo "Done"
