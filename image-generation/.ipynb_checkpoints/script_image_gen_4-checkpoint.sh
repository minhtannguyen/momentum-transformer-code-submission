#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 4

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_final_step_0.7_delta_0.0001_seed_0" --stepsize 0.7 --delta 0.0001 --manualSeed 0 --gpu-id 4 &

# GPU 5

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_final_step_0.7_delta_0.0001_seed_1" --stepsize 0.7 --delta 0.0001 --manualSeed 1 --gpu-id 5 &

# GPU 6

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_final_step_0.7_delta_0.0001_seed_2" --stepsize 0.7 --delta 0.0001 --manualSeed 2 --gpu-id 6 &

# GPU 0

python main.py --dataset mnist --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/linear_causal_seed_1" --manualSeed 1 --gpu-id 0 &

python main.py --dataset mnist --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/linear_causal_seed_2" --manualSeed 2 --gpu-id 0 &

python main.py --dataset mnist --attention_type full --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/full_softfmax_seed_1" --manualSeed 1 --gpu-id 0 &

python main.py --dataset mnist --attention_type full --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/full_softfmax_seed_2" --manualSeed 2 --gpu-id 0 &

# GPU 7

python main.py --dataset mnist --attention_type reformer --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --chunk_size 29 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/reformer_seed_1" --manualSeed 1 --gpu-id 7 &

python main.py --dataset mnist --attention_type reformer --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --chunk_size 29 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/reformer_seed_2" --manualSeed 2 --gpu-id 7 &

wait
echo "Done"
