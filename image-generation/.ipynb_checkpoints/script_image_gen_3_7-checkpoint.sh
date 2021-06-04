#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 4

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.1_step_0.3_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.3 --delta 0.0001 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.1_step_0.3_delta_0.0001_seed_1" --mu 0.1 --stepsize 0.3 --delta 0.0001 --manualSeed 1 --gpu-id 0 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.1_step_0.3_delta_0.0001_seed_3" --mu 0.1 --stepsize 0.3 --delta 0.0001 --manualSeed 3 --gpu-id 1 &

wait
echo "Done"
