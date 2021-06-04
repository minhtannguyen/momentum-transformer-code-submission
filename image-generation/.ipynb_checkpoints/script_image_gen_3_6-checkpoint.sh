#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 4

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.6_step_0.6_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.6 --delta 0.0001 --manualSeed 0 --gpu-id 4 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.6_step_0.6_delta_0.0001_seed_1" --mu 0.6 --stepsize 0.6 --delta 0.0001 --manualSeed 1 --gpu-id 4 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.6_step_0.6_delta_0.0001_seed_3" --mu 0.6 --stepsize 0.6 --delta 0.0001 --manualSeed 3 --gpu-id 4 &

# GPU 5

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.3_step_0.9_delta_0.0001_seed_0" --mu 0.3 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 5 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.3_step_0.9_delta_0.0001_seed_1" --mu 0.3 --stepsize 0.9 --delta 0.0001 --manualSeed 1 --gpu-id 5 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.3_step_0.9_delta_0.0001_seed_3" --mu 0.3 --stepsize 0.9 --delta 0.0001 --manualSeed 3 --gpu-id 5 &

# GPU 6

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.6_step_0.9_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 6 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.6_step_0.9_delta_0.0001_seed_1" --mu 0.6 --stepsize 0.9 --delta 0.0001 --manualSeed 1 --gpu-id 6 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.6_step_0.9_delta_0.0001_seed_3" --mu 0.6 --stepsize 0.9 --delta 0.0001 --manualSeed 3 --gpu-id 6 &

# GPU 7

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.1_step_0.9_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.1_step_0.9_delta_0.0001_seed_1" --mu 0.1 --stepsize 0.9 --delta 0.0001 --manualSeed 1 --gpu-id 7 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.1_step_0.9_delta_0.0001_seed_3" --mu 0.1 --stepsize 0.9 --delta 0.0001 --manualSeed 3 --gpu-id 7 &

# Others

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.1_step_0.6_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.6 --delta 0.0001 --manualSeed 0 --gpu-id 4 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.1_step_0.6_delta_0.0001_seed_1" --mu 0.1 --stepsize 0.6 --delta 0.0001 --manualSeed 1 --gpu-id 5 &

python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.1_step_0.6_delta_0.0001_seed_3" --mu 0.1 --stepsize 0.6 --delta 0.0001 --manualSeed 3 --gpu-id 6 &

wait
echo "Done"
