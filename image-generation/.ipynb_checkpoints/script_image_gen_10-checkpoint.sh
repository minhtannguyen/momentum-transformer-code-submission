#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 5

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_final_mu_0.6_step_0.9_rstep_0.6_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 0.6 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 5 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_final_mu_0.6_step_0.9_rstep_0.6_delta_0.0001_seed_1" --mu 0.6 --stepsize 0.9 --res_stepsize 0.6 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 1 --gpu-id 6 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_final_mu_0.6_step_0.9_rstep_0.6_delta_0.0001_seed_3" --mu 0.6 --stepsize 0.9 --res_stepsize 0.6 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 3 --gpu-id 7 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_wang_causal_final_mu_0.6_step_0.9_rstep_0.99_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 0.99 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 0 --gpu-id 5 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_wang_causal_final_mu_0.6_step_0.9_rstep_0.99_delta_0.0001_seed_1" --mu 0.6 --stepsize 0.9 --res_stepsize 0.99 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 1 --gpu-id 6 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_wang_causal_final_mu_0.6_step_0.9_rstep_0.99_delta_0.0001_seed_3" --mu 0.6 --stepsize 0.9 --res_stepsize 0.99 --delta 0.0001 --adaptive_type "wang" --is_resw False  --manualSeed 3 --gpu-id 7 &

wait
echo "Done"
