#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 5

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_mu_0.6_step_0.9_rstep_0.1_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 0.1 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 0 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_mu_0.6_step_0.9_rstep_0.3_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 0.3 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 0 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_mu_0.6_step_0.9_rstep_0.5_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 0.5 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 0 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_mu_0.6_step_0.9_rstep_0.6_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 0.6 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 0 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_mu_0.6_step_0.9_rstep_0.8_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 0.8 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 0 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_mu_0.6_step_0.9_rstep_0.9_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 0.9 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 0 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_mu_0.6_step_0.9_rstep_1.0_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 1.0 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 0 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_mu_0.6_step_0.9_rstep_2.0_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 2.0 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 0 &

python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/adaptivemomentum_fr_causal_mu_0.6_step_0.9_rstep_0.99_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --res_stepsize 0.99 --delta 0.0001 --adaptive_type "fr" --is_resw False  --manualSeed 0 --gpu-id 0 &

wait
echo "Done"
