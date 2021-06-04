#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 5

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_final_mu_0.6_step_0.9_delta_0.0001_rmu_0_1_rstep_0.9_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.1 --res_stepsize 0.9  --manualSeed 0 --gpu-id 5 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_final_mu_0.6_step_0.9_delta_0.0001_rmu_0_1_rstep_0.9_seed_1" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.1 --res_stepsize 0.9  --manualSeed 1 --gpu-id 6 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_final_mu_0.6_step_0.9_delta_0.0001_rmu_0_1_rstep_0.9_seed_3" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.1 --res_stepsize 0.9  --manualSeed 3 --gpu-id 7 &

# python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_final_mu_0.6_step_0.9_delta_0.0001_rmu_0_1_rstep_0.9_seed_3" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.1 --res_stepsize 0.9  --manualSeed 3 --gpu-id 7 &

# python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_final_mu_0.6_step_0.9_delta_0.0001_rmu_0_1_rstep_0.9_seed_4" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.1 --res_stepsize 0.9  --manualSeed 4 --gpu-id 0 &



wait
echo "Done"
