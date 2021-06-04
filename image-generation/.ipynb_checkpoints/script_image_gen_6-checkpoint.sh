#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 5

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_1_rstep_0.1_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.1 --res_stepsize 0.1  --manualSeed 0 --gpu-id 5 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_3_rstep_0.1_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.3 --res_stepsize 0.1  --manualSeed 0 --gpu-id 5 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_6_rstep_0.1_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.6 --res_stepsize 0.1  --manualSeed 0 --gpu-id 5 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_9_rstep_0.1_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.9 --res_stepsize 0.1  --manualSeed 0 --gpu-id 5 &

# GPU 6

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_99_rstep_0.1_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.99 --res_stepsize 0.1  --manualSeed 0 --gpu-id 6 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_999_rstep_0.1_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.999 --res_stepsize 0.1  --manualSeed 0 --gpu-id 6 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_1_rstep_0.3_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.1 --res_stepsize 0.3  --manualSeed 0 --gpu-id 6 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_3_rstep_0.3_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.3 --res_stepsize 0.3  --manualSeed 0 --gpu-id 6 &

# GPU 7

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_6_rstep_0.3_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.6 --res_stepsize 0.3  --manualSeed 0 --gpu-id 7 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_9_rstep_0.3_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.9 --res_stepsize 0.3  --manualSeed 0 --gpu-id 7 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_99_rstep_0.3_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.99 --res_stepsize 0.3  --manualSeed 0 --gpu-id 7 &

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 96000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_mu_0.6_step_0.9_delta_0.0001_rmu_0_999_rstep_0.3_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.999 --res_stepsize 0.3  --manualSeed 0 --gpu-id 7 &

wait
echo "Done"
