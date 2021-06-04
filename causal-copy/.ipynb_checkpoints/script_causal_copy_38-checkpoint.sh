#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.1 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.2_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.2 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.3 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.4_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.4 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.5_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.5 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.7_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.7 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.8_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_2.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.99_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.99 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.999_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.999 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.01_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.01 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_0.001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.001 --epochs 600 --manualSeed 0 --gpu-id 3 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_4.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --epochs 600 --manualSeed 0 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_6.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 6.0 --epochs 600 --manualSeed 0 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_dy_causal_mu_0.1_step_0.6_rstep_8.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 8.0 --epochs 600 --manualSeed 0 --gpu-id 0 & 

wait
echo "Done"
