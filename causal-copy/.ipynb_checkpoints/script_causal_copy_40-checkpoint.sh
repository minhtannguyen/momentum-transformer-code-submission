#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# FR

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.1 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.2_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.2 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.3 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.4_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.4 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.5_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.5 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.7_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.7 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.8_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_2.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.99_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.99 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.999_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.999 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.01_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.01 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 1 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_0.001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.001 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 3 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_4.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_6.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 6.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_fr_causal_mu_0.1_step_0.6_rstep_8.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 8.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 100 --manualSeed 0 --gpu-id 0 & 


# PR

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.1 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.2_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.2 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.3 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.4_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.4 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.5_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.5 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.7_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.7 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.8_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_2.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.99_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.99 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.999_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.999 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.01_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.01 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 2 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_0.001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.001 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 3 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_4.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_6.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 6.0 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_pr_causal_mu_0.1_step_0.6_rstep_8.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 8.0 --res_delta 0.0001 --adaptive_type "pr" --epochs 100 --manualSeed 0 --gpu-id 0 & 

wait
echo "Continue"

# HS

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.1 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.2_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.2 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.3 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.4_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.4 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.5_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.5 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.7_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.7 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.8_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_2.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.99_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.99 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.999_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.999 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.01_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.01 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 1 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_0.001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.001 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 3 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_4.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_6.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 6.0 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_hs_causal_mu_0.1_step_0.6_rstep_8.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 8.0 --res_delta 0.0001 --adaptive_type "hs" --epochs 100 --manualSeed 0 --gpu-id 0 & 


# DY

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.1 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.2_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.2 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.3 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.4_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.4 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.5_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.5 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.7_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.7 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.8_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_2.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.99_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.99 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.999_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.999 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.01_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.01 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 2 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_0.001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.001 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 3 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_4.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_6.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 6.0 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_dy_causal_mu_0.1_step_0.6_rstep_8.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 8.0 --res_delta 0.0001 --adaptive_type "dy" --epochs 100 --manualSeed 0 --gpu-id 0 & 

wait
echo "Continue"

# Wang

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.1 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.2_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.2 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.3 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.4_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.4 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.5_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.5 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.7_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.7 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.8_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_2.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.99_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.99 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.999_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.999 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.01_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.01 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 2 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_0.001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.001 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 3 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_4.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_6.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 6.0 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_v2_wang_causal_mu_0.1_step_0.6_rstep_8.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 8.0 --res_delta 0.0001 --adaptive_type "wang" --epochs 100 --manualSeed 0 --gpu-id 0 & 


wait
echo "Done"
