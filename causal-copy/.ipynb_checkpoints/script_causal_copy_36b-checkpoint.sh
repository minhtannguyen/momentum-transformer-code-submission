#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_5.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 5.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_6.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 6.0 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_7.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 7.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_8.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 8.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_9.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 9.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_10.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 10.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_11.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 11.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_12.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 12.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_13.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 13.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_14.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 14.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_15.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 15.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_20.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 20.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_30.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 30.0 --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_50.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 50.0 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_100.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 100.0 --epochs 600 --manualSeed 0 --gpu-id 3 &

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_pr_causal_mu_0.1_step_0.6_rstep_1000.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1000.0 --epochs 600 --manualSeed 0 --gpu-id 0 & 

wait
echo "Done"
