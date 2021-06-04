#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# mu_0.1_step_0.6_rstep_2.0_delta_0.0001

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentumW_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --is_resw True --epochs 600 --manualSeed 0 --gpu-id 4 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentumW_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --is_resw True --epochs 600 --manualSeed 1 --gpu-id 4 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentumW_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --is_resw True --epochs 600 --manualSeed 2 --gpu-id 4 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentumW_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --is_resw True --epochs 600 --manualSeed 3 --gpu-id 4 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentumW_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --is_resw True --epochs 600 --manualSeed 4 --gpu-id 4 &


wait
echo "Done"
