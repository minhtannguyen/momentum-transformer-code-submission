#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentumW_v3_causal_mu_0.1_step_0.6_rmu_0.99_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 0 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentumW_v3_causal_mu_0.1_step_0.6_rmu_0.99_rstep_0.6_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.6 --epochs 600 --manualSeed 1 --gpu-id 0 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentumW_v3_causal_mu_0.1_step_0.6_rmu_0.99_rstep_0.6_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.6 --epochs 600 --manualSeed 2 --gpu-id 0 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentumW_v3_causal_mu_0.1_step_0.6_rmu_0.99_rstep_0.6_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.6 --epochs 600 --manualSeed 3 --gpu-id 0 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentumW_v3_causal_mu_0.1_step_0.6_rmu_0.99_rstep_0.6_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.6 --epochs 600 --manualSeed 4 --gpu-id 0 &

wait
echo "Done"
