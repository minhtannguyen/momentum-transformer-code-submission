#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.999_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.999 --res_stepsize 0.1 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.1_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.1 --res_stepsize 0.1 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.1_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.1 --res_stepsize 0.3 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.1_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.1 --res_stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.1_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.1 --res_stepsize 0.9 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.1_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.1 --res_stepsize 1.0 --epochs 600 --manualSeed 0 --gpu-id 1 &




python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.999_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.999 --res_stepsize 0.3 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.3_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.3 --res_stepsize 0.1 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.3_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.3 --res_stepsize 0.3 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.3_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.3 --res_stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.3_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.3 --res_stepsize 0.9 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.3_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.3 --res_stepsize 1.0 --epochs 600 --manualSeed 0 --gpu-id 1 &



python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.999_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.999 --res_stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.6_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.6 --res_stepsize 0.1 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.6_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.6 --res_stepsize 0.3 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.6_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.6 --res_stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.6_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.6 --res_stepsize 0.9 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.6_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.6 --res_stepsize 1.0 --epochs 600 --manualSeed 0 --gpu-id 2 &



python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.999_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.999 --res_stepsize 0.9 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.9_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.9 --res_stepsize 0.1 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.9_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.9 --res_stepsize 0.3 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.9_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.9 --res_stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.9_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.9 --res_stepsize 0.9 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.9_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.9 --res_stepsize 1.0 --epochs 600 --manualSeed 0 --gpu-id 2 &



python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.999_rstep_0.99_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.999 --res_stepsize 0.99 --epochs 600 --manualSeed 0 --gpu-id 3 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.99_rstep_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.1 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.99_rstep_0.3_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.3 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.99_rstep_0.6_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.99_rstep_0.9_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.9 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_v2_causal_mu_0.1_step_0.6_rmu_0.99_rstep_1.0_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 1.0 --epochs 600 --manualSeed 0 --gpu-id 2 &

wait
echo "Done"
