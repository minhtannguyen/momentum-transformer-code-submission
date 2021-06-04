#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# mu_0.1_step_0.6_rstep_4.0_delta_0.001

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.001_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.001_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.001_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.001_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 0 & 

# mu_0.1_step_0.6_rstep_4.0_delta_0.01

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.01_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.01_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.01_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.01_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.01_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 0 & 

# mu_0.1_step_0.6_rstep_4.0_delta_0.1

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.1_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.1 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.1_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.1 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.1_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.1 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.1_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.1 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 0 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_4.0_delta_0.1_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 4.0 --res_delta 0.1 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 1 & 

# mu_0.1_step_0.6_rstep_0.6_delta_0.01

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.01_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.01_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.01_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.01_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.01_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.01 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 1 & 

# mu_0.1_step_0.6_rstep_0.6_delta_0.001

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.001_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.001_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.001_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.001_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.001 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 1 & 

# mu_0.1_step_0.6_rstep_0.8_delta_0.0001

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.8_delta_0.0001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.8_delta_0.0001_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.8_delta_0.0001_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 1 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.8_delta_0.0001_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.8_delta_0.0001_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 2 & 

# mu_0.1_step_0.6_rstep_0.9_delta_0.0001

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.9_delta_0.0001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.9_delta_0.0001_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.9_delta_0.0001_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.9_delta_0.0001_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.9_delta_0.0001_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.9 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 2 & 

# mu_0.1_step_0.6_rstep_1.0_delta_0.0001

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_1.0_delta_0.0001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_1.0_delta_0.0001_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_1.0_delta_0.0001_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_1.0_delta_0.0001_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_1.0_delta_0.0001_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 1.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 2 & 

# mu_0.1_step_0.6_rstep_2.0_delta_0.0001

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 2 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_2.0_delta_0.0001_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 3 &

# mu_0.1_step_0.6_rstep_0.6_delta_0.0001

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.0001_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.0001_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 1 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.0001_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 2 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.0001_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 3 --gpu-id 3 & 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_fr_causal_mu_0.1_step_0.6_rstep_0.6_delta_0.0001_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.6 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 4 --gpu-id 3 & 

wait
echo "Done"
