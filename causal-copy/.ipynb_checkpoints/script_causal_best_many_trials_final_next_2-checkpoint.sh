#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_1.0_final_seed_0" --attention_type causal-ncmomentum --stepsize 1.0 --epochs 600 --manualSeed 0 --gpu-id 0 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_1.0_final_seed_1" --attention_type causal-ncmomentum --stepsize 1.0 --epochs 600 --manualSeed 1 --gpu-id 0 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_1.0_final_seed_2" --attention_type causal-ncmomentum --stepsize 1.0 --epochs 600 --manualSeed 2 --gpu-id 0 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_1.0_final_seed_3" --attention_type causal-ncmomentum --stepsize 1.0 --epochs 600 --manualSeed 3 --gpu-id 0 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_1.0_final_seed_4" --attention_type causal-ncmomentum --stepsize 1.0 --epochs 600 --manualSeed 4 --gpu-id 0 &

# GPU 2

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_2.0_final_seed_0" --attention_type causal-ncmomentum --stepsize 2.0 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_2.0_final_seed_1" --attention_type causal-ncmomentum --stepsize 2.0 --epochs 600 --manualSeed 1 --gpu-id 2 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_2.0_final_seed_2" --attention_type causal-ncmomentum --stepsize 2.0 --epochs 600 --manualSeed 2 --gpu-id 2 &

# GPU 3

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_2.0_final_seed_3" --attention_type causal-ncmomentum --stepsize 2.0 --epochs 600 --manualSeed 3 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_2.0_final_seed_4" --attention_type causal-ncmomentum --stepsize 2.0 --epochs 600 --manualSeed 4 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_4.0_final_seed_0" --attention_type causal-ncmomentum --stepsize 4.0 --epochs 600 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_4.0_final_seed_1" --attention_type causal-ncmomentum --stepsize 4.0 --epochs 600 --manualSeed 1 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_4.0_final_seed_2" --attention_type causal-ncmomentum --stepsize 4.0 --epochs 600 --manualSeed 2 --gpu-id 3 &

# GPU 4

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_4.0_final_seed_3" --attention_type causal-ncmomentum --stepsize 4.0 --epochs 600 --manualSeed 3 --gpu-id 4 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_4.0_final_seed_4" --attention_type causal-ncmomentum --stepsize 4.0 --epochs 600 --manualSeed 4 --gpu-id 4 &

# GPU 5

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.9_final_seed_0" --attention_type causal-ncmomentum --stepsize 0.9 --epochs 600 --manualSeed 0 --gpu-id 5 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.9_final_seed_1" --attention_type causal-ncmomentum --stepsize 0.9 --epochs 600 --manualSeed 1 --gpu-id 5 &

# GPU 6

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.9_final_seed_2" --attention_type causal-ncmomentum --stepsize 0.9 --epochs 600 --manualSeed 2 --gpu-id 6 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.9_final_seed_3" --attention_type causal-ncmomentum --stepsize 0.9 --epochs 600 --manualSeed 3 --gpu-id 6 &

# GPU 7

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.9_final_seed_4" --attention_type causal-ncmomentum --stepsize 0.9 --epochs 600 --manualSeed 4 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_8.0_final_seed_0" --attention_type causal-ncmomentum --stepsize 8.0 --epochs 600 --manualSeed 0 --gpu-id 7 &

# GPU 1

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_8.0_final_seed_1" --attention_type causal-ncmomentum --stepsize 8.0 --epochs 600 --manualSeed 1 --gpu-id 1 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_8.0_final_seed_2" --attention_type causal-ncmomentum --stepsize 8.0 --epochs 600 --manualSeed 2 --gpu-id 1 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_final_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_final_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 1 --gpu-id 1 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_final_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 2 --gpu-id 1 &

# GPU 2

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_final_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 3 --gpu-id 2 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_final_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 4 --gpu-id 2 &


wait
echo "Done"
