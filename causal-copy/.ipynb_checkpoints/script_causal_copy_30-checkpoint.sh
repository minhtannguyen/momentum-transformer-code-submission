#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_final_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 3 --gpu-id 2 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_final_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 4 --gpu-id 2 &

# GPU 0

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.6_seed_0" --attention_type causal-ncmomentum --stepsize 0.6 --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.4_seed_0" --attention_type causal-ncmomentum --stepsize 0.4 --manualSeed 0 --gpu-id 2 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.1_seed_0" --attention_type causal-ncmomentum --stepsize 0.1 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.2_seed_0" --attention_type causal-ncmomentum --stepsize 0.2 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.3_seed_0" --attention_type causal-ncmomentum --stepsize 0.3 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.5_seed_0" --attention_type causal-ncmomentum --stepsize 0.5 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.7_seed_0" --attention_type causal-ncmomentum --stepsize 0.7 --manualSeed 0 --gpu-id 4 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.8_seed_0" --attention_type causal-ncmomentum --stepsize 0.8 --manualSeed 0 --gpu-id 4 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.9_seed_0" --attention_type causal-ncmomentum --stepsize 0.9 --manualSeed 0 --gpu-id 5 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_1.0_seed_0" --attention_type causal-ncmomentum --stepsize 1.0 --manualSeed 0 --gpu-id 5 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_2.0_seed_0" --attention_type causal-ncmomentum --stepsize 2.0 --manualSeed 0 --gpu-id 6 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_4.0_seed_0" --attention_type causal-ncmomentum --stepsize 4.0 --manualSeed 0 --gpu-id 6 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.01_seed_0" --attention_type causal-ncmomentum --stepsize 0.01 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.001_seed_0" --attention_type causal-ncmomentum --stepsize 0.001 --manualSeed 0 --gpu-id 7 &


# # GPU 6

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_0" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_1" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 1 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_2" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 2 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_3" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 3 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_4" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 4 --gpu-id 6 &

wait
echo "Done"
