#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.6_seed_0" --attention_type causal-ncmomentum --stepsize 0.6 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_v2_step_0.6_seed_0" --attention_type causal-ncmomentum-v2 --stepsize 0.6 --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_v2_step_0.4_seed_0" --attention_type causal-ncmomentum-v2 --stepsize 0.4 --manualSeed 0 --gpu-id 4 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_v3_step_0.6_seed_0" --attention_type causal-ncmomentum-v3 --stepsize 0.6 --manualSeed 0 --gpu-id 5 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_v3_step_0.4_seed_0" --attention_type causal-ncmomentum-v3 --stepsize 0.4 --manualSeed 0 --gpu-id 6 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_causal_step_0.4_seed_0" --attention_type causal-ncmomentum --stepsize 0.4 --manualSeed 0 --gpu-id 7 &


# # GPU 6

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_0" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_1" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 1 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_2" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 2 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_3" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 3 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_4" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 4 --gpu-id 6 &

wait
echo "Done"
