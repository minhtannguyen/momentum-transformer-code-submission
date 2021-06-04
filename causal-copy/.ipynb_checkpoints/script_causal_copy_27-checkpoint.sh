#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 7

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_predict_seed_0" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_predict_seed_1" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --manualSeed 1 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_predict_seed_2" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --manualSeed 2 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_predict_seed_3" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --manualSeed 3 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_predict_seed_4" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --manualSeed 4 --gpu-id 7 &

python main.py --save_to "/tanData/momentum_transformer/linear_causal_predict_seed_0" --manualSeed 0 --gpu-id 7 &

# # GPU 6

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_0" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_1" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 1 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_2" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 2 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_3" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 3 --gpu-id 6 &

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_predict_seed_4" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 4 --gpu-id 6 &

wait
echo "Done"
