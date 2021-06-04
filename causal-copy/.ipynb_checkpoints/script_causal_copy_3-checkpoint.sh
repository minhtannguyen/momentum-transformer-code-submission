#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.6_step_0.1_beta_0.1" --attention_type adam-linear --mu 0.6 --stepsize 0.1 --beta 0.1 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.6_step_0.01_beta_0.1" --attention_type adam-linear --mu 0.6 --stepsize 0.01 --beta 0.1 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.6_step_0.001_beta_0.1" --attention_type adam-linear --mu 0.6 --stepsize 0.001 --beta 0.1 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.6_step_0.0001_beta_0.1" --attention_type adam-linear --mu 0.6 --stepsize 0.0001 --beta 0.1 --manualSeed 0 --gpu-id 0 &


# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.1_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.1 --beta 0.3 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.01_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.01 --beta 0.3 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.001 --beta 0.3 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.0001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.0001 --beta 0.3 --manualSeed 0 --gpu-id 1 &


wait
echo "Done"
