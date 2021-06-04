#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 4

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.00001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --manualSeed 0 --gpu-id 4 &

# GPU 5

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.000001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.000001 --beta 0.3 --manualSeed 0 --gpu-id 5 &

# GPU 5

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.0000001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.0000001 --beta 0.3 --manualSeed 0 --gpu-id 6 &


# GPU 7

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_-0.0001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize -0.0001 --beta 0.3 --manualSeed 0 --gpu-id 7 &


wait
echo "Done"
