#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# # GPU 4

# python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.00001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --manualSeed 0 --gpu-id 4 &

# # GPU 5

# python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.000001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.000001 --beta 0.3 --manualSeed 0 --gpu-id 5 &

# # GPU 5

# python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.0000001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.0000001 --beta 0.3 --manualSeed 0 --gpu-id 6 &


# # GPU 7

# python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_-0.0001_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize -0.0001 --beta 0.3 --manualSeed 0 --gpu-id 7 &

# python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.01_stepf_1e_-10_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.01 --beta 0.3 --final_stepsize 0.0000000001 --manualSeed 0 --gpu-id 1 &

# python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.01_stepf_1e_-11_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.01 --beta 0.3 --final_stepsize 0.00000000001 --manualSeed 0 --gpu-id 1 &

# python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.01_stepf_1e_-12_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.01 --beta 0.3 --final_stepsize 0.000000000001 --manualSeed 0 --gpu-id 1 &


wait
echo "Done"
