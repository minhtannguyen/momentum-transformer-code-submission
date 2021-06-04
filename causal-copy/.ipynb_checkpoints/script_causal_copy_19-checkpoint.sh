#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0


# GPU 1

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_0.01_beta_0.3_factor_0.1" --attention_type adam-linear --mu 0.9 --stepsize 0.01 --stepsize_factor 0.1 --beta 0.3 --manualSeed 0 --gpu-id 1 &


# GPU 2


python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_0.00008_beta_0.3_factor_0.5" --attention_type adam-linear --mu 0.9 --stepsize 0.00008 --stepsize_factor 0.5 --beta 0.3 --manualSeed 0 --gpu-id 2 &



# GPU 3

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_1.0_beta_0.3_factor_0.02" --attention_type adam-linear --mu 0.9 --stepsize 1.0 --stepsize_factor 0.02 --beta 0.3 --manualSeed 0 --gpu-id 3 &

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_0.00125_beta_0.3_factor_0.2" --attention_type adam-linear --mu 0.9 --stepsize 0.00125 --stepsize_factor 0.2 --beta 0.3 --manualSeed 0 --gpu-id 3 &

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_0.08_beta_0.3_factor_0.05" --attention_type adam-linear --mu 0.9 --stepsize 0.08 --stepsize_factor 0.05 --beta 0.3 --manualSeed 0 --gpu-id 3 &

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_0.0001_beta_0.3_factor_0.2" --attention_type adam-linear --mu 0.9 --stepsize 0.0001 --stepsize_factor 0.2 --beta 0.3 --manualSeed 0 --gpu-id 3 &


# GPU 4



# GPU 5


# GPU 6

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_0.001_beta_0.3_factor_0.1" --attention_type adam-linear --mu 0.9 --stepsize 0.001 --stepsize_factor 0.1 --beta 0.3 --manualSeed 0 --gpu-id 6 &

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_0.004_beta_0.3_factor_0.05" --attention_type adam-linear --mu 0.9 --stepsize 0.004 --stepsize_factor 0.05 --beta 0.3 --manualSeed 0 --gpu-id 6 &


# GPU 7

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_0.0001_beta_0.3_factor_0.1" --attention_type adam-linear --mu 0.9 --stepsize 0.0001 --stepsize_factor 0.1 --beta 0.3 --manualSeed 0 --gpu-id 7 &

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_mu_0.9_step_0.02_beta_0.3_factor_0.02" --attention_type adam-linear --mu 0.9 --stepsize 0.02 --stepsize_factor 0.02 --beta 0.3 --manualSeed 0 --gpu-id 7 &

wait
echo "Done"
