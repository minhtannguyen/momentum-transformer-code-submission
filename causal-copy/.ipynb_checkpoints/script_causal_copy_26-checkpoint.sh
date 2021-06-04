#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.999" --attention_type fradamax-linear --stepsize 0.3 --beta 0.999 --epochs 600 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.999_seed_1" --attention_type fradamax-linear --stepsize 0.3 --beta 0.999 --epochs 600 --manualSeed 1 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.999_seed_2" --attention_type fradamax-linear --stepsize 0.3 --beta 0.999 --epochs 600 --manualSeed 2 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.999_seed_3" --attention_type fradamax-linear --stepsize 0.3 --beta 0.999 --epochs 600 --manualSeed 3 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.999_seed_4" --attention_type fradamax-linear --stepsize 0.3 --beta 0.999 --epochs 600 --manualSeed 4 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.6" --attention_type fradamax-linear --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.6_seed_1" --attention_type fradamax-linear --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 1 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.6_seed_2" --attention_type fradamax-linear --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 2 --gpu-id 3 &

# GPU 4

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.6_seed_3" --attention_type fradamax-linear --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 3 --gpu-id 4 &

# GPU 5

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.6_seed_4" --attention_type fradamax-linear --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 4 --gpu-id 5 &

wait
echo "Continue"
