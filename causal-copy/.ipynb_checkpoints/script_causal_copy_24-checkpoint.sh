#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_long" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 200 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_long_seed_1" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 200 --manualSeed 1 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_long_seed_2" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 200 --manualSeed 2 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_long_seed_3" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 200 --manualSeed 3 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_long_seed_4" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 200 --manualSeed 4 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_long" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 200 --manualSeed 0 --gpu-id 1  &

# GPU 2

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_long_seed_1" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 200 --manualSeed 1 --gpu-id 2  &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_long_seed_2" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 200 --manualSeed 2 --gpu-id 2  &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_long_seed_3" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 200 --manualSeed 3 --gpu-id 2  &

# GPU 3

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_long_seed_4" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 200 --manualSeed 4 --gpu-id 3  &


wait
echo "Done"
