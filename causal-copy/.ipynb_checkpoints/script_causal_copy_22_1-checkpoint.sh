#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.8" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.8 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.8_seed_1" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.8 --manualSeed 1 --gpu-id 0 &


# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.8_seed_3" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.8 --manualSeed 3 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.8_seed_4" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.8 --manualSeed 4 --gpu-id 1 &


# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_seed_1" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --manualSeed 1 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_seed_2" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --manualSeed 2 --gpu-id 2 &


# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_seed_4" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --manualSeed 4 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.8" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.8 --manualSeed 0 --gpu-id 3 &

# GPU 4

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.8_seed_2" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.8 --manualSeed 2 --gpu-id 4 &

# GPU 5

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.8_seed_3" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.8 --manualSeed 3 --gpu-id 5 &

# GPU 6

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.8_seed_4" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.8 --manualSeed 4 --gpu-id 6 &

# GPU 7

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.6" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.6 --manualSeed 0 --gpu-id 7 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.6_seed_1" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.6 --manualSeed 1 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.8_seed_2" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.8 --manualSeed 2 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.6_seed_2" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.6 --manualSeed 2 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.6_seed_3" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.6 --manualSeed 3 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_seed_3" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --manualSeed 3 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.6_seed_4" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.6 --manualSeed 4 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.99_step_0.3_beta_0.8_seed_1" --attention_type adamax-linear --mu 0.99 --stepsize 0.3 --beta 0.8 --manualSeed 1 --gpu-id 3 &


wait
echo "Done"
