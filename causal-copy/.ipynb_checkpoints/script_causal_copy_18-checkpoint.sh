#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_2.0" --attention_type nesterov-linear --stepsize 2.0 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_1.5" --attention_type nesterov-linear --stepsize 1.5 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_1.0" --attention_type nesterov-linear --stepsize 1.0 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.9" --attention_type nesterov-linear --stepsize 0.9 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.8" --attention_type nesterov-linear --stepsize 0.8 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.7" --attention_type nesterov-linear --stepsize 0.7 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.6" --attention_type nesterov-linear --stepsize 0.6 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.5" --attention_type nesterov-linear --stepsize 0.5 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.4" --attention_type nesterov-linear --stepsize 0.4 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.3" --attention_type nesterov-linear --stepsize 0.3 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.2" --attention_type nesterov-linear --stepsize 0.2 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.1" --attention_type nesterov-linear --stepsize 0.1 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.01" --attention_type nesterov-linear --stepsize 0.01 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.001" --attention_type nesterov-linear --stepsize 0.001 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.0001" --attention_type nesterov-linear --stepsize 0.0001 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_step_0.00001" --attention_type nesterov-linear --stepsize 0.00001 --manualSeed 0 --gpu-id 1 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.6_factor_1.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --stepsize_factor 1.5 --manualSeed 0 --gpu-id 1 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.3_factor_1.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.3 --stepsize_factor 1.5 --manualSeed 0 --gpu-id 1 &



# GPU 2

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_2.0_factor_0.5" --attention_type momentum-linear --mu 0.1 --stepsize 2.0 --stepsize_factor 0.5 --manualSeed 0 --gpu-id 2 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.5_factor_0.5" --attention_type momentum-linear --mu 0.1 --stepsize 1.5 --stepsize_factor 0.5 --manualSeed 0 --gpu-id 2 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.0_factor_0.5" --attention_type momentum-linear --mu 0.1 --stepsize 1.0 --stepsize_factor 0.5 --manualSeed 0 --gpu-id 2 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.9_factor_0.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.9 --stepsize_factor 0.5 --manualSeed 0 --gpu-id 2 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.6_factor_0.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --stepsize_factor 0.5 --manualSeed 0 --gpu-id 2 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.3_factor_0.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.3 --stepsize_factor 0.5 --manualSeed 0 --gpu-id 2 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.1_factor_0.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.1 --stepsize_factor 0.5 --manualSeed 0 --gpu-id 2 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.1_factor_1.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.1 --stepsize_factor 1.5 --manualSeed 0 --gpu-id 2 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.05_factor_1.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.05 --stepsize_factor 1.5 --manualSeed 0 --gpu-id 2 &


# GPU 3

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_2.0_factor_2.0" --attention_type momentum-linear --mu 0.1 --stepsize 2.0 --stepsize_factor 2.0 --manualSeed 0 --gpu-id 3 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.5_factor_2.0" --attention_type momentum-linear --mu 0.1 --stepsize 1.5 --stepsize_factor 2.0 --manualSeed 0 --gpu-id 3 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.0_factor_2.0" --attention_type momentum-linear --mu 0.1 --stepsize 1.0 --stepsize_factor 2.0 --manualSeed 0 --gpu-id 3 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.9_factor_2.0" --attention_type momentum-linear --mu 0.1 --stepsize 0.9 --stepsize_factor 2.0 --manualSeed 0 --gpu-id 3 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.6_factor_2.0" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --stepsize_factor 2.0 --manualSeed 0 --gpu-id 3 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.3_factor_2.0" --attention_type momentum-linear --mu 0.1 --stepsize 0.3 --stepsize_factor 2.0 --manualSeed 0 --gpu-id 3 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.1_factor_2.0" --attention_type momentum-linear --mu 0.1 --stepsize 0.1 --stepsize_factor 2.0 --manualSeed 0 --gpu-id 3 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.05_factor_2.0" --attention_type momentum-linear --mu 0.1 --stepsize 0.05 --stepsize_factor 2.0 --manualSeed 0 --gpu-id 3 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.025_factor_2.0" --attention_type momentum-linear --mu 0.1 --stepsize 0.025 --stepsize_factor 2.0 --manualSeed 0 --gpu-id 3 &

# GPU 4

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_2.0_factor_0.25" --attention_type momentum-linear --mu 0.1 --stepsize 2.0 --stepsize_factor 0.25 --manualSeed 0 --gpu-id 4 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.5_factor_0.25" --attention_type momentum-linear --mu 0.1 --stepsize 1.5 --stepsize_factor 0.25 --manualSeed 0 --gpu-id 4 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.0_factor_0.25" --attention_type momentum-linear --mu 0.1 --stepsize 1.0 --stepsize_factor 0.25 --manualSeed 0 --gpu-id 4 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.9_factor_0.25" --attention_type momentum-linear --mu 0.1 --stepsize 0.9 --stepsize_factor 0.25 --manualSeed 0 --gpu-id 4 &


# GPU 5

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.6_factor_0.25" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --stepsize_factor 0.25 --manualSeed 0 --gpu-id 5 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.3_factor_0.25" --attention_type momentum-linear --mu 0.1 --stepsize 0.3 --stepsize_factor 0.25 --manualSeed 0 --gpu-id 5 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.1_factor_0.25" --attention_type momentum-linear --mu 0.1 --stepsize 0.1 --stepsize_factor 0.25 --manualSeed 0 --gpu-id 5 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.025_factor_1.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.025 --stepsize_factor 1.5 --manualSeed 0 --gpu-id 5 &

# GPU 6

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_2.0_factor_1.5" --attention_type momentum-linear --mu 0.1 --stepsize 2.0 --stepsize_factor 1.5 --manualSeed 0 --gpu-id 6 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.5_factor_1.5" --attention_type momentum-linear --mu 0.1 --stepsize 1.5 --stepsize_factor 1.5 --manualSeed 0 --gpu-id 6 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.0_factor_1.5" --attention_type momentum-linear --mu 0.1 --stepsize 1.0 --stepsize_factor 1.5 --manualSeed 0 --gpu-id 6 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_0.9_factor_1.5" --attention_type momentum-linear --mu 0.1 --stepsize 0.9 --stepsize_factor 1.5 --manualSeed 0 --gpu-id 6 &

# GPU 7


wait
echo "Done"
