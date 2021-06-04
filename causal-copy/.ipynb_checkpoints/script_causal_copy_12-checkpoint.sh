#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.0_step_1.0" --attention_type normgrad-linear --mu 0.0 --stepsize 1.0 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.0_step_0.9" --attention_type normgrad-linear --mu 0.0 --stepsize 0.9 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.0_step_0.6" --attention_type normgrad-linear --mu 0.0 --stepsize 0.6 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.0_step_0.3" --attention_type normgrad-linear --mu 0.0 --stepsize 0.3 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.0_step_0.1" --attention_type normgrad-linear --mu 0.0 --stepsize 0.1 --epochs 10 --manualSeed 0 --gpu-id 0 &


# GPU 1

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.1_step_1.0" --attention_type normgrad-linear --mu 0.1 --stepsize 1.0 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.1_step_0.9" --attention_type normgrad-linear --mu 0.1 --stepsize 0.9 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.1_step_0.6" --attention_type normgrad-linear --mu 0.1 --stepsize 0.6 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.1_step_0.3" --attention_type normgrad-linear --mu 0.1 --stepsize 0.3 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.1_step_0.1" --attention_type normgrad-linear --mu 0.1 --stepsize 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &


# GPU 2

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.3_step_1.0" --attention_type normgrad-linear --mu 0.3 --stepsize 1.0 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.3_step_0.9" --attention_type normgrad-linear --mu 0.3 --stepsize 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.3_step_0.6" --attention_type normgrad-linear --mu 0.3 --stepsize 0.6 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.3_step_0.3" --attention_type normgrad-linear --mu 0.3 --stepsize 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.3_step_0.1" --attention_type normgrad-linear --mu 0.3 --stepsize 0.1 --epochs 10 --manualSeed 0 --gpu-id 2 &


# GPU 3

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.6_step_1.0" --attention_type normgrad-linear --mu 0.6 --stepsize 1.0 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.6_step_0.9" --attention_type normgrad-linear --mu 0.6 --stepsize 0.9 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.6_step_0.6" --attention_type normgrad-linear --mu 0.6 --stepsize 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.6_step_0.3" --attention_type normgrad-linear --mu 0.6 --stepsize 0.3 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.6_step_0.1" --attention_type normgrad-linear --mu 0.6 --stepsize 0.1 --epochs 10 --manualSeed 0 --gpu-id 3 &

# GPU 4

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.9_step_0.1" --attention_type normgrad-linear --mu 0.9 --stepsize 0.1 --epochs 10 --manualSeed 0 --gpu-id 4 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.9_step_0.9" --attention_type normgrad-linear --mu 0.9 --stepsize 0.9 --epochs 10 --manualSeed 0 --gpu-id 4 &

# GPU 5

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.9_step_0.6" --attention_type normgrad-linear --mu 0.9 --stepsize 0.6 --epochs 10 --manualSeed 0 --gpu-id 5 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.9_step_1.0" --attention_type normgrad-linear --mu 0.9 --stepsize 1.0 --epochs 10 --manualSeed 0 --gpu-id 5 &

# GPU 6

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_v5_mu_0.9_step_0.3" --attention_type normgrad-linear --mu 0.9 --stepsize 0.3 --epochs 10 --manualSeed 0 --gpu-id 6 &

wait
echo "Done"
