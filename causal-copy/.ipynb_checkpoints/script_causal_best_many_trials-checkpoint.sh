#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main.py --save_to "/tanData/momentum_transformer/full_softfmax_seed_1" --attention_type full --manualSeed 1 --gpu-id 0 &

python main.py --save_to "/tanData/momentum_transformer/full_softfmax_seed_2" --attention_type full --manualSeed 2 --gpu-id 0 &

python main.py --save_to "/tanData/momentum_transformer/reformer_seed_1" --attention_type reformer --manualSeed 1 --gpu-id 0 &

python main.py --save_to "/tanData/momentum_transformer/reformer_seed_2" --attention_type reformer --manualSeed 2 --gpu-id 0 &

# GPU 1

python main.py --save_to "/tanData/momentum_transformer/linear_causal_seed_1" --manualSeed 1 --gpu-id 1 &

python main.py --save_to "/tanData/momentum_transformer/linear_causal_seed_2" --manualSeed 2 --gpu-id 1 &

python main_recurrent.py --save_to "/tanData/momentum_transformer/rnn_linear_causal_seed_1" --attention_type causal-linear --manualSeed 1 --gpu-id 1 &

python main_recurrent.py --save_to "/tanData/momentum_transformer/rnn_linear_causal_seed_2" --attention_type causal-linear --manualSeed 2 --gpu-id 1 &

# GPU 2

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_seed_1" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 1 --gpu-id 2  &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_seed_2" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 2 --gpu-id 2  &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.00001_beta_0.3_seed_1" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --manualSeed 1 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.00001_beta_0.3_seed_2" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --manualSeed 2 --gpu-id 2 &

# GPU 3

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.5_factor_0.5_seed_1" --attention_type momentum-linear --mu 0.1 --stepsize 1.5 --stepsize_factor 0.5 --manualSeed 1 --gpu-id 3 &

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_mu_0.1_step_1.5_factor_0.5_seed_2" --attention_type momentum-linear --mu 0.1 --stepsize 1.5 --stepsize_factor 0.5 --manualSeed 2 --gpu-id 3 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_mu_0.9_step_1.5_seed_1" --attention_type normgrad-linear --mu 0.9 --stepsize 1.5 --manualSeed 1 --gpu-id 3 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_mu_0.9_step_1.5_seed_2" --attention_type normgrad-linear --mu 0.9 --stepsize 1.5 --manualSeed 2 --gpu-id 3 &

# GPU 4

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.1_seed_1" --attention_type frmomentum-linear --stepsize 0.1 --manualSeed 1 --gpu-id 4 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.1_seed_2" --attention_type frmomentum-linear --stepsize 0.1 --manualSeed 2 --gpu-id 4 &


wait
echo "Done"
