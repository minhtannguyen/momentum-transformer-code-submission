#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.1" --attention_type frmomentum-linear --stepsize 0.1 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.1" --attention_type frmomentum-linear-v2 --stepsize 0.1 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v3_step_0.1" --attention_type frmomentum-linear-v3 --stepsize 0.1 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v3_step_0.01" --attention_type frmomentum-linear-v3 --stepsize 0.01 --manualSeed 0 --gpu-id 0 &

# GPU 4

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.01" --attention_type frmomentum-linear --stepsize 0.01 --manualSeed 0 --gpu-id 4 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.01" --attention_type frmomentum-linear-v2 --stepsize 0.01 --manualSeed 0 --gpu-id 4 &

# GPU 5

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.05" --attention_type frmomentum-linear --stepsize 0.05 --manualSeed 0 --gpu-id 5 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.05" --attention_type frmomentum-linear-v2 --stepsize 0.05 --manualSeed 0 --gpu-id 5 &

# GPU 6

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v3_step_0.05" --attention_type frmomentum-linear-v3 --stepsize 0.05 --manualSeed 0 --gpu-id 6 &

python main_momentum_v3.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.1_step_sche_5" --attention_type frmomentum-linear --stepsize 0.1 --stepsize_schedule 5 --manualSeed 0 --gpu-id 6 &

# GPU 7

python main_momentum_v2.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.1_step_f_0.001" --attention_type frmomentum-linear --stepsize 0.1 --final_stepsize 0.001 --manualSeed 0 --gpu-id 7 &

python main_momentum_v2.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.1_step_f_0.01" --attention_type frmomentum-linear --stepsize 0.1 --final_stepsize 0.01 --manualSeed 0 --gpu-id 7 &

# Others

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.3" --attention_type frmomentum-linear --stepsize 0.3 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.3" --attention_type frmomentum-linear-v2 --stepsize 0.3 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v3_step_0.3" --attention_type frmomentum-linear-v3 --stepsize 0.3 --manualSeed 0 --gpu-id 2 &

python main_momentum_v2.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_1.0_step_f_0.001" --attention_type frmomentum-linear --stepsize 1.0 --final_stepsize 0.001 --manualSeed 0 --gpu-id 3 &

wait
echo "Done"
