#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# GPU 6

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_1.0" --attention_type frmomentum-linear --stepsize 1.0 --epochs 10 --manualSeed 0 --gpu-id 6 &

# GPU 7

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.9" --attention_type frmomentum-linear --stepsize 0.9 --epochs 10 --manualSeed 0 --gpu-id 7 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.6" --attention_type frmomentum-linear --stepsize 0.6 --epochs 10 --manualSeed 0 --gpu-id 7 &

wait
echo "Continue"

# GPU 6

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.3" --attention_type frmomentum-linear --stepsize 0.3 --epochs 10 --manualSeed 0 --gpu-id 6 &

# GPU 7

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.1" --attention_type frmomentum-linear --stepsize 0.1 --epochs 10 --manualSeed 0 --gpu-id 7 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.01" --attention_type frmomentum-linear --stepsize 0.01 --epochs 10 --manualSeed 0 --gpu-id 7 &


wait
echo "Done"
