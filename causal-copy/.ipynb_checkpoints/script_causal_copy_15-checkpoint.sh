#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_0.3" --attention_type frmomentum-linear --stepsize 0.3 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v2_step_0.3" --attention_type frmomentum-linear-v2 --stepsize 0.3 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_v3_step_0.3" --attention_type frmomentum-linear-v3 --stepsize 0.3 --manualSeed 0 --gpu-id 2 &

python main_momentum_v2.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_step_1.0_step_f_0.001" --attention_type frmomentum-linear --mu --stepsize 1.0 --final_stepsize 0.001 --manualSeed 0 --gpu-id 3 &

wait
echo "Done"
