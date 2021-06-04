#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v3_step_0.4" --attention_type frmomentum-linear-v3 --epochs 200 --stepsize 0.4 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v3_step_0.4_seed_1" --attention_type frmomentum-linear-v3 --epochs 200 --stepsize 0.4 --manualSeed 1 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v3_step_0.4_seed_2" --attention_type frmomentum-linear-v3 --epochs 200 --stepsize 0.4 --manualSeed 2 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v3_step_0.4_seed_3" --attention_type frmomentum-linear-v3 --epochs 200 --stepsize 0.4 --manualSeed 3 --gpu-id 0 &


# GPU 1

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.6 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6_seed_1" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.6 --manualSeed 1 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6_seed_2" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.6 --manualSeed 2 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6_seed_3" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.6 --manualSeed 3 --gpu-id 1 &


# GPU 2

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_step_0.6" --attention_type frmomentum-linear --epochs 200 --stepsize 0.6 --manualSeed 0 --gpu-id 2 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_step_0.6_seed_1" --attention_type frmomentum-linear --epochs 200 --stepsize 0.6 --manualSeed 1 --gpu-id 2 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_step_0.6_seed_2" --attention_type frmomentum-linear --epochs 200 --stepsize 0.6 --manualSeed 2 --gpu-id 2 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_step_0.6_seed_3" --attention_type frmomentum-linear --epochs 200 --stepsize 0.6 --manualSeed 3 --gpu-id 2 &


# GPU 3

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.4" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.4 --manualSeed 0 --gpu-id 3 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.4_seed_1" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.4 --manualSeed 1 --gpu-id 3 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.4_seed_2" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.4 --manualSeed 2 --gpu-id 3 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.4_seed_3" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.4 --manualSeed 3 --gpu-id 3 &


# GPU 4

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v3_step_0.4_seed_4" --attention_type frmomentum-linear-v3 --epochs 200 --stepsize 0.4 --manualSeed 4 --gpu-id 4 &


# GPU 5

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6_seed_4" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.6 --manualSeed 4 --gpu-id 5 &

# GPU 6

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_step_0.6_seed_4" --attention_type frmomentum-linear --epochs 200 --stepsize 0.6 --manualSeed 4 --gpu-id 6 &

# GPU 7

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.4_seed_4" --attention_type frmomentum-linear-v2 --epochs 200 --stepsize 0.4 --manualSeed 4 --gpu-id 7 &

wait
echo "Done"