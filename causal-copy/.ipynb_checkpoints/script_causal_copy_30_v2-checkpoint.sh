#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# GPU 7

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.6_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.6 --epochs 50 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.4_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.4 --epochs 50 --manualSeed 0 --gpu-id 7 &

wait
echo "Continue"

# GPU 7

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.1_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.1 --epochs 50 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.9_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.9 --epochs 50 --manualSeed 0 --gpu-id 7 &

wait
echo "Continue"

# GPU 7

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.2_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.2 --epochs 50 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_2.0_seed_0" --attention_type causal-ncmomentum-py --stepsize 2.0 --epochs 50 --manualSeed 0 --gpu-id 7 &

wait
echo "Continue"

# GPU 7

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.5_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.5 --epochs 50 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.7_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.7 --epochs 50 --manualSeed 0 --gpu-id 7 &

wait
echo "Continue"

# GPU 7

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.8_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.8 --epochs 50 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.3_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.3 --epochs 50 --manualSeed 0 --gpu-id 7 &

wait
echo "Continue"

# GPU 7

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.99_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.99 --epochs 50 --manualSeed 0 --gpu-id 7 &

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/ncmomentum_py_causal_step_0.01_seed_0" --attention_type causal-ncmomentum-py --stepsize 0.01 --epochs 50 --manualSeed 0 --gpu-id 7 &

wait
echo "Done"
