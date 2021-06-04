#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.1_beta_0.999" --attention_type fradamax-linear --stepsize 0.1 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.1_beta_0.01" --attention_type fradamax-linear --stepsize 0.1 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.1_beta_0.99" --attention_type fradamax-linear --stepsize 0.1 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.1_beta_0.1" --attention_type fradamax-linear --stepsize 0.1 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.1_beta_0.9" --attention_type fradamax-linear --stepsize 0.1 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.1_beta_0.3" --attention_type fradamax-linear --stepsize 0.1 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.1_beta_0.8" --attention_type fradamax-linear --stepsize 0.1 --beta 0.8 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.1_beta_0.6" --attention_type fradamax-linear --stepsize 0.1 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.999" --attention_type fradamax-linear --stepsize 0.3 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.01" --attention_type fradamax-linear --stepsize 0.3 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.99" --attention_type fradamax-linear --stepsize 0.3 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.1" --attention_type fradamax-linear --stepsize 0.3 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.9" --attention_type fradamax-linear --stepsize 0.3 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.3" --attention_type fradamax-linear --stepsize 0.3 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.8" --attention_type fradamax-linear --stepsize 0.3 --beta 0.8 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.3_beta_0.6" --attention_type fradamax-linear --stepsize 0.3 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.6_beta_0.999" --attention_type fradamax-linear --stepsize 0.6 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.6_beta_0.01" --attention_type fradamax-linear --stepsize 0.6 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.6_beta_0.99" --attention_type fradamax-linear --stepsize 0.6 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.6_beta_0.1" --attention_type fradamax-linear --stepsize 0.6 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.6_beta_0.9" --attention_type fradamax-linear --stepsize 0.6 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.6_beta_0.3" --attention_type fradamax-linear --stepsize 0.6 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.6_beta_0.8" --attention_type fradamax-linear --stepsize 0.6 --beta 0.8 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.6_beta_0.6" --attention_type fradamax-linear --stepsize 0.6 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.9_beta_0.999" --attention_type fradamax-linear --stepsize 0.9 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.9_beta_0.01" --attention_type fradamax-linear --stepsize 0.9 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.9_beta_0.99" --attention_type fradamax-linear --stepsize 0.9 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.9_beta_0.1" --attention_type fradamax-linear --stepsize 0.9 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.9_beta_0.9" --attention_type fradamax-linear --stepsize 0.9 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.9_beta_0.3" --attention_type fradamax-linear --stepsize 0.9 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.9_beta_0.8" --attention_type fradamax-linear --stepsize 0.9 --beta 0.8 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.9_beta_0.6" --attention_type fradamax-linear --stepsize 0.9 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_1.0_beta_0.999" --attention_type fradamax-linear --stepsize 1.0 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_1.0_beta_0.01" --attention_type fradamax-linear --stepsize 1.0 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_1.0_beta_0.99" --attention_type fradamax-linear --stepsize 1.0 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_1.0_beta_0.1" --attention_type fradamax-linear --stepsize 1.0 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_1.0_beta_0.9" --attention_type fradamax-linear --stepsize 1.0 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_1.0_beta_0.3" --attention_type fradamax-linear --stepsize 1.0 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_1.0_beta_0.8" --attention_type fradamax-linear --stepsize 1.0 --beta 0.8 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_1.0_beta_0.6" --attention_type fradamax-linear --stepsize 1.0 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.01_beta_0.999" --attention_type fradamax-linear --stepsize 0.01 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.01_beta_0.01" --attention_type fradamax-linear --stepsize 0.01 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.01_beta_0.99" --attention_type fradamax-linear --stepsize 0.01 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.01_beta_0.1" --attention_type fradamax-linear --stepsize 0.01 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.01_beta_0.9" --attention_type fradamax-linear --stepsize 0.01 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.01_beta_0.3" --attention_type fradamax-linear --stepsize 0.01 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.01_beta_0.8" --attention_type fradamax-linear --stepsize 0.01 --beta 0.8 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.01_beta_0.6" --attention_type fradamax-linear --stepsize 0.01 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.001_beta_0.999" --attention_type fradamax-linear --stepsize 0.001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.001_beta_0.01" --attention_type fradamax-linear --stepsize 0.001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.001_beta_0.99" --attention_type fradamax-linear --stepsize 0.001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.001_beta_0.1" --attention_type fradamax-linear --stepsize 0.001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.001_beta_0.9" --attention_type fradamax-linear --stepsize 0.001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.001_beta_0.3" --attention_type fradamax-linear --stepsize 0.001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.001_beta_0.8" --attention_type fradamax-linear --stepsize 0.001 --beta 0.8 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.001_beta_0.6" --attention_type fradamax-linear --stepsize 0.001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.0001_beta_0.999" --attention_type fradamax-linear --stepsize 0.0001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.0001_beta_0.01" --attention_type fradamax-linear --stepsize 0.0001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.0001_beta_0.99" --attention_type fradamax-linear --stepsize 0.0001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.0001_beta_0.1" --attention_type fradamax-linear --stepsize 0.0001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.0001_beta_0.9" --attention_type fradamax-linear --stepsize 0.0001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.0001_beta_0.3" --attention_type fradamax-linear --stepsize 0.0001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.0001_beta_0.8" --attention_type fradamax-linear --stepsize 0.0001 --beta 0.8 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.0001_beta_0.6" --attention_type fradamax-linear --stepsize 0.0001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.00001_beta_0.999" --attention_type fradamax-linear --stepsize 0.00001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.00001_beta_0.01" --attention_type fradamax-linear --stepsize 0.00001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.00001_beta_0.99" --attention_type fradamax-linear --stepsize 0.00001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.00001_beta_0.1" --attention_type fradamax-linear --stepsize 0.00001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.00001_beta_0.9" --attention_type fradamax-linear --stepsize 0.00001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.00001_beta_0.3" --attention_type fradamax-linear --stepsize 0.00001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.00001_beta_0.8" --attention_type fradamax-linear --stepsize 0.00001 --beta 0.8 --epochs 10 --manualSeed 0 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_step_0.00001_beta_0.6" --attention_type fradamax-linear --stepsize 0.00001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"