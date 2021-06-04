#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.01_beta_0.999" --attention_type adamax-linear --mu 0.0 --stepsize 0.01 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.001_beta_0.999" --attention_type adamax-linear --mu 0.0 --stepsize 0.001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.0001_beta_0.999" --attention_type adamax-linear --mu 0.0 --stepsize 0.0001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.00001_beta_0.999" --attention_type adamax-linear --mu 0.0 --stepsize 0.00001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.01_beta_0.99" --attention_type adamax-linear --mu 0.0 --stepsize 0.01 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.001_beta_0.99" --attention_type adamax-linear --mu 0.0 --stepsize 0.001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.0001_beta_0.99" --attention_type adamax-linear --mu 0.0 --stepsize 0.0001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.00001_beta_0.99" --attention_type adamax-linear --mu 0.0 --stepsize 0.00001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.01_beta_0.9" --attention_type adamax-linear --mu 0.0 --stepsize 0.01 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.001_beta_0.9" --attention_type adamax-linear --mu 0.0 --stepsize 0.001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.0001_beta_0.9" --attention_type adamax-linear --mu 0.0 --stepsize 0.0001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.00001_beta_0.9" --attention_type adamax-linear --mu 0.0 --stepsize 0.00001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.01_beta_0.6" --attention_type adamax-linear --mu 0.0 --stepsize 0.01 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.001_beta_0.6" --attention_type adamax-linear --mu 0.0 --stepsize 0.001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.0001_beta_0.6" --attention_type adamax-linear --mu 0.0 --stepsize 0.0001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.00001_beta_0.6" --attention_type adamax-linear --mu 0.0 --stepsize 0.00001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.01_beta_0.3" --attention_type adamax-linear --mu 0.0 --stepsize 0.01 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.001_beta_0.3" --attention_type adamax-linear --mu 0.0 --stepsize 0.001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.0001_beta_0.3" --attention_type adamax-linear --mu 0.0 --stepsize 0.0001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.00001_beta_0.3" --attention_type adamax-linear --mu 0.0 --stepsize 0.00001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.01_beta_0.1" --attention_type adamax-linear --mu 0.0 --stepsize 0.01 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.001_beta_0.1" --attention_type adamax-linear --mu 0.0 --stepsize 0.001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.0001_beta_0.1" --attention_type adamax-linear --mu 0.0 --stepsize 0.0001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.00001_beta_0.1" --attention_type adamax-linear --mu 0.0 --stepsize 0.00001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.01_beta_0.01" --attention_type adamax-linear --mu 0.0 --stepsize 0.01 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.001_beta_0.01" --attention_type adamax-linear --mu 0.0 --stepsize 0.001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.0001_beta_0.01" --attention_type adamax-linear --mu 0.0 --stepsize 0.0001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.0_step_0.00001_beta_0.01" --attention_type adamax-linear --mu 0.0 --stepsize 0.00001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"


####################

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.01_beta_0.999" --attention_type adamax-linear --mu 0.1 --stepsize 0.01 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.001_beta_0.999" --attention_type adamax-linear --mu 0.1 --stepsize 0.001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.0001_beta_0.999" --attention_type adamax-linear --mu 0.1 --stepsize 0.0001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.00001_beta_0.999" --attention_type adamax-linear --mu 0.1 --stepsize 0.00001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.01_beta_0.99" --attention_type adamax-linear --mu 0.1 --stepsize 0.01 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.001_beta_0.99" --attention_type adamax-linear --mu 0.1 --stepsize 0.001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.0001_beta_0.99" --attention_type adamax-linear --mu 0.1 --stepsize 0.0001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.00001_beta_0.99" --attention_type adamax-linear --mu 0.1 --stepsize 0.00001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.01_beta_0.9" --attention_type adamax-linear --mu 0.1 --stepsize 0.01 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.001_beta_0.9" --attention_type adamax-linear --mu 0.1 --stepsize 0.001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.0001_beta_0.9" --attention_type adamax-linear --mu 0.1 --stepsize 0.0001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.00001_beta_0.9" --attention_type adamax-linear --mu 0.1 --stepsize 0.00001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.01_beta_0.6" --attention_type adamax-linear --mu 0.1 --stepsize 0.01 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.001_beta_0.6" --attention_type adamax-linear --mu 0.1 --stepsize 0.001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.0001_beta_0.6" --attention_type adamax-linear --mu 0.1 --stepsize 0.0001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.00001_beta_0.6" --attention_type adamax-linear --mu 0.1 --stepsize 0.00001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.01_beta_0.3" --attention_type adamax-linear --mu 0.1 --stepsize 0.01 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.001_beta_0.3" --attention_type adamax-linear --mu 0.1 --stepsize 0.001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.0001_beta_0.3" --attention_type adamax-linear --mu 0.1 --stepsize 0.0001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.00001_beta_0.3" --attention_type adamax-linear --mu 0.1 --stepsize 0.00001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.01_beta_0.1" --attention_type adamax-linear --mu 0.1 --stepsize 0.01 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.001_beta_0.1" --attention_type adamax-linear --mu 0.1 --stepsize 0.001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.0001_beta_0.1" --attention_type adamax-linear --mu 0.1 --stepsize 0.0001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.00001_beta_0.1" --attention_type adamax-linear --mu 0.1 --stepsize 0.00001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.01_beta_0.01" --attention_type adamax-linear --mu 0.1 --stepsize 0.01 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.001_beta_0.01" --attention_type adamax-linear --mu 0.1 --stepsize 0.001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.0001_beta_0.01" --attention_type adamax-linear --mu 0.1 --stepsize 0.0001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.1_step_0.00001_beta_0.01" --attention_type adamax-linear --mu 0.1 --stepsize 0.00001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

####################

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.01_beta_0.999" --attention_type adamax-linear --mu 0.3 --stepsize 0.01 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.001_beta_0.999" --attention_type adamax-linear --mu 0.3 --stepsize 0.001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.0001_beta_0.999" --attention_type adamax-linear --mu 0.3 --stepsize 0.0001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.00001_beta_0.999" --attention_type adamax-linear --mu 0.3 --stepsize 0.00001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.01_beta_0.99" --attention_type adamax-linear --mu 0.3 --stepsize 0.01 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.001_beta_0.99" --attention_type adamax-linear --mu 0.3 --stepsize 0.001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.0001_beta_0.99" --attention_type adamax-linear --mu 0.3 --stepsize 0.0001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.00001_beta_0.99" --attention_type adamax-linear --mu 0.3 --stepsize 0.00001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.01_beta_0.9" --attention_type adamax-linear --mu 0.3 --stepsize 0.01 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.001_beta_0.9" --attention_type adamax-linear --mu 0.3 --stepsize 0.001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.0001_beta_0.9" --attention_type adamax-linear --mu 0.3 --stepsize 0.0001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.00001_beta_0.9" --attention_type adamax-linear --mu 0.3 --stepsize 0.00001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.01_beta_0.6" --attention_type adamax-linear --mu 0.3 --stepsize 0.01 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.001_beta_0.6" --attention_type adamax-linear --mu 0.3 --stepsize 0.001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.0001_beta_0.6" --attention_type adamax-linear --mu 0.3 --stepsize 0.0001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.00001_beta_0.6" --attention_type adamax-linear --mu 0.3 --stepsize 0.00001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.01_beta_0.3" --attention_type adamax-linear --mu 0.3 --stepsize 0.01 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.001_beta_0.3" --attention_type adamax-linear --mu 0.3 --stepsize 0.001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.0001_beta_0.3" --attention_type adamax-linear --mu 0.3 --stepsize 0.0001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.00001_beta_0.3" --attention_type adamax-linear --mu 0.3 --stepsize 0.00001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.01_beta_0.1" --attention_type adamax-linear --mu 0.3 --stepsize 0.01 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.001_beta_0.1" --attention_type adamax-linear --mu 0.3 --stepsize 0.001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.0001_beta_0.1" --attention_type adamax-linear --mu 0.3 --stepsize 0.0001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.00001_beta_0.1" --attention_type adamax-linear --mu 0.3 --stepsize 0.00001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.01_beta_0.01" --attention_type adamax-linear --mu 0.3 --stepsize 0.01 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.001_beta_0.01" --attention_type adamax-linear --mu 0.3 --stepsize 0.001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.0001_beta_0.01" --attention_type adamax-linear --mu 0.3 --stepsize 0.0001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.3_step_0.00001_beta_0.01" --attention_type adamax-linear --mu 0.3 --stepsize 0.00001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

####################

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.01_beta_0.999" --attention_type adamax-linear --mu 0.6 --stepsize 0.01 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.001_beta_0.999" --attention_type adamax-linear --mu 0.6 --stepsize 0.001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.0001_beta_0.999" --attention_type adamax-linear --mu 0.6 --stepsize 0.0001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.00001_beta_0.999" --attention_type adamax-linear --mu 0.6 --stepsize 0.00001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.01_beta_0.99" --attention_type adamax-linear --mu 0.6 --stepsize 0.01 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.001_beta_0.99" --attention_type adamax-linear --mu 0.6 --stepsize 0.001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.0001_beta_0.99" --attention_type adamax-linear --mu 0.6 --stepsize 0.0001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.00001_beta_0.99" --attention_type adamax-linear --mu 0.6 --stepsize 0.00001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.01_beta_0.9" --attention_type adamax-linear --mu 0.6 --stepsize 0.01 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.001_beta_0.9" --attention_type adamax-linear --mu 0.6 --stepsize 0.001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.0001_beta_0.9" --attention_type adamax-linear --mu 0.6 --stepsize 0.0001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.00001_beta_0.9" --attention_type adamax-linear --mu 0.6 --stepsize 0.00001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.01_beta_0.6" --attention_type adamax-linear --mu 0.6 --stepsize 0.01 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.001_beta_0.6" --attention_type adamax-linear --mu 0.6 --stepsize 0.001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.0001_beta_0.6" --attention_type adamax-linear --mu 0.6 --stepsize 0.0001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.00001_beta_0.6" --attention_type adamax-linear --mu 0.6 --stepsize 0.00001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.01_beta_0.3" --attention_type adamax-linear --mu 0.6 --stepsize 0.01 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.001_beta_0.3" --attention_type adamax-linear --mu 0.6 --stepsize 0.001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.0001_beta_0.3" --attention_type adamax-linear --mu 0.6 --stepsize 0.0001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.00001_beta_0.3" --attention_type adamax-linear --mu 0.6 --stepsize 0.00001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.01_beta_0.1" --attention_type adamax-linear --mu 0.6 --stepsize 0.01 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.001_beta_0.1" --attention_type adamax-linear --mu 0.6 --stepsize 0.001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.0001_beta_0.1" --attention_type adamax-linear --mu 0.6 --stepsize 0.0001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.00001_beta_0.1" --attention_type adamax-linear --mu 0.6 --stepsize 0.00001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.01_beta_0.01" --attention_type adamax-linear --mu 0.6 --stepsize 0.01 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.001_beta_0.01" --attention_type adamax-linear --mu 0.6 --stepsize 0.001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.0001_beta_0.01" --attention_type adamax-linear --mu 0.6 --stepsize 0.0001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.6_step_0.00001_beta_0.01" --attention_type adamax-linear --mu 0.6 --stepsize 0.00001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

####################

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.01_beta_0.999" --attention_type adamax-linear --mu 0.9 --stepsize 0.01 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.001_beta_0.999" --attention_type adamax-linear --mu 0.9 --stepsize 0.001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.0001_beta_0.999" --attention_type adamax-linear --mu 0.9 --stepsize 0.0001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.00001_beta_0.999" --attention_type adamax-linear --mu 0.9 --stepsize 0.00001 --beta 0.999 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.01_beta_0.99" --attention_type adamax-linear --mu 0.9 --stepsize 0.01 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.001_beta_0.99" --attention_type adamax-linear --mu 0.9 --stepsize 0.001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.0001_beta_0.99" --attention_type adamax-linear --mu 0.9 --stepsize 0.0001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.00001_beta_0.99" --attention_type adamax-linear --mu 0.9 --stepsize 0.00001 --beta 0.99 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.01_beta_0.9" --attention_type adamax-linear --mu 0.9 --stepsize 0.01 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.001_beta_0.9" --attention_type adamax-linear --mu 0.9 --stepsize 0.001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.0001_beta_0.9" --attention_type adamax-linear --mu 0.9 --stepsize 0.0001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.00001_beta_0.9" --attention_type adamax-linear --mu 0.9 --stepsize 0.00001 --beta 0.9 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.01_beta_0.6" --attention_type adamax-linear --mu 0.9 --stepsize 0.01 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.001_beta_0.6" --attention_type adamax-linear --mu 0.9 --stepsize 0.001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.0001_beta_0.6" --attention_type adamax-linear --mu 0.9 --stepsize 0.0001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.00001_beta_0.6" --attention_type adamax-linear --mu 0.9 --stepsize 0.00001 --beta 0.6 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.01_beta_0.3" --attention_type adamax-linear --mu 0.9 --stepsize 0.01 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.001_beta_0.3" --attention_type adamax-linear --mu 0.9 --stepsize 0.001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.0001_beta_0.3" --attention_type adamax-linear --mu 0.9 --stepsize 0.0001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.00001_beta_0.3" --attention_type adamax-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.01_beta_0.1" --attention_type adamax-linear --mu 0.9 --stepsize 0.01 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.001_beta_0.1" --attention_type adamax-linear --mu 0.9 --stepsize 0.001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.0001_beta_0.1" --attention_type adamax-linear --mu 0.9 --stepsize 0.0001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.00001_beta_0.1" --attention_type adamax-linear --mu 0.9 --stepsize 0.00001 --beta 0.1 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"

# GPU 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.01_beta_0.01" --attention_type adamax-linear --mu 0.9 --stepsize 0.01 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.001_beta_0.01" --attention_type adamax-linear --mu 0.9 --stepsize 0.001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.0001_beta_0.01" --attention_type adamax-linear --mu 0.9 --stepsize 0.0001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.00001_beta_0.01" --attention_type adamax-linear --mu 0.9 --stepsize 0.00001 --beta 0.01 --epochs 10 --manualSeed 0 --gpu-id 3 &

wait
echo "Continue"