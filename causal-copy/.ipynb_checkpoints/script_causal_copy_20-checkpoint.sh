#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_momentum_learned.py --save_to "/tanData/momentum_transformer/rnn_lsmomentum_causal_mu_0.1_step_random" --attention_type lsmomentum-linear --mu 0.1 --stepsize 0.6 --random_stepsize --manualSeed 0 --gpu-id 0 &

python main_momentum_learned.py --save_to "/tanData/momentum_transformer/rnn_lsmomentum_causal_mu_0.1_step_0.6" --attention_type lsmomentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 0 &


# GPU 1




# GPU 2



# GPU 3


# GPU 4



# GPU 5


# GPU 6



# GPU 7




wait
echo "Done"
