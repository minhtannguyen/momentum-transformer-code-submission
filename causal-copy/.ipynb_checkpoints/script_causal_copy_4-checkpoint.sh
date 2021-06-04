#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.5_stepf_0.0_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.5 --beta 0.3 --final_stepsize 0.0 --manualSeed 0 --gpu-id 0 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.5_stepf_1e_-6_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.5 --beta 0.3 --final_stepsize 0.000001 --manualSeed 0 --gpu-id 0 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.5_stepf_1e_-7_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.5 --beta 0.3 --final_stepsize 0.0000001 --manualSeed 0 --gpu-id 0 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.5_stepf_1e_-8_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.5 --beta 0.3 --final_stepsize 0.00000001 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_1.0_stepf_0.0_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 1.0 --beta 0.3 --final_stepsize 0.0 --manualSeed 0 --gpu-id 1 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_1.0_stepf_1e_-6_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 1.0 --beta 0.3 --final_stepsize 0.000001 --manualSeed 0 --gpu-id 1 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_1.0_stepf_1e_-7_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 1.0 --beta 0.3 --final_stepsize 0.0000001 --manualSeed 0 --gpu-id 1 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_1.0_stepf_1e_-8_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 1.0 --beta 0.3 --final_stepsize 0.00000001 --manualSeed 0 --gpu-id 1 &

# GPU 2

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.1_stepf_0.0_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.1 --beta 0.3 --final_stepsize 0.0 --manualSeed 0 --gpu-id 2 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.1_stepf_1e_-6_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.1 --beta 0.3 --final_stepsize 0.000001 --manualSeed 0 --gpu-id 2 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.1_stepf_1e_-7_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.1 --beta 0.3 --final_stepsize 0.0000001 --manualSeed 0 --gpu-id 2 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.1_stepf_1e_-8_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.1 --beta 0.3 --final_stepsize 0.00000001 --manualSeed 0 --gpu-id 2 &


# GPU 3

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.2_stepf_0.0_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.2 --beta 0.3 --final_stepsize 0.0 --manualSeed 0 --gpu-id 3 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.2_stepf_1e_-6_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.2 --beta 0.3 --final_stepsize 0.000001 --manualSeed 0 --gpu-id 3 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.2_stepf_1e_-7_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.2 --beta 0.3 --final_stepsize 0.0000001 --manualSeed 0 --gpu-id 3 &

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_stepi_0.2_stepf_1e_-8_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.2 --beta 0.3 --final_stepsize 0.00000001 --manualSeed 0 --gpu-id 3 &

wait
echo "Done"
