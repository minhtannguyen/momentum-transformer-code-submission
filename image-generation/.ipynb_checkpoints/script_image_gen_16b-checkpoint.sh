#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python main_causal_res_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_resmomentum_causal_final_mu_0.1_step_0.9_rmu_0.1_rstep_0.9_delta_0.0001_seed_0" --mu 0.1 --stepsize 0.9 --res_mu 0.1 --res_stepsize 0.9 --delta 0.0001 --is_resw False --manualSeed 0 --gpu-id 0 &

python main_causal_res_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_resmomentum_causal_final_mu_0.1_step_0.9_rmu_0.1_rstep_0.9_delta_0.0001_seed_1" --mu 0.1 --stepsize 0.9 --res_mu 0.1 --res_stepsize 0.9 --delta 0.0001 --is_resw False --manualSeed 1 --gpu-id 1 &

wait
echo "Done"
