#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# GPU 0

python main.py --dataset mnist --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 16 --iterations 937500 --evaluate_frequency 3750 --save_frequency 3750 --save_to "/tanData/momentum_transformer/image_generation/linear_causal_bs_16_seed_0" --manualSeed 0 --gpu-id 1 &

python main.py --dataset mnist --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 16 --iterations 937500 --evaluate_frequency 3750 --save_frequency 3750 --save_to "/tanData/momentum_transformer/image_generation/linear_causal_bs_16_seed_1" --manualSeed 1 --gpu-id 1 &

python main.py --dataset mnist --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 16 --iterations 937500 --evaluate_frequency 3750 --save_frequency 3750 --save_to "/tanData/momentum_transformer/image_generation/linear_causal_bs_16_seed_3" --manualSeed 3 --gpu-id 1 &

# python main.py --dataset mnist --attention_type full --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 16 --iterations 937500 --evaluate_frequency 3750 --save_frequency 3750 --save_to "/tanData/momentum_transformer/image_generation/full_softfmax_bs_16_seed_0" --manualSeed 0 --gpu-id 7 &

# python main.py --dataset mnist --attention_type full --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 16 --iterations 937500 --evaluate_frequency 3750 --save_frequency 3750 --save_to "/tanData/momentum_transformer/image_generation/full_softfmax_bs_16_seed_1" --manualSeed 1 --gpu-id 6 &


wait
echo "Done"
