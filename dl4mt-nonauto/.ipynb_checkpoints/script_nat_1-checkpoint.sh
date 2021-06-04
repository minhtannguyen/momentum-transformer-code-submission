#!/bin/bash

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.1 --stepsize 0.1 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.1_step_0.1_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.3 --stepsize 0.1 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.3_step_0.1_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.1 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.6_step_0.1_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.9 --stepsize 0.1 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.9_step_0.1_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.1 --stepsize 0.3 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.1_step_0.3_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.3 --stepsize 0.3 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.3_step_0.3_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.3 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.6_step_0.3_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.9 --stepsize 0.3 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.9_step_0.3_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.1 --stepsize 0.6 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.1_step_0.6_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.3 --stepsize 0.6 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.3_step_0.6_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.6_step_0.6_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.9 --stepsize 0.6 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.9_step_0.6_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.1 --stepsize 0.9 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.1_step_0.9_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.3 --stepsize 0.9 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.3_step_0.9_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.9 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.6_step_0.9_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.9 --stepsize 0.9 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.9_step_0.9_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"