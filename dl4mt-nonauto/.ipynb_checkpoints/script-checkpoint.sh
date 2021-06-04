#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --main_path '/tanData/momentum_transformer/nonauto/full_softmax_seed_0' --seed 0 --num_gpus 2 --gpu 0

CUDA_VISIBLE_DEVICES=6,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'linear' --main_path '/tanData/momentum_transformer/nonauto/linear_seed_0_1' --seed 0 --num_gpus 2 --gpu 0

CUDA_VISIBLE_DEVICES=6,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.9 --stepsize 0.1 --main_path '/tanData/momentum_transformer/nonauto/momentum_seed_0' --seed 0 --num_gpus 2 --gpu 0

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'linear' --main_path '/tanData/momentum_transformer/nonauto/linear_with_mask_seed_0' --seed 0 --num_gpus 2 --gpu 0

CUDA_VISIBLE_DEVICES=0,1 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'linear' --main_path '/tanData/momentum_transformer/nonauto/linear_no_mask_seed_0' --seed 0 --num_gpus 2 --gpu 0

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.9 --stepsize 0.1 --main_path '/tanData/momentum_transformer/nonauto/momentum_seed_0' --seed 0 --num_gpus 2 --gpu 0

#################

CUDA_VISIBLE_DEVICES=4,5 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --maximum_steps 750000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_debug_seed_0' --seed 0 --num_gpus 2 --gpu 0

CUDA_VISIBLE_DEVICES=0,1 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'linear' --maximum_steps 750000 --main_path '/tanData/momentum_transformer/nonauto/linear_no_mask_debug_seed_0' --seed 0 --num_gpus 2 --gpu 0

CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.6_step_0.6_debug_seed_0' --seed 0 --num_gpus 2 --gpu 0 

CUDA_VISIBLE_DEVICES=6,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'linear' --maximum_steps 750000 --main_path '/tanData/momentum_transformer/nonauto/linear_with_mask_debug_seed_0' --seed 0 --num_gpus 2 --gpu 0


CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.8 --res_delta 0.01 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/momentum_mu_0.6_step_0.6_debug_seed_0' --seed 0 --num_gpus 2 --gpu 0 



CUDA_VISIBLE_DEVICES=2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.1 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.1_seed_0' --seed 0 --num_gpus 2 --gpu 0 



# for launching jobs on medusa
python /home/collab/tanmnguyen/repos/momentum-transformer/dl4mt-nonauto/run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.1 --res_delta 0.0001 --maximum_steps 50000 --main_path '/home/collab/tanmnguyen/results/momentum_transformer/nonauto/momentum_mu_0.6_step_0.6_rstep_0.1_seed_0' --data_root '/home/collab/tanmnguyen/tanData' --seed 0 --num_gpus 2 --gpu 0


# for loading model during test time

CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'big' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --main_path '/tanData/momentum_transformer/nonauto/full_softmax_seed_0_debug_debug' --model_path "/tanData/momentum_transformer/nonauto/full_softmax_big_seed_0_debug/models" --load_from "04.20_04.00.ar_distil_voc40k_2048_6_512_512_8_drop_0.1_0.0003_anne_anneal_steps_250000_high_tr4_2decs__refe_both_copy_dn_layer_argmax_" --seed 0 --num_gpus 4 --gpu 0



CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'big' --lr_schedule anneal --fast --valid_repeat_dec 20 --use_argmax --next_dec_input both --mode test --remove_repeats --load_dataset --debug --trg_len_option predict --use_predicted_trg_len --attn_type 'full' --model_path "/tanData/momentum_transformer/nonauto/full_softmax_big_seed_0_debug/models" --load_from "04.20_04.00.ar_distil_voc40k_2048_6_512_512_8_drop_0.1_0.0003_anne_anneal_steps_250000_high_tr4_2decs__refe_both_copy_dn_layer_argmax_"


CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'big' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --main_path '/tanData/momentum_transformer/nonauto/full_softmax_big_seed_0_debug' --seed 0 --num_gpus 4 --gpu 0