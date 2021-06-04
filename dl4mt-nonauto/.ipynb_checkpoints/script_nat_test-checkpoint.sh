#!/bin/bash

CUDA_VISIBLE_DEVICES=4,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 20 --use_argmax --next_dec_input both --mode test --remove_repeats --debug --trg_len_option predict --use_predicted_trg_len --attn_type 'full' --model_path "/tanData/momentum_transformer/nonauto/final/full_softmax_check_seed_0/models" --load_from "04.20_01.47.ar_distil_voc40k_2048_6_512_512_8_drop_0.1_0.0003_anne_anneal_steps_250000_high_tr4_2decs__refe_both_copy_dn_layer_argmax_" 



CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'big' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --main_path '/tanData/momentum_transformer/nonauto/full_softmax_seed_0_debug_debug' --model_path "/tanData/momentum_transformer/nonauto/full_softmax_big_seed_0_debug/models" --load_from "04.20_04.00.ar_distil_voc40k_2048_6_512_512_8_drop_0.1_0.0003_anne_anneal_steps_250000_high_tr4_2decs__refe_both_copy_dn_layer_argmax_" --seed 0 --num_gpus 4 --gpu 0



CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'big' --lr_schedule anneal --fast --valid_repeat_dec 20 --use_argmax --next_dec_input both --mode test --remove_repeats --load_dataset --debug --trg_len_option predict --use_predicted_trg_len --attn_type 'full' --model_path "/tanData/momentum_transformer/nonauto/full_softmax_big_seed_0_debug/models" --load_from "04.20_04.00.ar_distil_voc40k_2048_6_512_512_8_drop_0.1_0.0003_anne_anneal_steps_250000_high_tr4_2decs__refe_both_copy_dn_layer_argmax_"




python /home/collab/tanmnguyen/repos/momentum-transformer/dl4mt-nonauto/run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --maximum_steps 750000 --main_path '/home/collab/tanmnguyen/results/momentum_transformer/nonauto/full_softmax_check_seed_0' --data_root '/home/collab/tanmnguyen/tanData' --seed 0 --num_gpus 2 --gpu 0


CUDA_VISIBLE_DEVICES=4,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --main_path '/tanData/momentum_transformer/nonauto/full_softmax_debug_seed_0' --seed 0 --num_gpus 2 --gpu 0




CUDA_VISIBLE_DEVICES=4,5 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.1 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.1_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

CUDA_VISIBLE_DEVICES=6,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.2 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.2_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.3 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.3_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

CUDA_VISIBLE_DEVICES=6,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.4 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.4_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.5 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.5_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

CUDA_VISIBLE_DEVICES=6,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.6 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.6_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.7 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.7_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

CUDA_VISIBLE_DEVICES=6,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.8 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.8_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.9 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.9_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

CUDA_VISIBLE_DEVICES=6,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 1.0 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_1.0_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

CUDA_VISIBLE_DEVICES=4,5 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 2.0 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_2.0_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

CUDA_VISIBLE_DEVICES=6,7 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.01 --res_delta 0.0001 --maximum_steps 50000 --main_path '/tanData/momentum_transformer/nonauto/full_softmax_rstep_0.01_seed_0' --seed 0 --num_gpus 2 --gpu 0 &

wait
echo "Continue"

