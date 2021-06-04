#!/bin/bash

python /home/collab/tanmnguyen/repos/momentum-transformer/dl4mt-nonauto/run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.1 --res_delta 0.0001 --maximum_steps 50000 --main_path '/home/collab/tanmnguyen/results/momentum_transformer/nonauto/momentum_mu_0.6_step_0.1_rstep_0.8_seed_0' --data_root '/home/collab/tanmnguyen/tanData' --seed 0 --num_gpus 2 --gpu 0


python /home/collab/tanmnguyen/repos/momentum-transformer/dl4mt-nonauto/run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --maximum_steps 750000 --main_path '/home/collab/tanmnguyen/results/momentum_transformer/nonauto/momentum_mu_0.6_step_0.6_seed_0' --data_root '/home/collab/tanmnguyen/tanData' --seed 0 --num_gpus 2 --gpu 0

python /home/collab/tanmnguyen/repos/momentum-transformer/dl4mt-nonauto/run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --maximum_steps 750000 --main_path '/home/collab/tanmnguyen/results/momentum_transformer/nonauto/full_softmax_check_seed_0' --data_root '/home/collab/tanmnguyen/tanData' --seed 0 --num_gpus 2 --gpu 0





python /home/collab/tanmnguyen/repos/momentum-transformer/dl4mt-nonauto/run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --maximum_steps 750000 --main_path '/home/collab/tanmnguyen/results/momentum_transformer/nonauto/full_softmax_check_seed_0' --data_root '/home/collab/tanmnguyen/tanData' --seed 0 --num_gpus 2 --gpu 0

python /home/collab/tanmnguyen/repos/momentum-transformer/dl4mt-nonauto/run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --res_type 'adaptive' --adaptive_type 'wang' --res_stepsize 0.8 --res_delta 0.0001 --maximum_steps 750000 --main_path '/home/collab/tanmnguyen/results/momentum_transformer/nonauto/full_softmax_rstep_0.8_seed_0' --seed 0 --num_gpus 2 --gpu 0 &