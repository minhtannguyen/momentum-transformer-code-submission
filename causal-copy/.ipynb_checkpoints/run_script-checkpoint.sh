#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Transformers

python main.py --save_to "/tanData/momentum_transformer/linear_causal" --manualSeed 0 --gpu-id 1

python main.py --save_to "/tanData/momentum_transformer/full_softfmax_debug" --attention_type full --manualSeed 0 --gpu-id 4

python main.py --save_to "/tanData/momentum_transformer/reformer" --attention_type reformer --manualSeed 0 --gpu-id 2

python main.py --save_to "/tanData/momentum_transformer/linear" --attention_type linear --manualSeed 0 --gpu-id 0


# RNN Transformers

python main_recurrent.py --save_to "/tanData/momentum_transformer/rnn_linear_causal" --attention_type causal-linear --manualSeed 0 --gpu-id 1


python main_recurrent.py --save_to "/tanData/momentum_transformer/rnn_full_softfmax" --attention_type full --manualSeed 0 --gpu-id 1

# Momentum Transformers

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0_step_1" --attention_type momentum-linear --mu 0.0 --stepsize 1.0 --manualSeed 0 --gpu-id 0

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.9" --attention_type momentum-linear --mu 0.1 --stepsize 0.9 --manualSeed 0 --gpu-id 1


python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_normgrad_causal_debug" --attention_type normgrad-linear --mu 0.1 --stepsize 0.9 --manualSeed 0 --gpu-id 7


python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_debug" --attention_type frmomentum-linear --stepsize 0.9 --manualSeed 0 --gpu-id 7

python main_momentum_v4.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_debug" --attention_type frmomentum-linear --stepsize 0.9 --start_clip_epoch 2 --manualSeed 0 --gpu-id 7

python main_momentum_v5.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_debug" --attention_type frmomentum-linear --stepsize 0.9 --clip_cycle 2 --manualSeed 0 --gpu-id 7

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_frmomentum_causal_debug" --attention_type frmomentum-linear-v3 --stepsize 0.6 --manualSeed 0 --gpu-id 1


# Adam Transformers

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_debug" --attention_type adam-linear --mu 0.9 --stepsize 0.1 --beta 0.3 --manualSeed 0 --gpu-id 7

python main_adam_v2.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_debug" --attention_type adam-linear --mu 0.9 --stepsize 0.5 --beta 0.3 --final_stepsize 0.0 --manualSeed 0 --gpu-id 7

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_biascorrect_causal_mu_0.9_step_0.1_beta_0.3" --attention_type adam-linear --mu 0.9 --stepsize 0.1 --beta 0.3 --manualSeed 0 --gpu-id 0


python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_r_causal_debug" --attention_type adam-linear --mu 0.9 --stepsize 0.1 --beta 0.3 --manualSeed 0 --gpu-id 0

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adam_r_v2_causal_debug" --attention_type adam-linear --mu 0.9 --stepsize 0.000001 --beta 0.3 --manualSeed 0 --gpu-id 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamr_causal_debug" --attention_type adamr-linear --mu 0.9 --stepsize 0.5 --beta 0.3 --manualSeed 0 --gpu-id 0


python main_adam_analysis.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_debug" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --manualSeed 0 --gpu-id 0 

# Nesterov Transformer

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nesterov_causal_debug" --attention_type nesterov-linear --stepsize 1.0 --manualSeed 0 --gpu-id 0

# Momentum Transformer with a different stepsize for each attention unit

python main_momentum_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_momentum_causal_debug" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --stepsize_factor 0.5 --manualSeed 0 --gpu-id 0

# Adam Transformer with a different stepsize for each attention unit

python main_adam_local_stepsize.py --save_to "/tanData/momentum_transformer/rnn_local_adam_causal_debug" --attention_type adam-linear --mu 0.9 --stepsize 0.01 --stepsize_factor 0.1 --beta 0.3 --manualSeed 0 --gpu-id 7

# LSMomentum Transformer

python main_momentum_learned.py --save_to "/tanData/momentum_transformer/rnn_lsmomentum_causal_debug" --attention_type lsmomentum-linear --mu 0.1 --stepsize 0.6 --random_stepsize --manualSeed 0 --gpu-id 0

python main_momentum_learned.py --save_to "/tanData/momentum_transformer/rnn_lsmomentum_causal_debug" --attention_type lsmomentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 0

# python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_lsmomentum_causal_mu_0.1_step_0.6" --attention_type lsmomentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 0  


# Adam Analysis

python main_adam_analysis.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_analysis_seed_0" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --manualSeed 0 --gpu-id 0 

python main_adam_analysis.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_analysis_seed_1" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --manualSeed 1 --gpu-id 0 

# Adamax

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_debug" --attention_type adamax-linear --mu 0.9 --stepsize 0.1 --beta 0.3 --manualSeed 0 --gpu-id 0

# Adam reg

python main_adam_reg.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_reg_debug" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --reg_coef 0.00001 --manualSeed 0 --gpu-id 0 


python main_adam_reg.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.00001_beta_0.3_reg_0.00000001" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --reg_coef 0.00000001 --epochs 35 --manualSeed 0 --gpu-id 3 &

python main_adam_reg.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.00001_beta_0.3_reg_0.000000001" --attention_type adam-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --reg_coef 0.000000001 --epochs 35 --manualSeed 0 --gpu-id 3 &

# Adam clip

python main_adam_analysis.py --save_to "/tanData/momentum_transformer/rnn_adam_causal_mu_0.9_step_0.00001_beta_0.3_clip_1.0" --attention_type adamclip-linear --mu 0.9 --stepsize 0.00001 --beta 0.3 --manualSeed 0 --gpu-id 2

# Fradamax 

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_debug" --attention_type fradamax-linear --stepsize 0.1 --beta 0.9 --delta 0.001 --epochs 10 --manualSeed 0 --gpu-id 5 

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_debug" --attention_type fradamax-linear-v2 --stepsize 0.1 --beta 0.9 --delta 0.001 --epochs 10 --manualSeed 0 --gpu-id 2

# Causal Momentum Transformer

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/causal_momentum_debug" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 7

python prediction_momentum.py --save_to "/tanData/momentum_transformer/causal_momentum_prediction_debug" --continue_from "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_predict_seed_0/model_best_acc.pth.tar" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 5

python prediction_causal_momentum.py --save_to "/tanData/momentum_transformer/causal_momentum_prediction_debug" --continue_from "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_predict_seed_0/model_best_acc.pth.tar" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 5

python prediction.py --save_to "/tanData/momentum_transformer/causal_momentum_prediction_debug" --continue_from "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_predict_seed_0/model_best_acc.pth.tar" --attention_type causal-linear --manualSeed 0 --gpu-id 5

python prediction_momentum_v2.py --save_to "/tanData/momentum_transformer/causal_momentum_prediction_debug" --continue_from "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_predict_seed_0/model_best_acc.pth.tar" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --manualSeed 0 --gpu-id 5

python prediction.py --save_to "/tanData/momentum_transformer/causal_momentum_prediction_debug" --continue_from "/tanData/momentum_transformer/linear_causal_predict_seed_0/model_best_acc.pth.tar" --attention_type causal-linear --manualSeed 0 --gpu-id 5

python prediction_momentum.py --save_to "/tanData/momentum_transformer/causal_momentum_prediction_debug" --continue_from "/tanData/momentum_transformer/linear_causal_predict_seed_0/model_best_acc.pth.tar" --attention_type causal-linear --manualSeed 0 --gpu-id 5

python prediction_momentum_v2.py --save_to "/tanData/momentum_transformer/causal_momentum_prediction_debug" --continue_from "/tanData/momentum_transformer/linear_causal_predict_seed_0/model_best_acc.pth.tar" --attention_type causal-linear --manualSeed 0 --gpu-id 5

# Causal NCMomentum Transformer

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/causal_ncmomentum_debug" --attention_type causal-ncmomentum --stepsize 0.6 --manualSeed 0 --gpu-id 3

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/causal_ncmomentum_debug" --attention_type causal-ncmomentum-v2 --stepsize 0.6 --manualSeed 0 --gpu-id 1

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/causal_ncmomentum_debug" --attention_type causal-ncmomentum-v3 --stepsize 0.6 --manualSeed 0 --gpu-id 3

# Causal NCMomentum Transformer
python main_causal_momentum.py --save_to "/tanData/momentum_transformer/causal_ncmomentum_debug" --attention_type causal-ncmomentum --stepsize 0.1 --epochs 100 --manualSeed 0 --gpu-id 0

# linear
python main.py --save_to "/tanData/momentum_transformer/linear_causal_debug" --epochs 600 --manualSeed 0 --gpu-id 0 

########################################
### Image ###
########################################

# Linear Transformer
python main.py --dataset mnist --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/linear_causal_debug" --manualSeed 0 --gpu-id 0

python main.py --dataset mnist --attention_type reformer --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --chunk_size 29 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/reformer" --manualSeed 0 --gpu-id 6

python main.py --dataset mnist --attention_type full --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/full_softfmax" --manualSeed 0 --gpu-id 7

# Adaptive Transformer

python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum-py --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/causal_ncmomentum_py_debug" --stepsize 1.0 --delta 0.0001 --manualSeed 0 --gpu-id 7

# python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_step_1.0_delta_0.0001_seed_0" --stepsize 1.0 --delta 0.0001 --manualSeed 0 --gpu-id 0

python main.py --dataset mnist --attention_type full --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/causal_ncmomentum_py_debug" --manualSeed 0 --gpu-id 7


python main_causal_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 60000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/ncmomentum_causal_debug" --stepsize 1.0 --delta 0.0001 --manualSeed 0 --gpu-id 0 

# Image Generation CIFAR10

python main.py --dataset cifar10 --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_linear_causal_seed_0" --manualSeed 0 --gpu-id 3 &

python main.py --dataset cifar10 --attention_type full --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 1 --iterations 5000000 --evaluate_frequency 50000 --save_frequency 50000 --save_to "/tanData/momentum_transformer/image_generation/cifar_full_softfmax_seed_0" --manualSeed 0 --gpu-id 3 &

python main.py --dataset cifar10 --attention_type reformer --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --chunk_size 37 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_reformer_seed_0" --manualSeed 0 --gpu-id 3 &

python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "/tanData/momentum_transformer/image_generation/cifar_momentum_causal_debug" --mu 0.2 --stepsize 0.7 --delta 0.0001 --manualSeed 0 --gpu-id 2


# Image Generation Prediction MNIST

python prediction.py --dataset mnist --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --index 0-10 --offset 1 --recurrent --force_cpu --save_image /root/repos/momentum-transformer/image-generation/images/linear/mnist.{}.{}.png /tanData/momentum_transformer/image_generation/linear_causal_seed_1/model_best_bpd.pth.tar

python prediction.py --dataset mnist --attention_type full --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --index 0-10 --offset 1 --recurrent --force_cpu --save_image /root/repos/momentum-transformer/image-generation/images/softmax/mnist.{}.{}.png /tanData/momentum_transformer/image_generation/full_softfmax_seed_1/model_best_bpd.pth.tar

python prediction.py --dataset mnist --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --index 0-10 --offset 1 --recurrent --save_image /root/repos/momentum-transformer/image-generation/images/linear/mnist.{}.{}.png /tanData/momentum_transformer/image_generation/linear_causal_seed_1/model_best_bpd.pth.tar


python prediction_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --stepsize 0.7 --index 0-10 --offset 1 --recurrent --force_cpu --save_image /root/repos/momentum-transformer/image-generation/images/ncmomentum/mnist_ncmom.{}.{}.png /tanData/momentum_transformer/image_generation/ncmomentum_causal_final_step_0.7_delta_0.0001_seed_0/model.pth.tar

python prediction_momentum.py --dataset mnist --attention_type causal-ncmomentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --stepsize 0.7 --index 0-10 --offset 392 --force_cpu --save_image /root/repos/momentum-transformer/image-generation/images/ncmomentum/mnist_con_ncmom.{}.{}.png /tanData/momentum_transformer/image_generation/ncmomentum_causal_final_step_0.7_delta_0.0001_seed_0/model.pth.tar


python prediction_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --mu 0.2 --stepsize 0.7 --index 0-10 --offset 392 --recurrent --force_cpu --save_image /root/repos/momentum-transformer/image-generation/images/momentum/mnist_mom.{}.{}.png /tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.2_step_0.7_delta_0.0001_seed_0/model_best_loss.pth.tar


python prediction_resadaptive.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --mu 0.6 --stepsize 0.9 --res_stepsize 0.99 --delta 0.0001 --adaptive_type "wang" --is_resw False --index 0-10 --offset 1 --recurrent --force_cpu --save_image /root/repos/momentum-transformer/image-generation/images/adaptive_res_momentum/mnist_mom.{}.{}.png /tanData/momentum_transformer/image_generation/adaptivemomentum_wang_causal_final_mu_0.6_step_0.9_rstep_0.99_delta_0.0001_seed_1/model_best_bpd.pth.tar

python prediction_resadaptive.py --dataset cigfar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --mu 0.1 --stepsize 0.9 --res_stepsize 0.9 --delta 0.0001 --adaptive_type "wang" --is_resw False --index 0-10 --offset 1 --recurrent --force_cpu --save_image /root/repos/momentum-transformer/image-generation/images/cifar_adaptive_res_momentum/cifar_mom.{}.{}.png /tanData/momentum_transformer/image_generation/cifar_adaptivemomentum_wang_causal_final_mu_0.1_step_0.9_rstep_0.9_delta_0.0001_seed_0/model_best_bpd.pth.tar --image_shape 32,32,3


# Recurrent finetuning for copy task

python main_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_seed_0_finetune_debug" --continue_from "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_final_seed_0/model_best_loss.pth.tar" --attention_type causal-ncmomentum --stepsize 0.6 --epochs 800 --manualSeed 0 --gpu-id 7 &

# Momentum Residual for Copy Task

python main_causal_res_momentum.py --save_to "/tanData/momentum_transformer/resmomentum_causal_mu_0.1_step_0.6_debug" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 3 

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentum_causal_mu_0.1_step_0.6_debug" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.8 --res_delta 0.0001 --adaptive_type "fr" --epochs 600 --manualSeed 0 --gpu-id 4 

python main_causal_momentum.py --save_to "/tanData/momentum_transformer/momentum_causal_mu_0.1_step_0.6_debug" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 1 

python main_causal_adaptive_momentum.py --save_to "/tanData/momentum_transformer/adaptivemomentumW_fr_causal_debug" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 2.0 --res_delta 0.0001 --adaptive_type "fr" --is_resw True --epochs 600 --manualSeed 0 --gpu-id 4

# Momentum Residual for Image Generation

python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_debug" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.99 --res_stepsize 0.5  --manualSeed 0 --gpu-id 5 &


python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 192000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/resmomentum_causal_debug" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_stepsize 0.8 --adaptive_type "wang" --is_resw False  --manualSeed 0 --gpu-id 4

# python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "/tanData/momentum_transformer/image_generation/momentum_causal_final_mu_0.6_step_0.9_delta_0.0001_seed_0" --mu 0.6 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 6 &

# Res Softmax Transformer

python main_res.py --save_to "/tanData/momentum_transformer/res_softfmax_debug" --attention_type full --epochs 100 --res_mu 0.999 --res_stepsize 0.2 --is_resw True --manualSeed 0 --gpu-id 4

python main_adaptive.py --save_to "/tanData/momentum_transformer/adaptive_softfmax_debug" --attention_type full --epochs 100 --res_stepsize 0.2 --res_delta 0.001 --adaptive_type "fr" --is_resw True --manualSeed 0 --gpu-id 4

# Sparse Transformers

python main_sparse.py --save_to "/tanData/momentum_transformer/sparse_full_softfmax_diag_2_seed_0" --attention_type sparse-full --diag_size 2 --manualSeed 0 --gpu-id 5