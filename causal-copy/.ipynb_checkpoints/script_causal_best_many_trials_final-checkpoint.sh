#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GPU 0

python main.py --save_to "/tanData/momentum_transformer/full_softfmax_final_seed_0" --attention_type full --epochs 600 --manualSeed 0 --gpu-id 0 &

python main.py --save_to "/tanData/momentum_transformer/full_softfmax_final_seed_1" --attention_type full --epochs 600 --manualSeed 1 --gpu-id 0 &

python main.py --save_to "/tanData/momentum_transformer/full_softfmax_final_seed_2" --attention_type full --epochs 600 --manualSeed 2 --gpu-id 0 &

python main.py --save_to "/tanData/momentum_transformer/full_softfmax_final_seed_3" --attention_type full --epochs 600 --manualSeed 3 --gpu-id 0 &

python main.py --save_to "/tanData/momentum_transformer/full_softfmax_final_seed_4" --attention_type full --epochs 600 --manualSeed 4 --gpu-id 0 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_v2_step_0.3_beta_0.999_delta_0.1_final_seed_0" --attention_type fradamax-linear-v2 --stepsize 0.3 --beta 0.999 --delta 0.1 --epochs 600 --manualSeed 0 --gpu-id 0 &

# GPU 1

python main.py --save_to "/tanData/momentum_transformer/reformer_final_seed_0" --attention_type reformer --epochs 600 --manualSeed 0 --gpu-id 1 &

python main.py --save_to "/tanData/momentum_transformer/reformer_final_seed_1" --attention_type reformer --epochs 600 --manualSeed 1 --gpu-id 1 &

python main.py --save_to "/tanData/momentum_transformer/reformer_final_seed_2" --attention_type reformer --epochs 600 --manualSeed 2 --gpu-id 1 &

python main.py --save_to "/tanData/momentum_transformer/reformer_final_seed_3" --attention_type reformer --epochs 600 --manualSeed 3 --gpu-id 1 &

python main.py --save_to "/tanData/momentum_transformer/reformer_final_seed_4" --attention_type reformer --epochs 600 --manualSeed 4 --gpu-id 1 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_v2_step_0.3_beta_0.999_delta_0.1_final_seed_1" --attention_type fradamax-linear-v2 --stepsize 0.3 --beta 0.999 --delta 0.1 --epochs 600 --manualSeed 1 --gpu-id 1 &

# GPU 2

python main.py --save_to "/tanData/momentum_transformer/linear_causal_final_seed_0" --epochs 600 --manualSeed 0 --gpu-id 2 &

python main.py --save_to "/tanData/momentum_transformer/linear_causal_final_seed_1" --epochs 600 --manualSeed 1 --gpu-id 2 &

python main.py --save_to "/tanData/momentum_transformer/linear_causal_final_seed_2" --epochs 600 --manualSeed 2 --gpu-id 2 &

python main.py --save_to "/tanData/momentum_transformer/linear_causal_final_seed_3" --epochs 600 --manualSeed 3 --gpu-id 2 &

python main.py --save_to "/tanData/momentum_transformer/linear_causal_final_seed_4" --epochs 600 --manualSeed 4 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_v2_step_0.3_beta_0.999_delta_0.1_final_seed_2" --attention_type fradamax-linear-v2 --stepsize 0.3 --beta 0.999 --delta 0.1 --epochs 600 --manualSeed 2 --gpu-id 2 &

# GPU 3

python main_recurrent.py --save_to "/tanData/momentum_transformer/rnn_linear_causal_final_seed_0" --attention_type causal-linear --epochs 600 --manualSeed 0 --gpu-id 3 &

python main_recurrent.py --save_to "/tanData/momentum_transformer/rnn_linear_causal_final_seed_1" --attention_type causal-linear --epochs 600 --manualSeed 1 --gpu-id 3 &

python main_recurrent.py --save_to "/tanData/momentum_transformer/rnn_linear_causal_final_seed_2" --attention_type causal-linear --epochs 600 --manualSeed 2 --gpu-id 3 &

python main_recurrent.py --save_to "/tanData/momentum_transformer/rnn_linear_causal_final_seed_3" --attention_type causal-linear --epochs 600 --manualSeed 3 --gpu-id 3 &

python main_recurrent.py --save_to "/tanData/momentum_transformer/rnn_linear_causal_final_seed_4" --attention_type causal-linear --epochs 600 --manualSeed 4 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_v2_step_0.3_beta_0.999_delta_0.1_final_seed_3" --attention_type fradamax-linear-v2 --stepsize 0.3 --beta 0.999 --delta 0.1 --epochs 600 --manualSeed 3 --gpu-id 3 &

# GPU 4

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_fradamax_causal_v2_step_0.3_beta_0.999_delta_0.1_final_seed_4" --attention_type fradamax-linear-v2 --stepsize 0.3 --beta 0.999 --delta 0.1 --epochs 600 --manualSeed 4 --gpu-id 4 &

wait
echo "Continue"

# GPU 0

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_final_seed_0" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_final_seed_1" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 1 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_final_seed_2" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 2 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_final_seed_3" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 3 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_momentum_causal_mu_0.1_step_0.6_final_seed_4" --attention_type momentum-linear --mu 0.1 --stepsize 0.6 --epochs 600 --manualSeed 4 --gpu-id 0 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6_final_seed_4" --attention_type frmomentum-linear-v2 --stepsize 0.6 --epochs 600 --manualSeed 4 --gpu-id 0 &

# GPU 1

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6_final_seed_0" --attention_type frmomentum-linear-v2 --stepsize 0.6 --epochs 600 --manualSeed 0 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6_final_seed_1" --attention_type frmomentum-linear-v2 --stepsize 0.6 --epochs 600 --manualSeed 1 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6_final_seed_2" --attention_type frmomentum-linear-v2 --stepsize 0.6 --epochs 600 --manualSeed 2 --gpu-id 1 &

python main_momentum.py --save_to "/tanData/momentum_transformer/rnn_nfrmomentum_causal_v2_step_0.6_final_seed_3" --attention_type frmomentum-linear-v2 --stepsize 0.6 --epochs 600 --manualSeed 3 --gpu-id 1 &

# GPU 2

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_final_seed_0" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 0 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_final_seed_1" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 1 --gpu-id 2 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_final_seed_2" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 2 --gpu-id 2 &

# GPU 3

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_final_seed_3" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 3 --gpu-id 3 &

python main_adam.py --save_to "/tanData/momentum_transformer/rnn_adamax_causal_mu_0.9_step_0.3_beta_0.6_final_seed_4" --attention_type adamax-linear --mu 0.9 --stepsize 0.3 --beta 0.6 --epochs 600 --manualSeed 4 --gpu-id 3 &


wait
echo "Done"
