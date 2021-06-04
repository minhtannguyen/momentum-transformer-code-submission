#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python main_res.py --save_to "/tanData/momentum_transformer/res_softfmax_final_rmu_0.1_rstep_1.0_seed_1" --attention_type full --epochs 100 --res_mu 0.1 --res_stepsize 1.0 --is_resw False --manualSeed 1 --gpu-id 5 &

python main_res.py --save_to "/tanData/momentum_transformer/res_softfmax_final_rmu_0.1_rstep_1.0_seed_2" --attention_type full --epochs 100 --res_mu 0.1 --res_stepsize 1.0 --is_resw False --manualSeed 2 --gpu-id 5 &

python main_res.py --save_to "/tanData/momentum_transformer/res_softfmax_final_rmu_0.1_rstep_1.0_seed_3" --attention_type full --epochs 100 --res_mu 0.1 --res_stepsize 1.0 --is_resw False --manualSeed 3 --gpu-id 5 &

python main_res.py --save_to "/tanData/momentum_transformer/res_softfmax_final_rmu_0.1_rstep_1.0_seed_4" --attention_type full --epochs 100 --res_mu 0.1 --res_stepsize 1.0 --is_resw False --manualSeed 4 --gpu-id 6 &

wait
echo "Continue"
