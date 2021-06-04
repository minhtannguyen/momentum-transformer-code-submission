Copy Task
=========

In this task, the transformer learns to copy a random sequence of symbols. This
benchmark requires very sparse random attention and is a hard task for dense
attentions like linear attention.

The transformer reads, one element at a time, a sequence like the following

    0 t1 t2 t3 t4 0 t1 t2 t3 t4

and must predict the next element. The loss is taken in the second half of the
sequence as t1-t4 are completely random tokens *but* they are repeated.

Requirements
------------

The installation requirements are the following:

* torch
* pytorch-fast-transformers

They can be installed in most systems via

    pip install torch pytorch-fast-transformers

Running the code
----------------

Run following commands to reproduce results in Figure 1 in our paper.

```
python main.py --save_to "./softmax" --attention_type full --epochs 100 --manualSeed 0 --gpu-id 0
python main.py --save_to "./reformer" --attention_type reformer --epochs 100 --manualSeed 0 --gpu-id 0
python main.py --save_to "./linear" --attention_type causal-linear --epochs 100 --manualSeed 0 --gpu-id 0
python main_causal_momentum.py --save_to "./momentum" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --epochs 100 --manualSeed 0 --gpu-id 0
python main_causal_res_momentum.py --save_to "./momentum_momentum_connection" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_mu 0.99 --res_stepsize 0.6 --epochs 100 --manualSeed 0 --gpu-id 0
python main_causal_adaptive_momentum.py --save_to "./adaptivemomentum" --attention_type causal-momentum --mu 0.1 --stepsize 0.6 --res_stepsize 0.99 --res_delta 0.0001 --adaptive_type "nc" --epochs 100 --manualSeed 0 --gpu-id 0
<<<<<<< HEAD
```
=======
```
>>>>>>> 350e213ba188a726221f9c9fcae1211ed4a6fb9d
