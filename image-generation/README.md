Image Generation with Transformers
==================================

In this task, the transformer learns to model an image one byte at a time.
Namely, given a sequence of bytes,

    r1 g1 b1 r2 g2 b2 ... rn gn

the transformer must predict the next value, in this case bn.

Requirements
------------

The installation requirements to run the training script (`main.py`) are 

* torch
* torchvision
* pytorch-fast-transformers

They can be installed in most systems via

    pip install torch torchvision pytorch-fast-transformers

However, in order to run the image generation script (`prediction.py`), you
need also

* matplotlib
* imageio

Running the code
----------------

### Training

MNIST: Run following commands to reproduce results in Table 1 and Figure 2 in our paper.
```
python main.py --dataset mnist --attention_type full --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_to "./mnist_softmax" --manualSeed 0 --gpu-id 0
python main.py --dataset mnist --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_to "./mnist_linear" --manualSeed 0 --gpu-id 0
python main_causal_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "./momentum" --mu 0.6 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 0
python main_causal_res_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "./momentum_momentum_connection" --mu 0.6 --stepsize 0.9 --delta 0.0001 --res_mu 0.1 --res_stepsize 0.99  --manualSeed 0 --gpu-id 0
python main_causal_adaptive_momentum.py --dataset mnist --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 --batch_size 10 --iterations 1500000 --evaluate_frequency 6000 --save_frequency 6000 --save_to "./adaptivemomentum" --mu 0.6 --stepsize 0.9 --res_stepsize 0.99 --delta 0.0001 --adaptive_type "nc" --is_resw False  --manualSeed 0 --gpu-id 0
```

CIFAR10: Run following commands to reproduce results in Table 2 in our paper.
```
python main.py --dataset cifar10 --attention_type full --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 1 --iterations 5000000 --evaluate_frequency 50000 --save_frequency 50000 --save_to "./cifar_softmax" --manualSeed 0 --gpu-id 0
python main.py --dataset cifar10 --attention_type causal-linear --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "./cifar_linear" --manualSeed 0 --gpu-id 0
python main_causal_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "./cifar_momentum" --mu 0.1 --stepsize 0.9 --delta 0.0001 --manualSeed 0 --gpu-id 0
python main_causal_res_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "./cifar_momentum_momentum_connection" --mu 0.1 --stepsize 0.9 --res_mu 0.1 --res_stepsize 0.9 --delta 0.0001 --is_resw False --manualSeed 0 --gpu-id 0
python main_causal_adaptive_momentum.py --dataset cifar10 --attention_type causal-momentum --d_query 32 --n_heads 8 --n_layers 16 --mixtures 10 --lr 2e-4 --batch_size 4 --iterations 1250000 --evaluate_frequency 12500 --save_frequency 12500 --save_to "./cifar_adaptivemomentum" --mu 0.1 --stepsize 0.9 --res_stepsize 0.9 --delta 0.0001 --adaptive_type "nc" --is_resw False  --manualSeed 0 --gpu-id 0 
```

### Prediction

After training your model or downloading a model you can generate images using
the `prediction.py` script.

The following code generates images from MNIST after training for a few epochs.

    python prediction.py --dataset mnist --attention_type causal-linear \
        --d_query 32 --n_heads 8 --n_layers 8 --mixtures 10 \
        --index 0-10 --offset 1 --recurrent --force_cpu \
        --save_image /path/to/images/mnist.{}.{}.pth \
        /path/to/weights.XX.pth

The following arguments affect the generation process

1. `--index` Selects the images from the test set that will be used to
   condition the generation on
2. `--offset` Selects how many pixels to condition on (setting it to 1 amounts
   to unconditional generation)
3. `--recurrent` Chooses to use a recurrent model which is optimised for
   inference, instead of a batch model which is optimised for training
