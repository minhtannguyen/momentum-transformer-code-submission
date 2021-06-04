## This is the code to reproduce machine translation experiments in our paper.

Dependencies
------------------
### Python
* Python 3.6
* PyTorch 0.4
* Numpy
* NLTK
* torchtext ( need to install the multigpu version at https://github.com/mansimov/pytorch_text_multigpu )
* torchvision

### GPU
* CUDA (we recommend using the latest version. The version 8.0 was used in all our experiments.)

### Related code
* For preprocessing, we used the scripts from [Moses](https://github.com/moses-smt/mosesdecoder "Moses") and [Subword-NMT](https://github.com/rsennrich/subword-nmt "Subword-NMT").
* This code is based on [NA-NMT](https://github.com/MultiPath/NA-NMT "NA-NMT").

Downloading Datasets & Pre-trained Models
------------------
The original translation corpora can be downloaded from ([IWLST'16 En-De](https://wit3.fbk.eu/). For the preprocessed corpora, see below.

| | Dataset | Model |
| -------------      | --- | -------------  |
| IWSLT'16 En-De     | [Data](https://drive.google.com/file/d/1m7dZqEXHWPYcre6xxsFwFLrb9CRCZGmn/view?usp=sharing) |

Before you run the code
------------------
Set correct path to data in `data_path()` function located in [`data.py`](https://github.com/jasonleeinf/non-auto-decoding/blob/96f7765399133c79ad4d23768dd530ee3eb07990/data.py#L44):

Training New Models
------------------
Run following commands to reproduce results in Table 3 in our paper.

```
CUDA_VISIBLE_DEVICES=0,1 python run.py --dataset 'iwslt-ende' --data_root {data_dir} --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'full' --maximum_steps 750000 --main_path './softmax' --seed 0 --num_gpus 2 --gpu 0
CUDA_VISIBLE_DEVICES=0,1 python run.py --dataset 'iwslt-ende' --data_root {data_dir} --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'linear' --maximum_steps 750000 --main_path './linear' --seed 0 --num_gpus 2 --gpu 0
CUDA_VISIBLE_DEVICES=0,1 python run.py --dataset 'iwslt-ende' --data_root {data_dir} --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --maximum_steps 750000 --main_path './momentum' --seed 0 --num_gpus 2 --gpu 0 
CUDA_VISIBLE_DEVICES=0,1 python run.py --dataset 'iwslt-ende' --data_root {data_dir} --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --res_type 'momentum' --adaptive_type 'nc' --res_mu 0.3 --res_stepsize 0.9 --res_delta 0.0001 --maximum_steps 750000 --main_path './momentum_momentum_connection' --seed 0 --num_gpus 2 --gpu 0
CUDA_VISIBLE_DEVICES=0,1 python run.py --dataset 'iwslt-ende' --data_root {data_dir} --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --load_dataset --use_distillation --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --res_type 'adaptive' --adaptive_type 'nc' --res_stepsize 0.9 --res_delta 0.0001 --maximum_steps 750000 --main_path './adaptivemomentum' --seed 0 --num_gpus 2 --gpu 0
```

Loading & Decoding from Pre-trained Models and Test Set
------------------
```
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --mode test --test_which 'test' --remove_repeats --debug --load_dataset --use_distillation --trg_len_option predict --use_predicted_trg_len --attn_type 'full' --model_path {model_dir} --load_from {model_name} --gpu 0
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --mode test --test_which 'test' --remove_repeats --debug --load_dataset --use_distillation --trg_len_option predict --use_predicted_trg_len --attn_type 'linear' --model_path {model_dir} --load_from {model_name} --gpu 0
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --mode test --test_which 'test' --remove_repeats --debug --load_dataset --use_distillation --trg_len_option predict --use_predicted_trg_len --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --model_path {model_dir} --load_from {model_name} --gpu 0
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --mode test --test_which 'test' --remove_repeats --debug --load_dataset --use_distillation --trg_len_option predict --use_predicted_trg_len --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --res_type 'momentum' --adaptive_type 'nc' --res_mu 0.3 --res_stepsize 0.9 --res_delta 0.0001 --model_path {model_dir} --load_from {model_name} --gpu 0
CUDA_VISIBLE_DEVICES=0 python run.py --dataset 'iwslt-ende' --vocab_size 40000 --ffw_block highway --params 'small' --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --layerwise_denoising_weight --mode test --test_which 'test' --remove_repeats --debug --load_dataset --use_distillation --trg_len_option predict --use_predicted_trg_len --attn_type 'momentum' --mu 0.6 --stepsize 0.6 --res_type 'adaptive' --adaptive_type 'nc' --res_stepsize 0.9 --res_delta 0.0001 --model_path {model_dir} --load_from {model_name} --gpu 0

```

For adaptive decoding, add the flag `--adaptive_decoding jaccard` to the above.
