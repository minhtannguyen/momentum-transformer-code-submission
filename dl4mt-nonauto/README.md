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
The original translation corpora can be downloaded from ([IWLST'16 En-De](https://wit3.fbk.eu/). For the preprocessed corpora and pre-trained models, see below.

| | Dataset | Model |
| -------------      | --- | -------------  |
| IWSLT'16 En-De     | [Data](https://drive.google.com/file/d/1m7dZqEXHWPYcre6xxsFwFLrb9CRCZGmn/view?usp=sharing) | [Models](https://drive.google.com/open?id=1N8tfU5ttnov2jWk3-PHVMJClQA0pKXoN) |

Before you run the code
------------------
Set correct path to data in `data_path()` function located in [`data.py`](https://github.com/jasonleeinf/non-auto-decoding/blob/96f7765399133c79ad4d23768dd530ee3eb07990/data.py#L44):

Loading & Decoding from Pre-trained Models
------------------
1. For `vocab_size`, use `40000` for IWLST'16 En-De.
2. For `params`, use `small` for for IWLST'16 En-De.

#### Non-autoregressive
```bash
$ python run.py --dataset <dataset> --vocab_size <vocab_size> --ffw_block highway --params <params> --lr_schedule anneal --fast --valid_repeat_dec 20 --use_argmax --next_dec_input both --mode test --remove_repeats --debug --trg_len_option predict --use_predicted_trg_len --load_from <checkpoint>
```

For adaptive decoding, add the flag `--adaptive_decoding jaccard` to the above.

Training New Models
------------------

#### Non-autoregressive
```bash
$ python run.py --dataset <dataset> --vocab_size <vocab_size> --ffw_block highway --params <params> --lr_schedule anneal --fast --valid_repeat_dec 8 --use_argmax --next_dec_input both --denoising_prob --layerwise_denoising_weight --use_distillation
```

