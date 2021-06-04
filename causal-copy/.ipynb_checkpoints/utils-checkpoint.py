#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import time
from collections import namedtuple
import sys

import numpy as np
import torch
import torch.nn.functional as F

from radam import RAdam


def add_optimizer_arguments(parser):
    parser.add_argument(
        "--optimizer",
        choices=["radam", "adam"],
        default="radam",
        help="Choose the optimizer"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Set the learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Set the weight decay"
    )

def get_optimizer(params, args):
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    elif args.optimizer == "radam":
        return RAdam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError("Optimizer {} not available".format(args.optimizer))


def add_transformer_arguments(parser):
    parser.add_argument(
        "--attention_type",
        type=str,
        choices=["full", "causal-linear", "reformer", "momentum-linear", "adam-linear", "adamr-linear", "normgrad-linear", "frmomentum-linear", "frmomentum-linear-v2","frmomentum-linear-v3","nesterov-linear", "lsmomentum-linear", "adamax-linear", "adamclip-linear","fradamax-linear","fradamax-linear-v2","causal-momentum","causal-ncmomentum","causal-ncmomentum-v2","causal-ncmomentum-v3","causal-ncmomentum-py", "sparse-full", "causal-rank-k"],
        default="causal-linear",
        help="Attention model to be used"
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Number of self-attention layers"
    )
    parser.add_argument(
        "--n_heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--d_query",
        type=int,
        default=32,
        help="Dimension of the query, key, and value embedding"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout to be used for transformer layers"
    )
    parser.add_argument(
        "--softmax_temp",
        type=float,
        default=None,
        help=("Softmax temperature to be used for training "
              "(default: 1/sqrt(d_query))")
    )
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=0.1,
        help="Dropout to be used for attention layers"
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=32,
        help="Number of planes to use for hashing for reformer"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Number of queries in each block for reformer"
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=4,
        help="Number of rounds of hashing for reformer"
    )
    parser.add_argument(
        "--unmasked_reformer",
        action="store_false",
        dest="masked",
        help="If set the query can attend to itsself for reformer"
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.0,
        help="Momentum to be used for momentum transformer layers"
    )
    parser.add_argument(
        "--stepsize",
        type=float,
        default=1.0,
        help="Stepsize to be used for momentum transformer layers"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="Beta scalar to be used for scaling the gradients in adam transformer layers"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0001,
        help="Delta scalar to be used for scaling the gradients in fradam transformer layers"
    )
    parser.add_argument(
        "--res_stepsize",
        type=float,
        default=1.0,
        help="Stepsize to be used for momentum residual connections"
    )
    parser.add_argument(
        "--res_mu",
        type=float,
        default=0.0,
        help="Momentum to be used for momentum residual connections"
    )
    parser.add_argument(
        "--res_delta",
        type=float,
        default=0.0001,
        help="Delta to be used for momentum residual connections"
    )
    parser.add_argument(
        "--adaptive_type",
        type=str,
        choices=["wang", "fr", "pr", "hs", "dy"],
        default="wang",
        help="Adaptive momentum type to be used for momentum residual connections"
    )
    parser.add_argument(
        "--is_resw",
        type=eval,
        choices=[True, False],
        default=False,
        help="Use additional weights for momentum update in the momentum residual connections or not"
    )
    parser.add_argument(
        "--diag_size",
        type=int,
        default=0,
        help="Size of diagonal vector in sparse transformers"
    )
    parser.add_argument(
        "--blend",
        type=float,
        default=0.9,
        help="Blending of sparse and linear transformers"
    )
    parser.add_argument(
        "--kernel_1",
        type=str,
        choices=['tanh', 'elu', 'relu', 'celu', 'sigmoid', 'leakyrelu', 'softplus'],
        default='tanh',
        help="The first kernel used in the rank kernel transformer"
    )
    parser.add_argument(
        "--kernel_2",
        type=str,
        choices=['tanh', 'elu', 'relu', 'celu', 'sigmoid', 'leakyrelu', 'softplus', 'none'],
        default='none',
        help="The second kernel used in the rank kernel transformer"
    )

    return parser


def print_transformer_arguments(args):
    print((
        "Transformer Config:\n"
        "    Attention type: {attention_type}\n"
        "    Number of layers: {n_layers}\n"
        "    Number of heads: {n_heads}\n"
        "    Key/Query/Value dimension: {d_query}\n"
        "    Transformer layer dropout: {dropout}\n"
        "    Softmax temperature: {softmax_temp}\n"
        "    Attention dropout: {attention_dropout}\n"
        "    Number of hashing planes: {bits}\n"
        "    Chunk Size: {chunk_size}\n"
        "    Rounds: {rounds}\n"
        "    Masked: {masked}"
    ).format(**vars(args)))


class EpochStats(object):
    def __init__(self, metric_names=[], freq=1, out=sys.stdout):
        self._start = time.time()
        self._samples = 0
        self._loss = 0
        self._metrics = [0]*len(metric_names)
        self._metric_names = metric_names
        self._out = out
        self._freq = freq
        self._max_line = 0

    def update(self, n_samples, loss, metrics=[]):
        self._samples += n_samples
        self._loss += loss*n_samples
        for i, m in enumerate(metrics):
            self._metrics[i] += m*n_samples

    def _get_progress_text(self):
        time_per_sample = (time.time()-self._start) / self._samples
        loss = self._loss / self._samples
        metrics = [
            m/self._samples
            for m in self._metrics
        ]
        text = "Loss: {} ".format(loss)
        text += " ".join(
            "{}: {}".format(mn, m)
            for mn, m in zip(self._metric_names, metrics)
        )
        if self._out.isatty():
            to_add = " [{} sec/sample]".format(time_per_sample)
            if len(text) + len(to_add) > self._max_line:
                self._max_line = len(text) + len(to_add)
            text += " " * (self._max_line-len(text)-len(to_add)) + to_add
        else:
            text += " time: {}".format(time_per_sample)
        return text

    def progress(self):
        if self._samples < self._freq:
            return
        text = self._get_progress_text()
        if self._out.isatty():
            print("\r" + text, end="", file=self._out)
        else:
            print(text, file=self._out, flush=True)
        self._loss = 0
        self._samples = 0
        self._last_progress = 0
        for i in range(len(self._metrics)):
            self._metrics[i] = 0
        self._start = time.time()

    def finalize(self):
        self._freq = 1
        self.progress()
        if self._out.isatty():
            print("", file=self._out)


def load_model(saved_file, model, optimizer, device):
    data = torch.load(saved_file, map_location=device)
    model.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    epoch = data["epoch"]

    return epoch


def save_model(save_file, model, optimizer, epoch):
    torch.save(
        dict(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch
        ),
        save_file.format(epoch)
    )
    
class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False): 
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume: 
                self.file = open(fpath, 'r') 
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')  
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume: 
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
