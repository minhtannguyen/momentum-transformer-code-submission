#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

import argparse
import math
import sys
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader, IterableDataset

from utils import add_optimizer_arguments, get_optimizer, \
    add_transformer_arguments, print_transformer_arguments, \
    EpochStats, load_model, save_model, Logger

sys.path.append('../')
from fast_transformers.masking import LengthMask, TriangularCausalMask
from fast_transformers.builders import TransformerEncoderBuilder

# My new import
import os
import random
import fcntl
import time

from tensorboardX import SummaryWriter

class CopyTask(IterableDataset):
    def __init__(self, max_sequence, n_classes):
        self._max_sequence = max_sequence
        self._n_classes = n_classes
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Make some local copies
        max_seq = self._max_sequence
        n_classes = self._n_classes

        # Generate the random sequence
        n = torch.randint(max_seq//4, (max_seq-1)//2, tuple())
        random_sequence = (torch.rand(n)*n_classes).long() + 1

        # Generate the input, target and loss mask
        x = torch.zeros(max_seq, dtype=torch.long)
        y = torch.zeros(max_seq, dtype=torch.long)
        mask = torch.zeros(max_seq)
        x[:n] = random_sequence
        x[n+1:2*n+1] = random_sequence
        y[:-1] = x[1:]
        mask[n-1:2*n] = 1

        return x, y, mask


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos_embedding =  self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x =  torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)


class SequencePredictor(torch.nn.Module):
    def __init__(self, d_model, sequence_length, n_classes,
                 attention_type="full", n_layers=4, n_heads=4,
                 d_query=32, dropout=0.1, softmax_temp=None,
                 attention_dropout=0.1,
                 bits=32, rounds=4,
                 chunk_size=32, masked=True):
        super(SequencePredictor, self).__init__()

        self.pos_embedding = PositionalEncoding(
            d_model//2,
            max_len=sequence_length
        )
        self.value_embedding = torch.nn.Embedding(
            n_classes+1,
            d_model//2
        )
        self.builder_dict = OrderedDict({
            "attention_type": attention_type,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "feed_forward_dimensions": n_heads*d_query*4,
            "query_dimensions": d_query,
            "value_dimensions": d_query,
            "dropout": dropout,
            "softmax_temp": softmax_temp,
            "attention_dropout": attention_dropout,
            "bits": bits,
            "rounds": rounds,
            "chunk_size": chunk_size,
            "masked": masked
        })

        self.transformer = TransformerEncoderBuilder.from_dictionary(
            self.builder_dict,
            strict=True
        ).get()

        hidden_size = n_heads*d_query
        self.predictor = torch.nn.Linear(
            hidden_size,
            n_classes+1
        )

    def forward(self, x):
        # x: [128, 64]
        x = x.view(x.shape[0], -1) # x: [128, 64]
        x = self.value_embedding(x).transpose(1, 0) # x: [64, 128, 128]
        x = self.pos_embedding(x) # x: [64, 128, 256]
        triangular_mask = TriangularCausalMask(x.shape[1], device=x.device) # triangular_mask: [128, 128]
        y_hat = self.transformer(x, attn_mask=triangular_mask) # y_hat: [64, 128, 256]
        y_hat = self.predictor(y_hat) # y_hat: [64, 128, 11]

        return y_hat


def loss(y, y_hat, loss_mask):
    y_hat = y_hat.transpose(1, 0).contiguous()
    L, N, C = y_hat.shape
    l = torch.nn.functional.cross_entropy(
        y_hat.view(L*N, C),
        y.contiguous().view(L*N),
        reduction="none"
    ).view(L, N)
    # this means longer sequences have higher weight but it sounds ok
    l = (loss_mask * l).mean() / loss_mask.mean()
    accuracy = ((y == y_hat.argmax(dim=-1)).float() * loss_mask).mean() / loss_mask.mean()

    return l, accuracy.item()

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    total_samples = 0
    with torch.no_grad():
        for i, (x, y, m) in zip(range(20), dataloader):
            x = x.to(device).t()
            y = y.to(device).t()
            m = m.to(device).t()
            y_hat = model(x)
            l, acc = loss(y, y_hat, m)
            total_loss += x.shape[1] * l.item()
            total_acc += x.shape[1] * acc
            total_samples += x.shape[1]
            
    test_loss = total_loss/total_samples
    test_acc = total_acc/total_samples
    print(
        "Testing =>",
        "Loss:",
        test_loss,
        "Accuracy:",
        test_acc
    )

    return test_loss, test_acc


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a transformer for a copy task"
    )

    add_optimizer_arguments(parser)
    add_transformer_arguments(parser)

    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Set the maximum sequence length"
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=10,
        help="Set the number of classes"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="How many epochs to train for"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="How many samples to use together"
    )
    parser.add_argument(
        "--reduce_lr_at",
        type=int,
        default=30,
        help="At this epoch divide the lr by 10"
    )

    parser.add_argument(
        "--save_to",
        default=None,
        help="Set a file to save the models to."
    )
    parser.add_argument(
        "--continue_from",
        default=None,
        help="Load the model from a file"
    )
    parser.add_argument(
        "--save_frequency",
        default=1,
        type=int,
        help="Save every that many epochs"
    )
    # My new argument
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
    
    

    args = parser.parse_args(argv)
    print_transformer_arguments(args)
    
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = torch.cuda.is_available()
    
    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    # Make the dataset and the model
    test_set = CopyTask(args.sequence_length, args.n_classes)
    model = SequencePredictor(
        args.d_query*args.n_heads, args.sequence_length, args.n_classes,
        attention_type=args.attention_type,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_query=args.d_query,
        dropout=args.dropout,
        softmax_temp=None,
        attention_dropout=args.attention_dropout,
        bits=args.bits,
        rounds=args.rounds,
        chunk_size=args.chunk_size,
        masked=args.masked
    )

    # Choose a device and move everything there
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Running on {}".format(device))
    model.to(device)
    # Start training
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        pin_memory=device=="cuda"
    )
    optimizer = get_optimizer(model.parameters(), args)
    start_epoch = 1
    if args.continue_from:
        start_epoch = load_model(
            args.continue_from,
            model,
            optimizer,
            device
        )
    start_time = time.time()
    test_loss, test_acc = evaluate(model, test_loader, device)
    total_time = time.time() - start_time
    
    print("test_loss = %f"%test_loss)
    print("test_acc = %f"%test_acc)
    print("test_time = %f"%total_time)

if __name__ == "__main__":
    main()
