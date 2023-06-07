import os
from os.path import exists
import torch
import torch.nn as nn
from torch.nn.fundtional import log_softmax, pad
import math
import copy
import time
import copy
from torch.optim.lr_schedular import LamdaLR
import pandas as pd
import altair as alt
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    
class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EncoderDecorder:
    def __init__(self, encoder, decorder, src_embed, tgt_embed, generator) -> None:
        super(EncoderDecorder, self).__init__
        self.encoder = encoder
        self.decorder = decorder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        encode = self.encode(src, src_mask)
        return self.decode(encode, src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        embed_src = self.src_embed(src)
        return self.encoder(embed_src, src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        embed_tgt = self.tgt_embed(tgt)
        return self.decorder(embed_tgt, memory, src_mask, tgt_mask)
    

class Generator(nn.Module):
    def __init__(self, d_model, vocab) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=1)