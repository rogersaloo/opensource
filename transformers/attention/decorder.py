import torch
import torch.nn as nn
from torch.nn.fundtional import log_softmax, pad
from common import clones, LayerNorm, SublayerConnection


class Decorder(nn.Module):
    """Generic N layer decorder with a masking

    Args:
        x (Tensors): input tensors from the target
        N (int): Number of layer within the deorder
    ----
    forward args
    ----
        memory (Tensors): The input from the encorder
        src_mask: the masking from the encorder
        tgt_mask: the mask from the decorder
    """
    def __init__(self, layer, N):
        super(Decorder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    

class DecorderLayer(nn.Module):
    """Decorder made of self attention, source attention and feed forward

    Args:
        size (int): size of the layer, default=512
        self_attn : attention on the inputs from the target decorder
        src_attn: attention from the decorder output
        feed_forward: The network containing the feed forward network and the layer normalization
        dropout: Dropout on the feed forward layer

    ----
    forward args
    ----
        memory (Tensors): The input from the encorder
        src_mask: the masking from the encorder
        tgt_mask: the mask from the decorder
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecorderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.src_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)4
    

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0