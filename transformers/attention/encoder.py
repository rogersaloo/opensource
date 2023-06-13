import torch
import torch.nn as nn
from torch.nn.fundtional import log_softmax, pad
import copy
import copy
import warnings
from common import LayerNorm, clones, SublayerConnection
# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


    
class Encoder(nn.Module):
    """class creating the generic encorder

    Args:
        N (_type_): the depth of the attention decorder
        layer: implementation of an individual attention layer
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)



class EncoderLayer(nn.Module):
    """Encorder layer with a self attention, feedforward and dropout

    Args:
        size (_type_): size of the input embedding
        self_attention: Implementation of the self attention module 
        feed_forward: feed forward network ro concatinate the encoder attention output
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout),2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
