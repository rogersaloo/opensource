import torch
import math
import torch.nn as nn
from torch.nn.functional import log_softmax, pad
import copy
import copy
import warnings
# Set to False to skip notebook execution (e.g. for debugging)
warnings.filterwarnings("ignore")
RUN_EXAMPLES = True


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    

class LayerNorm(nn.Module):
    """Create a layer normalization used in the tranfomer encoder and decorder

    Args:
        features (_type_): number of features default: 512 
    """
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    """Residual connection for the ecoder decorder bypass

    Args:
        dropout  (_type_): size of the dropout layer needed
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

def attention(query, key, value, mask=None, dropout=None):
    """Compute scaled dot matrix attention
    The entry contains three values the query, key and value

    Args:
        query (_type_): _description_
        key (_type_): _description_
        value (_type_): _description_
        mask (_type_, optional): _description_. Defaults to None.
        dropout (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.module):
    def __init__(self, h, d_model, dropout=0.1);
        """Input model skize and the number of heads
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # we assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model))
        self.attn = None 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            #same mask applied to all heads
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        #1) Do all the linear projections 
        query, key, value = [lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
                             for lin, x in zip(self.linears, (query, key, value))]
        
        #2) Apply attention on all the projeted vectors
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        #3) Concatenate using view and apply a final linear layer
        x = (x.transpose(1, 2).contigous().view(nbatches, -1, self.h * self.d_k))
        del query
        del key
        del value
        return self.linear[-1](x)


