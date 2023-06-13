import torch.nn as nn


class EncoderDecorder:
    """Class combining the encorder and the decorder"""
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
    """Contains the last layer outputed to concatinate the decorder output
    Args:
        d_model (_type_): The instantiated model
    """
    def __init__(self, d_model, vocab) -> None:
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return nn.log_softmax(self.proj(x), dim=1)