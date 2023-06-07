





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