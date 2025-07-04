import torch
import torch.nn as nn

class TransformerTextDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers=4, nhead=8, dim_feedforward=512, max_len=128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        self.output_proj = nn.Linear(embed_dim, vocab_size)



    def forward(self, tgt_ids, memory):
        """
        tgt_ids: [B, T]           → input token IDs (e.g. <start>, word1, ...)
        memory:  [B, 1, D]        → image embedding from CLIP (CLS token)
        returns: [B, T, vocab_size]
        """
        B, T = tgt_ids.size()
        device = tgt_ids.device
        pos = torch.arange(T, device=device).unsqueeze(0)

        tgt_embed = self.embed_tokens(tgt_ids) + self.pos_embed(pos)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        decoded = self.transformer_decoder(
            tgt=tgt_embed,
            memory=memory,
            tgt_mask=tgt_mask
        )

        return self.output_proj(decoded)  # [B, T, vocab_size]

