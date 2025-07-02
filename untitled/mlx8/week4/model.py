
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_dim, embed_dim, num_patches):
        super().__init__()
        self.linear = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x):
        x = self.linear(x)
        return x + self.pos_embed

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
                                      nn.ReLU(),
                                      nn.Linear(embed_dim * 4, embed_dim))
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.ln1(x + attn_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.cross_attn = torch.nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
                                      nn.ReLU(),
                                      nn.Linear(embed_dim * 4, embed_dim))
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ln3 = nn.LayerNorm(embed_dim)

    def forward(self, tgt, memory, tgt_mask=None):
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.ln1(tgt + attn_out)
        cross_out, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.ln2(tgt + cross_out)
        ff_out = self.ff(tgt)
        tgt = self.ln3(tgt + ff_out)

        return tgt


class CaptionTransformer(nn.Module):
    def __init__(self, patch_dim, embed_dim, num_patches, vocab_size, num_enc_layers=4, num_dec_layers=4, n_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_dim, embed_dim, num_patches)
        self.encoder = nn.ModuleList(TransformerEncoderBlock(embed_dim, n_heads) for _ in range(num_enc_layers))
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.decoder = nn.ModuleList(TransformerDecoderBlock(embed_dim, n_heads) for _ in range(num_dec_layers))
        self.pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim))
        self.out_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, patches, decoder_input_ids):
        x = self.patch_embed(patches)
        for layer in self.encoder:
            x = layer(x)

        tgt = self.token_embed(decoder_input_ids) + self.pos_embed[:, :decoder_input_ids.size(1), :]
        tgt_mask = torch.triu(torch.ones(tgt.size(1), tgt.size(1), device=tgt.device), diagonal=1).bool()

        for layer in self.decoder:
            tgt = layer(tgt, x, tgt_mask=tgt_mask)
        return self.out_proj(tgt)















