import math

import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset
from transformers import AutoModel, AutoProcessor, AutoTokenizer

import utils

CLIP = "openai/clip-vit-base-patch32"
VIT = "google/vit-base-patch16-224-in21k"
IMAGES_PATH = "data/image_features.pt"

import numpy as np


class Flickr30kDataset(Dataset):
    def __init__(self, split="train"):
        _, unique_images, rows = utils.get_captions()

        # Split the images (not the individual caption rows) into train/val/test
        np.random.seed(42)  # reproducible splits
        np.random.shuffle(unique_images)

        n_images = len(unique_images)
        train_end = int(0.8 * n_images)
        val_end = int(0.9 * n_images)
        test_end = int(1.0 * n_images)

        if split == "train":
            split_images = set(unique_images[:train_end])
        elif split == "val":
            split_images = set(unique_images[train_end:val_end])
        else:  # "test"
            split_images = set(unique_images[val_end:test_end])

        # Keep only caption rows whose image belongs to the chosen split
        self.captions = [row for row in rows if row["image"] in split_images]

        self.tokenizer = utils.TOKENIZER
        self.image_features = torch.load(IMAGES_PATH, weights_only=False)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions[idx]
        img_filename = row["image"]
        image = torch.tensor(self.image_features[img_filename])
        caption = row["caption"]
        # Pre‑tokenize caption once (LongTensor [L])
        input_ids = self.tokenizer(
            caption,
            max_length=30,
            add_special_tokens=False,
            truncation=True,
        ).input_ids
        input_ids = (
            [self.tokenizer.bos_token_id] + input_ids + [self.tokenizer.eos_token_id]
        )
        input_ids.extend([self.tokenizer.pad_token_id] * (32 - len(input_ids)))
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        return image, input_tensor


def attention(k_dim, q, k, v, mask_tensor):
    kt = k.transpose(-2, -1)
    # do attention(Q, K, V) = softmax(Q·K^T / sqrt(dims))·V to get hidden state (where · is dot product)
    attn_dot_product = torch.matmul(q, kt)
    attn_scaled = attn_dot_product / math.sqrt(k_dim)
    if mask_tensor is not None:
        attn_scaled = attn_scaled.masked_fill(mask_tensor, -torch.inf)
    attn_probs = torch.softmax(attn_scaled, dim=-1)
    return torch.matmul(attn_probs, v)


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.k_dim = model_dim // num_heads
        self.wqkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.endmulti = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def rearrange(self, vector, B, L):
        return vector.reshape(B, L, self.num_heads, self.k_dim).transpose(1, 2)

    def forward(self, x, pad_mask):
        B, L, D = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(self.model_dim, dim=-1)
        qh = self.rearrange(q, B, L)
        kh = self.rearrange(k, B, L)
        vh = self.rearrange(v, B, L)

        mask_tensor = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        # Expand to 1×1×L×L so it can broadcast with per‑batch pad masks and per‑head scores
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)
        mask_tensor = mask_tensor | pad_mask

        attended = attention(self.k_dim, qh, kh, vh, mask_tensor=mask_tensor)
        concatted = attended.transpose(1, 2).reshape(B, L, self.model_dim)
        concatted = self.dropout(concatted)
        return self.endmulti(concatted)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(model_dim, ffn_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim, bias=True),
        )

    def forward(self, x):
        return self.sequence(x)


class Decoder(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.masked_self_mha = SelfAttention(model_dim=model_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim=model_dim, ffn_dim=ffn_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data, pad_mask):
        stage1 = self.masked_self_mha(data, pad_mask)
        addnormed_text = self.norm1(data + self.dropout(stage1))
        ffned = self.ffn(addnormed_text)
        return self.norm2(addnormed_text + self.dropout(ffned))


class VitEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = utils.get_device()
        self.vit_processor = AutoProcessor.from_pretrained(VIT, use_fast=False)
        self.vit_model = AutoModel.from_pretrained(VIT, use_safetensors=True).to(
            self.device
        )

        # Freeze the pre-trained weights
        for param in self.vit_model.parameters():
            param.requires_grad = False

    def forward(self, images):
        inputs = self.vit_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vit_model(**inputs).last_hidden_state.mean(dim=1)
        return outputs


class CombinedTransformer(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_decoders: int,
        dropout: float,
    ):
        super().__init__()

        self.vit_encoder = VitEncoder()  # inference only

        self.tokenizer = utils.TOKENIZER

        actual_vocab_size = max(self.tokenizer.get_vocab().values()) + 1
        self.token_embedding = nn.Embedding(actual_vocab_size, model_dim)

        self.image_projection = nn.Linear(768, model_dim)
        self.linear = nn.Linear(model_dim, actual_vocab_size)

        def make_decoder() -> nn.Module:
            return Decoder(
                model_dim=model_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        self.decoder_series = nn.ModuleList(
            [make_decoder() for _ in range(num_decoders)]
        )

    # for one step
    def decode_step(self, image_features, input_ids, pad_mask):
        # Create embeddings for the tokens generated so far
        tok_embed = self.token_embedding(input_ids)

        # Combine image and text embeddings
        decoder_input = torch.cat(
            [image_features.unsqueeze(1), tok_embed],
            dim=1,
        )

        # Pass through all decoder layers
        for decoder in self.decoder_series:
            decoder_input = decoder(decoder_input, pad_mask)

        # Get logits for only the very last token in the sequence
        logits = self.linear(decoder_input[:, -1, :])
        return logits

    def forward(self, images, input_ids, pad_mask):
        device = next(self.parameters()).device
        input_ids = input_ids.to(device)  # [B, L]
        tok_embed = self.token_embedding(input_ids)  # [B, L, D]

        # Encode image
        img_encoded = images.to(device)
        img_embed = self.image_projection(img_encoded).unsqueeze(1)  # [B, 1, D]

        # Prepend image embedding to caption embeddings
        decoder_input = torch.cat([img_embed, tok_embed[:, :-1, :]], dim=1)  # [B, L, D]
        pad_mask = torch.cat(
            [
                torch.zeros_like(pad_mask[:, :1]),  # image vector is never pad
                pad_mask[:, :-1],
            ],  # align rest of caption
            dim=1,
        )
        for decoder in self.decoder_series:
            decoder_input = decoder(decoder_input, pad_mask)

        return self.linear(decoder_input[:, 1:, :])  # [B, L-1, vocab]


# for inference
def encode_image(image, vit_encoder, transformer):
    image_features = vit_encoder([image])
    return transformer.image_projection(image_features)  # [B, model_dim]
