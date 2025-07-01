import math

import torch
import torch.nn as nn
import os
import csv
import kagglehub
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import logging

CLIP = "openai/clip-vit-base-patch32"
VIT = "google/vit-base-patch16-224-in21k"
IMAGES_NPZ_PATH = "data/image_features.npz"

import numpy as np


class Flickr30kDataset(Dataset):
    def __init__(self, split="train"):
        data_dir = kagglehub.dataset_download("adityajn105/flickr30k")

        captions_path = os.path.join(data_dir, "captions.txt")

        # Load the caption records without pandas
        with open(captions_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(
                f, delimiter=","
            )  # assumes CSV with header "image,caption"
            all_captions = [row for row in reader]

        # Collect unique image filenames
        unique_images = list({row["image"] for row in all_captions})

        # Split the images (not the individual caption rows) into train/val/test
        np.random.seed(42)  # reproducible splits
        np.random.shuffle(unique_images)

        n_images = len(unique_images)
        train_end = int(0.08 * n_images)
        val_end = int(0.09 * n_images)
        test_end = int(0.1 * n_images)

        if split == "train":
            split_images = set(unique_images[:train_end])
        elif split == "val":
            split_images = set(unique_images[train_end:val_end])
        else:  # "test"
            split_images = set(unique_images[val_end:test_end])

        # Keep only caption rows whose image belongs to the chosen split
        self.captions = [row for row in all_captions if row["image"] in split_images]

        self.img_dir = os.path.join(data_dir, "Images")
        self.image_features = np.load(IMAGES_NPZ_PATH)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions[idx]
        img_filename = row["image"]
        image = torch.tensor(self.image_features[img_filename])
        caption = row["caption"]
        return image, caption


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
    def __init__(
        self, model_dim: int, num_heads: int, mask: bool, dropout: float = 0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.k_dim = model_dim // num_heads
        self.wqkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.endmulti = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.mask = mask

    def rearrange(self, vector, B, L):
        return vector.reshape(B, L, self.num_heads, self.k_dim).transpose(1, 2)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(self.model_dim, dim=-1)
        qh = self.rearrange(q, B, L)
        kh = self.rearrange(k, B, L)
        vh = self.rearrange(v, B, L)

        mask_tensor = None
        if self.mask:
            mask_tensor = torch.triu(
                torch.ones(L, L, device=x.device), diagonal=1
            ).bool()

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
        self.masked_self_mha = SelfAttention(
            model_dim=model_dim, num_heads=num_heads, mask=True
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim=model_dim, ffn_dim=ffn_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        stage1 = self.masked_self_mha(data)
        addnormed_text = self.norm1(data + self.dropout(stage1))
        ffned = self.ffn(addnormed_text)
        return self.norm2(addnormed_text + self.dropout(ffned))


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
        # Load pre-trained models

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # avoid errors
        self.tokenizer = tokenizer

        actual_vocab_size = max(self.tokenizer.get_vocab().values()) + 1
        self.token_embedding = nn.Embedding(actual_vocab_size, model_dim)

        self.vit_processor = AutoProcessor.from_pretrained(VIT, use_fast=False)
        self.vit_model = AutoModel.from_pretrained(VIT, use_safetensors=True)

        # Freeze the pre-trained weights
        for param in self.vit_model.parameters():
            param.requires_grad = False

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

    def forward(self, images, captions):
        device = next(self.parameters()).device
        tokenized = self.tokenizer(captions, padding=True, return_tensors="pt").to(device)
        input_ids = tokenized.input_ids  # [B, L]
        labels = input_ids[:, 1:]  # [B, L-1]
        # Embed captions
        tok_embed = self.token_embedding(input_ids)  # [B, L, D]

        # Encode image
        img_encoded = images.to(device)
        img_embed = self.image_projection(img_encoded).unsqueeze(1)  # [B, 1, D]

        # Prepend image embedding to caption embeddings
        decoder_input = torch.cat([img_embed, tok_embed[:, :-1, :]], dim=1)  # [B, L, D]

        for decoder in self.decoder_series:
            decoder_input = decoder(decoder_input)

        logits = self.linear(decoder_input[:, 1:, :])# [B, L-1, vocab]
        return logits, labels
