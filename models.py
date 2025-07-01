import math

import torch
import torch.nn as nn
import os
import csv
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoModel, AutoProcessor

CLIP = "openai/clip-vit-base-patch32"
VIT = "google/vit-base-patch16-224-in21k"

import numpy as np


class Flickr30kDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        self.data_dir = data_dir
        self.transform = transform

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
        train_end = int(0.8 * n_images)
        val_end = int(0.9 * n_images)

        if split == "train":
            split_images = set(unique_images[:train_end])
        elif split == "val":
            split_images = set(unique_images[train_end:val_end])
        else:  # "test"
            split_images = set(unique_images[val_end:])

        # Keep only caption rows whose image belongs to the chosen split
        self.captions = [row for row in all_captions if row["image"] in split_images]

        self.img_dir = os.path.join(data_dir, "Images")

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        row = self.captions[idx]
        img_filename = row["image"]
        img_path = os.path.join(self.img_dir, img_filename)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

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
        self.clip_processor = AutoProcessor.from_pretrained(CLIP, use_fast=False)
        self.clip_model = AutoModel.from_pretrained(CLIP, use_safetensors=True)
        self.vit_processor = AutoProcessor.from_pretrained(VIT, use_fast=False)
        self.vit_model = AutoModel.from_pretrained(VIT, use_safetensors=True)

        # Freeze the pre-trained weights
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.vit_model.parameters():
            param.requires_grad = False

        self.image_projection = nn.Linear(768, model_dim)
        self.text_projection = nn.Linear(512, model_dim)

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
        self.linear = nn.Linear(model_dim, model_dim)

    def encode_text(self, text):
        """Extract CLIP text features"""
        inputs = self.clip_processor(
            text=text, return_tensors="pt", padding=True, truncation=True
        )
        # may not need line below
        text_features = self.clip_model.get_text_features(**inputs)
        return text_features  # Shape: [batch_size, 512]

    def encode_image_vit(self, images):
        """Extract ViT features"""
        inputs = self.vit_processor(images=images, return_tensors="pt")
        outputs = self.vit_model(**inputs)
        return outputs.last_hidden_state  # Shape: [batch_size, 197, 768]

    def forward(self, images, texts):
        encoded_texts = self.encode_text(texts)
        encoded_images = self.encode_image_vit(images)
        encoded_images = encoded_images.mean(dim=1)

        embed_texts = self.text_projection(encoded_texts)
        embed_images = self.image_projection(encoded_images)

        # Add a sequence dimension (dim=1) to each tensor, changing shape from [B, D] to [B, 1, D]
        embed_texts_seq = embed_texts.unsqueeze(1)
        embed_images_seq = embed_images.unsqueeze(1)

        # Concatenate along the new sequence dimension (dim=1) to create a single tensor
        # of shape [B, 2, D] where 2 is the sequence length.
        combined = torch.concat([embed_texts_seq, embed_images_seq], dim=1)
        for decoder in self.decoder_series:
            combined = decoder(combined)
        return self.linear(combined)

