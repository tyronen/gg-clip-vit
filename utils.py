import logging
from contextlib import nullcontext

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

MODEL_FILE = "data/models.pth"

TOKENIZER = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True
)
TOKENIZER.bos_token = "<|im_start|>"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def amp_components(device, train=False):
    if device.type == "cuda" and train:
        return autocast(), GradScaler()
    else:
        # fall-back: no automatic casting, dummy scaler
        return nullcontext(), GradScaler(enabled=False)


def collate_fn(batch):
    images, input_ids = zip(*batch)
    images = torch.stack(images)  # [B, 768]
    input_ids = torch.stack(input_ids)  # [B, L]
    pad_mask = input_ids == TOKENIZER.pad_token_id
    return {"images": images, "input_ids": input_ids, "pad_mask": pad_mask}


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, device, batch_size, train=False):
        num_workers = 8 if device.type == "cuda" else 0 if device.type == "mps" else 4
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            pin_memory=(device.type == "cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )
