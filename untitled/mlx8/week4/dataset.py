
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
import pandas as pd
import os
from torch.utils.data import Dataset
import json
import torch

IMAGE_DIR = './mlx8/week4/flickr30k-images'
CAPTION_CSV = './mlx8/week4/flickr_annotations_30k.csv'
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if tokenizer.bos_token is None:
    tokenizer.add_special_tokens({'bos_token': '<s>'})

if tokenizer.eos_token is None:
    tokenizer.add_special_tokens({'eos_token': '</s>'})

def image_to_patches(image, patch_size=16):
    image = transforms.Resize((224, 224))(image)
    image = transforms.ToTensor()(image)
    c, h, w = image.shape
    assert h % patch_size == 0 and w % patch_size == 0

    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4)
    patches = patches.reshape(-1, c * patch_size * patch_size)
    return patches

class CaptionDataset(Dataset):
    def __init__(self, csv_path, imge_dir, tokenizer, max_len=30, patch_size=16, limit=100):
        self.df = pd.read_csv(csv_path)[:limit]
        self.image_dir = imge_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.patch_size = patch_size
        self.data = []
        self._prepare()


    def _prepare(self):
        for _, row in self.df.iterrows():
            img_path = os.path.join(self.image_dir, row['filename'])
            if not os.path.exists(img_path): continue
            image = Image.open(img_path).convert('RGB')
            patches = image_to_patches(image, self.patch_size)
            captions = json.loads(row['raw'])
            for caption in captions:
                tokens = tokenizer.encode(caption, add_special_tokens=True, max_length=self.max_len, truncation=True)
                input_ids = tokens[:-1]
                label_ids = tokens[1:]
                self.data.append((patches, input_ids, label_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patches, input_ids, label_ids = self.data[idx]
        return {'patches': torch.tensor(patches, dtype=torch.float),
                'decoder_input_ids': torch.tensor(input_ids, dtype=torch.long),
                'decoder_label_ids': torch.tensor(label_ids, dtype=torch.long)}


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    return {'patches': torch.stack([x['patches'] for x in batch]),
                'decoder_input_ids': pad_sequence([x['decoder_input_ids'] for x in batch], batch_first=True, padding_value=tokenizer.pad_token_id),
                'decoder_label_ids': pad_sequence([x['decoder_label_ids'] for x in batch], batch_first=True, padding_value=-100)}




















