import os
import csv
import kagglehub
from tqdm import tqdm
import models
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor
import utils

from PIL import Image


class ViTFeatureDataset(Dataset):
    def __init__(self, image_filenames, image_dir):
        self.image_filenames = image_filenames
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]
        path = os.path.join(self.image_dir, filename)
        image = Image.open(path).convert("RGB")
        return filename, image


def collate_fn(batch):
    filenames, images = zip(*batch)
    return list(filenames), list(images)


def main():
    device = utils.get_device()

    imagepath = kagglehub.dataset_download("adityajn105/flickr30k")

    # Load ViT
    processor = AutoProcessor.from_pretrained(models.VIT)
    model = AutoModel.from_pretrained(models.VIT, use_safetensors=True).to(device)
    model.eval()

    # Read list of unique image filenames
    with open(f"{imagepath}/captions.txt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        image_filenames = sorted({row["image"] for row in reader})

    # Dataset and DataLoader
    dataset = ViTFeatureDataset(image_filenames, f"{imagepath}/Images")
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    features = {}

    for filenames, images in tqdm(dataloader):
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
        for fname, vec in zip(filenames, outputs):
            features[fname] = vec

    # Save
    if os.path.exists(models.IMAGES_PATH):
        os.remove(models.IMAGES_PATH)
    torch.save(features, models.IMAGES_PATH)
    print(f"Saved {len(features)} image embeddings to {models.IMAGES_PATH}")


if __name__ == "__main__":
    main()
