import os
import csv
import kagglehub
import numpy as np
from tqdm import tqdm
from PIL import Image
import models
import torch
from transformers import AutoModel, AutoProcessor
import utils

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

    # Extract features
    features = {}
    for filename in tqdm(image_filenames):
        path = os.path.join(f"{imagepath}/Images", filename)
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**inputs).last_hidden_state  # [1, 197, 768]
            pooled = output.mean(dim=1).squeeze().cpu().numpy()  # [768]
            features[filename] = pooled

    # Save
    np.savez_compressed(models.IMAGES_NPZ_PATH, **features)
    print(f"Saved {len(features)} image embeddings to {models.IMAGES_NPZ_PATH}")

if __name__ == "__main__":
    main()