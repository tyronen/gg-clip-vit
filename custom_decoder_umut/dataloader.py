import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader                                                                               
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    def __init__(self, images_folder, captions_csv, tokenizer, max_len=128):                                                   
        self.images_folder = images_folder                                                                                     
        self.data = pd.read_csv(                                                                                               
            captions_csv,
            delimiter='|',                                                                                                     
            usecols=[0, 2],        # Image name and caption                                                                    
            skiprows=1,            # Skip header                                                                               
            names=['image_name', 'caption'],                                                                                   
        )

        self.tokenizer = tokenizer
        self.max_len = max_len

        # Define image transforms to fit your encoder input (e.g., CLIP)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # depends on your encoder input size
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                 std=(0.26862954, 0.26130258, 0.27577711)),  # CLIP normalization
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.images_folder, row['image_name'])
    
        # Load image safely
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
    
        # Get caption safely
        caption = row['caption']
        if not isinstance(caption, str) or caption.strip() == "" or caption == "nan":
            print(f"[Warning] Invalid caption at index {idx}: {caption}")
            caption = "<unk>"
        else:
            caption ="<|startoftext|> " + caption.strip() + " <|endoftext|>"
    
        try:
            encoded = self.tokenizer(
                caption,
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
        except Exception as e:
            print(f"[Tokenizer Error] Caption at index {idx}: {caption} | Error: {e}")
            encoded = self.tokenizer(
                "<unk> <|endoftext|>",
                padding='max_length',
                truncation=True,
                max_length=self.max_len,
                return_tensors='pt'
            )
    
        tgt_ids = encoded.input_ids.squeeze(0)
        return image, tgt_ids

