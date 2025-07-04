import torch, gc, os
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.optim import Adam
from dataloader import ImageCaptionDataset
from transformers import AutoTokenizer
from decoder import TransformerTextDecoder
from transformers import CLIPModel, CLIPProcessor
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)

special_token = "<|startoftext|>"
if special_token not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'bos_token': special_token})
    print(f"✅ Added {special_token} to tokenizer.")

from torch.utils.data import DataLoader 

#Instantiate the dataset
dataset = ImageCaptionDataset(images_folder='../Images', captions_csv='../Images/results.csv', tokenizer=tokenizer,  max_len=128) 

#Create dataloader
dataloader = DataLoader(dataset, batch_size=24, 
    shuffle=True,                                                                                                                                                                          
    num_workers=16, pin_memory = torch.cuda.is_available()) 

# Load model and processor
model_id = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(model_id, output_hidden_states=True)
processor = CLIPProcessor.from_pretrained(model_id, use_fast=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Your decoder, vocab size must match tokenizer
vocab_size = tokenizer.vocab_size + 1
embed_dim = 1024  # or match CLIP's embedding dim if you want

decoder = TransformerTextDecoder(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_layers=4,
    nhead=8,
    dim_feedforward=2048,
    max_len=128
).to(device)

old_embeddings = decoder.embed_tokens  # or decoder.embedding if that's your name
old_vocab_size, embed_dim = old_embeddings.weight.shape

new_vocab_size = len(tokenizer) + 1
if new_vocab_size > old_vocab_size:
    # Create new embedding layer
    new_embeddings = nn.Embedding(new_vocab_size, embed_dim)
    # Copy weights from old to new
    new_embeddings.weight.data[:old_vocab_size] = old_embeddings.weight.data
    # Replace in the model
    decoder.embed_tokens = new_embeddings.to(device)

decoder.eval()


# CLIP model to eval mode, freeze weights
clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False
clip_model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = Adam(decoder.parameters(), lr=1e-4)

# Training loop
def train_decoder_model(decoder, clip_model, dataloader, tokenizer=tokenizer, device=device, criterion=criterion,
                        optimizer=optimizer, num_epochs=5, vocab_size=vocab_size):
    loop = tqdm(dataloader, desc="Training")
    print("🔍   Verifying GPU usage:")
    print("🧠 Is model on GPU? ", next(clip_model.parameters()).is_cuda)

    decoder.train()
    clip_model.eval()

    for param in clip_model.parameters():
        param.requires_grad = False
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, tgt_ids) in enumerate(loop):
            images = images.to(device)
            tgt_ids = tgt_ids.to(device)
    
            with torch.no_grad():
                outputs = clip_model.vision_model(pixel_values=images, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]  # all hidden layers except the last
                memory = hidden_states
    
            decoder_input = tgt_ids[:, :-1]     # [B, T-1]
            target_output = tgt_ids[:, 1:]      # [B, T-1]
    
            logits = decoder(decoder_input, memory)  # [B, T-1, V]
            loss = criterion(logits.reshape(-1, vocab_size), target_output.reshape(-1))
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
    
            loop.set_postfix(
                loss=loss.item(),
                avg_loss=total_loss / (batch_idx + 1)
            )

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx} Loss {loss.item():.4f}")

		# Memory management
        del images, tgt_ids, logits, memory, decoder_input, target_output, loss, outputs
        torch.cuda.empty_cache()
        gc.collect()

        save_path = f"decoder_HBOS_E{epoch+1}.pth"
        torch.save(decoder.state_dict(), save_path)
    # Save model after final epoch
    save_path = f"decoder_HBOS_E{num_epochs}.pth"
    torch.save(decoder.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

train_decoder_model(decoder=decoder, clip_model=clip_model, dataloader=dataloader, num_epochs=5)
#nohup torchrun --nproc-per-node=1 train.py > log.md 2>&1 &

