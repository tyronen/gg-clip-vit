import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
import sys
sys.path.append('/Users/jc/untitled/mlx8/week4')
from dataset import CaptionDataset, collate_fn, tokenizer
from model import CaptionTransformer
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def loss_fn(logits, labels):
    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    return F.cross_entropy(logits, labels, ignore_index=-100)

def train():
    dataset = CaptionDataset('./mlx8/week4/flickr_annotations_30k.csv', './mlx8/week4/flickr30k-images', tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    model = CaptionTransformer(patch_dim=3*16*16, embed_dim=512, num_patches=196, vocab_size=len(tokenizer))
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader):
            patches = batch['patches'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            decoder_label_ids = batch['decoder_label_ids'].to(device)

            logits = model(patches, decoder_input_ids)
            loss = loss_fn(logits, decoder_label_ids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1} Loss: {total_loss / len(dataloader)}')

def generate_caption(model, patches, tokenizer, max_len=30):
    model.eval()
    sos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    generated = [sos_id]
    for _ in range(max_len):
        input_ids = torch.tensor([generated], dtype=torch.long).to(patches.device)
        logits = model(patches.unsqueeze(0), input_ids)
        next_token = logits[0, -1].argmax(-1).item()
        if next_token == eos_id:
            break
        generated.append(next_token)
    return tokenizer.decode(generated[1:])

if __name__ == '__main__':
    train()






