import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from decoder import TransformerTextDecoder
from PIL import Image

from decoder import TransformerTextDecoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and clip model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", output_hidden_states=True).to(device)
clip_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load trained decoder
vocab_size = tokenizer.vocab_size + 1
embed_dim = 1024  # same as during training



decoder = TransformerTextDecoder(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_layers=4,
    nhead=8,
    dim_feedforward=2048,
    max_len=128
).to(device)

decoder.load_state_dict(torch.load("decoder_E2.pth", map_location=device))
decoder.eval()

print(decoder)

