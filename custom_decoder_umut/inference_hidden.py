import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from decoder import TransformerTextDecoder
from PIL import Image

# === Load components ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and clip model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", output_hidden_states=True).to(device)
clip_model.eval()
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


special_token = "<|startoftext|>"
if special_token not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({'bos_token': special_token})
    print(f"✅ Added {special_token} to tokenizer.")


# Load trained decoder
vocab_size = tokenizer.vocab_size + 1
embed_dim = 1024  # same as during training
print("pad_token:", tokenizer.pad_token)

decoder = TransformerTextDecoder(
    vocab_size=vocab_size,
    embed_dim=embed_dim,
    num_layers=4,
    nhead=8,
    dim_feedforward=2048,
    max_len=128
).to(device)

print(tokenizer.tokenize("<|startoftext|>"))
print(tokenizer.tokenize("<|endoftext|>"))


trained_model = "decoder_eos_H_E1"
trained_model = "decoder_pool_E1"
trained_model = "decoder_eos_H_E4_nvidia"
decoder.load_state_dict(torch.load(f"models/{trained_model}.pth", map_location=device))

import torch.nn as nn

old_embeddings = decoder.embed_tokens  # or decoder.embedding if that's your name
old_vocab_size, embed_dim = old_embeddings.weight.shape

new_vocab_size = len(tokenizer) + 1
if new_vocab_size > old_vocab_size:
    # Create new embedding layer
    new_embeddings = nn.Embedding(new_vocab_size, embed_dim)
    # Copy weights from old to new
    new_embeddings.weight.data[:old_vocab_size] = old_embeddings.weight.data
    # Replace in the model
    decoder.embed_tokens = new_embeddings

decoder.eval()
print(tokenizer.tokenize("<|startoftext|>"))

import torch.nn.functional as F


def generate_caption(
    image_path,
    max_len=30,
    start_token="<|startoftext|>",   # Use your actual start token
    end_token="<|endoftext|>",
    top_k=50,
    top_p=0.9,
    temperature=1.0
):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = clip_model.vision_model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        memory = hidden_states

    start_id = tokenizer.convert_tokens_to_ids(start_token)
    end_id = tokenizer.convert_tokens_to_ids(end_token)

    if start_id is None or end_id is None:
        raise ValueError(f"Start or end token not found in tokenizer. Got start_id={start_id}, end_id={end_id}")

    generated = [start_id]

    for _ in range(max_len):
        decoder_input = torch.tensor([generated], dtype=torch.long).to(device)  # shape (1, seq_len)

        # Pass decoder_input and memory to decoder
        logits = decoder(decoder_input, memory)  # Adjust if your decoder expects different dims
        next_token_logits = logits[0, -1, :] / temperature

        # Top-k sampling
        top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
        probs = torch.zeros_like(next_token_logits).scatter_(0, top_k_indices, F.softmax(top_k_values, dim=-1))

        # Top-p filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
        probs = probs / probs.sum()

        next_token_id = torch.multinomial(probs, 1).item()

        if next_token_id == end_id:
            break

        generated.append(next_token_id)

    caption = tokenizer.decode(generated[1:], skip_special_tokens=True)
    return caption.capitalize()


image_name = input("Enter the image: ")
caption = generate_caption(f"{image_name}.jpg")
print("Generated Caption:", caption)


print("BOS token:", tokenizer.bos_token)
print("BOS token ID:", tokenizer.bos_token_id)

print("EOS token:", tokenizer.eos_token)
print("EOS token ID:", tokenizer.eos_token_id)
print("Tokenized EOS manually:", tokenizer("<|endoftext|>").input_ids)

print("CLS token:", tokenizer.cls_token)
print("SEP token:", tokenizer.sep_token)

print("All special tokens:", tokenizer.all_special_tokens)

