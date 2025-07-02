from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
import datasets
import utils
from tqdm import tqdm


def generate_caption(device, model, processor, image_path):
    # Load and process the image
    image = Image.open(image_path)

    # Create a conversation format (Qwen expects chat format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Apply chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Process inputs
    inputs = processor(
        text=text,
        images=[image],
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Decode the response
    caption = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return caption


def main():
    utils.setup_logging()
    device = utils.get_device()
    # Load the model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.float16, device_map="auto"
    ).to(device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct").to(device)
    # Load images (from anywhere - web scraping, stock photos, etc.)
    idata = datasets.load_dataset("1aurent/unsplash_lite")

    synthetic_dataset = []
    for row in tqdm(idata[0]):
        # Generate caption using Qwen
        caption = generate_caption(device, model, processor, row["image"])
        synthetic_dataset.append({"image": row["image"], "caption": caption})

    # Upload to Hugging Face
    dataset = datasets.Dataset.from_list(synthetic_dataset)
    dataset.push_to_hub("tyronen/synthetic-captions")


if __name__ == "__main__":
    main()
