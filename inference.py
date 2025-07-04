import streamlit as st
import torch
import requests
from PIL import Image
import io
import logging
import models
import utils

# Suppress warnings
utils.setup_logging()
logging.getLogger("transformers").setLevel(logging.ERROR)

# Configuration - update these paths as needed
DEVICE = utils.get_device()


@st.cache_resource
def load_models():
    """Load the trained model"""
    vit_encoder = models.VitEncoder()
    checkpoint = torch.load(utils.MODEL_FILE, map_location=DEVICE)
    model = models.CombinedTransformer(
        model_dim=checkpoint["model_dim"],
        ffn_dim=checkpoint["ffn_dim"],
        num_heads=checkpoint["num_heads"],
        num_decoders=checkpoint["num_decoders"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)
    model.eval()
    st.success("Model loaded successfully!")
    return vit_encoder, model


@st.cache_data
def load_image_from_url(url):
    """Load image from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content))
        return image.convert("RGB")
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        return None


def generate_caption(vit_encoder, model, image, max_length=50):
    """Generate caption for the image"""
    if model is None:
        return "Model not loaded"

    try:
        with torch.no_grad():
            # Encode image
            image_features = models.encode_image(
                image, vit_encoder, model
            )  # [1, model_dim]

            # Initialize with BOS token
            generated = [model.tokenizer.bos_token_id]

            # Generate tokens one by one
            for _ in range(max_length):
                # Convert to tensor
                input_ids = torch.tensor([generated], device=DEVICE)

                # Create the necessary all-False pad mask
                seq_len = input_ids.shape[1] + 1
                pad_mask = torch.zeros(1, seq_len, device=DEVICE, dtype=torch.bool)

                # Use the new decode_step method to get the next token's logits
                logits = model.decode_step(image_features, input_ids, pad_mask)

                # Get next token (greedy decoding)
                next_token = torch.argmax(logits, dim=-1).item()

                # Stop if EOS token
                if next_token == model.tokenizer.eos_token_id:
                    break

                generated.append(next_token)

            # Decode the generated tokens
            caption = model.tokenizer.decode(generated[1:], skip_special_tokens=True)
            return caption

    except Exception as e:
        st.error(f"Error generating caption: {str(e)}")
        return "Error generating caption"


def main():
    st.set_page_config(
        page_title="Image Captioning Server", page_icon="🖼️", layout="wide"
    )

    st.title("🖼️ Image Captioning Server")
    st.markdown("Upload an image URL and get an AI-generated caption!")

    # Load model
    vit_encoder, model = load_models()

    if model is None:
        st.error("Please check your model path and try again.")
        return

    # Initialize session state variables if they don't exist
    if "image_url" not in st.session_state:
        st.session_state.image_url = ""
    if "caption" not in st.session_state:
        st.session_state.caption = ""

    # Create two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Input")

        # Image URL input
        image_url = st.text_input(
            "Enter Image URL:",
            value=st.session_state.image_url,
            placeholder="https://example.com/image.jpg",
            help="Enter a direct URL to an image file",
        )
        # When the input changes, update the session state
        if image_url != st.session_state.image_url:
            st.session_state.image_url = image_url
            st.session_state.caption = ""  # Clear old caption on new image
            st.rerun()  # Rerun to update the right column immediately

        # Example URLs
        st.markdown("**Example URLs:**")
        example_urls = [
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4",
            "https://images.unsplash.com/photo-1551963831-b3b1ca40c98e",
            "https://images.unsplash.com/photo-1518717758536-85ae29035b6d",
        ]

        for i, url in enumerate(example_urls):
            if st.button(f"Example {i + 1}", key=f"example_{i}"):
                st.session_state.image_url = url
                st.session_state.caption = ""
                st.rerun()

        # Generation parameters
        st.header("Parameters")
        max_length = st.slider(
            "Max Caption Length",
            min_value=10,
            max_value=100,
            value=50,
            help="Maximum number of tokens to generate",
        )

    with col2:
        st.header("Output")

        if st.session_state.image_url:
            # Load and display image
            image = load_image_from_url(st.session_state.image_url)

            if image is not None:
                st.image(image, caption="Input Image", use_container_width=True)

                # Generate caption button
                if st.button("Generate Caption", type="primary"):
                    with st.spinner("Generating caption..."):
                        caption = generate_caption(
                            vit_encoder, model, image, max_length
                        )
                        st.session_state.caption = caption
                        logging.info(f"Caption: {st.session_state.caption}")

                    if st.session_state.caption:
                        st.success("Caption generated!")
                        st.write("**Generated Caption:**")
                        st.write(f"*{st.session_state.caption}*")
                        # Copy to clipboard button
                        st.code(st.session_state.caption, language=None)
            else:
                st.error("Failed to load image from URL")
        else:
            st.info("👆 Enter an image URL to get started")


if __name__ == "__main__":
    main()
