import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import wandb
from tqdm import tqdm
import utils
from torch.utils.data import DataLoader
import numpy as np
import models
import kagglehub

hyperparameters = {
    "model_dim": 512,
    "ffn_dim": 2048,
    "num_heads": 8,
    "num_decoders": 6,
    "learning_rate": 1e-4,
    "batch_size": 256,
    "epochs": 50,
    "dropout": 0.1,
    "patience": 5,
    "temperature": 0.05,
}

parser = argparse.ArgumentParser(description="Train simple model")
parser.add_argument("--entity", help="W and B entity", default="mlx-institute")
parser.add_argument("--project", help="W and B project", default="custom-decoder")
args = parser.parse_args()


def collate_fn(batch):
    images, texts = zip(*batch)
    return {"images": list(images), "texts": list(texts)}


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, device):
        num_workers = 8 if device.type == "cuda" else 0 if device.type == "mps" else 4
        super().__init__(
            dataset,
            batch_size=hyperparameters["batch_size"],
            shuffle=True,
            drop_last=True,
            pin_memory=(device.type == "cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )


def validate_model(run, model, validation_dataloader, epoch, device):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    image_similarities = []
    text_similarities = []

    all_images = []
    all_texts = []

    with torch.no_grad():  # Important: no gradients during validation
        for i, batch in tqdm(enumerate(validation_dataloader), desc="Validation"):
            # move only tensor items to the target device
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }

            # Forward pass (shared with training)
            (
                loss,
                logits,
                labels,
                image_embeds,
                text_embeds,
            ) = _contrastive_forward(batch, model, hyperparameters["temperature"])

            if i == 0:  # Only for first batch

                # Check if embeddings are too similar to each other
                image_mean = image_embeds.mean(dim=0)
                text_mean = text_embeds.mean(dim=0)

                # Check variance across dimensions
                image_var = image_embeds.var(dim=0).mean()
                text_var = text_embeds.var(dim=0).mean()

                all_images.extend(image_embeds.cpu())
                all_texts.extend(text_embeds.cpu())

                run.log(
                    {
                        "embedding_means_image_norm": image_mean.norm().item(),
                        "embedding_means_text_norm": text_mean.norm().item(),
                        "embedding_variance_image": image_var.item(),
                        "embedding_variance_text": text_var.item(),
                    },
                    step=epoch,
                )

            # loss already computed by _contrastive_forward
            total_loss += loss.item()
            num_batches += 1

    run.log(
        {
            "validation_image_similarities_mean": np.mean(image_similarities),
            "validation_image_similarities_std": np.std(image_similarities),
            "validation_text_similarities_mean": np.mean(text_similarities),
            "validation_text_similarities_std": np.std(text_similarities),
        },
        step=epoch,
    )

    model.train()  # Set back to training mode

    return total_loss / num_batches


# ------------------------------------------------------------------ #
# Shared forward pass + in‑batch contrastive loss
# ------------------------------------------------------------------ #
def _contrastive_forward(batch, model, temperature):
    image_embeds, text_embeds = model(batch["images"], batch["texts"])

    logits = torch.matmul(image_embeds, text_embeds.T) / temperature  # [B,2B]
    labels = torch.arange(len(image_embeds), device=image_embeds.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    loss = (loss_i2t + loss_t2i) / 2
    return loss, logits, labels, image_embeds, text_embeds


def main():
    utils.setup_logging()
    device = utils.get_device()

    config = {
        **hyperparameters,
    }
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="mlx-institute",
        # Set the wandb project where this run will be logged.
        project="custom-decoder",
        # Track hyperparameters and run metadata.
        config=config,
    )

    imagepath = kagglehub.dataset_download("adityajn105/flickr30k")

    train_dataset = models.Flickr30kDataset(imagepath, split="train")
    validation_dataset = models.Flickr30kDataset(imagepath, split="val")
    test_dataset = models.Flickr30kDataset(imagepath, split="test")
    logging.info(
        f"Dataset sizes: training {len(train_dataset)} validation: {len(validation_dataset)} test: {len(test_dataset)}"
    )
    training_dataloader = CustomDataLoader(train_dataset, device)
    validation_dataloader = CustomDataLoader(validation_dataset, device)
    test_dataloader = CustomDataLoader(test_dataset, device)

    maybe_autocast, scaler = utils.amp_components(device, True)
    model = models.CombinedTransformer(
        model_dim=hyperparameters["model_dim"],
        ffn_dim=hyperparameters["ffn_dim"],
        num_heads=hyperparameters["num_heads"],
        num_decoders=hyperparameters["num_decoders"],
        dropout=hyperparameters["dropout"],
    )

    params = [
        {
            "params": model.parameters(),
            "lr": hyperparameters["learning_rate"],
        },
    ]
    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0
    grad_norm = 0
    for epoch in range(hyperparameters["epochs"]):

        total_train_loss = 0.0
        num_train_batches = 0
        for i, batch in enumerate(tqdm(training_dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            with maybe_autocast:
                loss, logits, labels, image, text = _contrastive_forward(
                    batch, model, hyperparameters["temperature"]
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += loss.item()
            num_train_batches += 1
            grad_norm = total_norm.item()

        logging.info(f"Epoch {epoch + 1}/{hyperparameters['epochs']}")
        avg_train_loss = total_train_loss / num_train_batches
        avg_val_loss = validate_model(run, model, validation_dataloader, epoch, device)
        scheduler.step(avg_val_loss)

        run.log(
            {
                "learning_rate": optimizer.params["lr"],
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "grad_norm": grad_norm,
            },
            step=epoch,
        )
        last_epoch += 1
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "parameters": params,
                    "embed_dim": hyperparameters["embed_dim"],
                    "dropout_rate": hyperparameters["dropout_rate"],
                },
                utils.MODEL_FILE,
            )
        else:
            patience_counter += 1
            if patience_counter >= hyperparameters["patience"]:
                run.log({"early_stopping_epochs": epoch + 1})
                break
    checkpoint = torch.load(utils.MODEL_FILE)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss = validate_model(
        run,
        model,
        test_dataloader,
        last_epoch + 1,
        device,
    )
    run.log(
        {"test_loss": test_loss},
        step=last_epoch + 1,
    )
    artifact = wandb.Artifact(name="basic-decoder-model", type="model")
    artifact.add_file(utils.MODEL_FILE)
    run.log_artifact(artifact)
    run.finish(0)


if __name__ == "__main__":
    main()
