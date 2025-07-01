import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
import wandb
from tqdm import tqdm
import utils
from torch.utils.data import DataLoader
import models
import subprocess

hyperparameters = {
    "batch_size": 192,
    "model_dim": 512,
    "ffn_dim": 2048,
    "num_heads": 8,
    "num_decoders": 4,
    "learning_rate": 1e-4,
    "epochs": 50,
    "dropout": 0.1,
    "patience": 3,
    "data_fraction": 0.1,
    "label_smoothing": 0.1,
}

sweep_config = {
    "method": "random",  # can be 'grid', 'random', or 'bayes'
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {"values": [192]},
        "model_dim": {"values": [384, 512]},
        "ffn_dim": {"values": [1536, 2048]},
        "num_heads": {"values": [8]},
        "num_decoders": {"values": [4]},
        "learning_rate": {"distribution": "log_uniform_values", "min": 5e-5, "max": 5e-4},
        "epochs": {"values": [20]},
        "dropout": {"values": [0.0, 0.1, 0.2]},
        "patience": {"values": [3, 5, 10]},
        "data_fraction": { "values": [0.1]},
        "label_smoothing": {"values": [0.0, 0.05, 0.1]},
    },
}

parser = argparse.ArgumentParser(description="Train simple model")
parser.add_argument("--entity", help="W and B entity", default="mlx-institute")
parser.add_argument("--project", help="W and B project", default="custom-decoder")
parser.add_argument("--sweep", help="Run a sweep", action="store_true")
parser.add_argument("--check", help="Make sure it works", action="store_true")
args = parser.parse_args()


def collate_fn(batch):
    images, input_ids = zip(*batch)
    images     = torch.stack(images)            # [B, 768]
    input_ids  = torch.stack(input_ids)         # [B, L]
    return {"images": images, "input_ids": input_ids}


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, device, batch_size, train=False):
        num_workers = 8 if device.type == "cuda" else 0 if device.type == "mps" else 4
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            drop_last=train,
            pin_memory=(device.type == "cuda"),
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )


def validate_model(run, model, validation_dataloader, epoch, device, config):
    model.eval()

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Important: no gradients during validation
        for i, batch in tqdm(enumerate(validation_dataloader), desc="Validation"):
            # move only tensor items to the target device
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }

            # Forward pass (shared with training)
            loss, logits, labels = loss_fn(batch, model, config["label_smoothing"])

            # loss already computed by loss_fn
            total_loss += loss.item()
            num_batches += 1

    model.train()  # Set back to training mode

    return total_loss / num_batches


def loss_fn(batch, model, label_smoothing):
    logits, labels = model(batch["images"], batch["input_ids"])  

    vocab_size = logits.size(-1)
    loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels.contiguous().view(-1),
        ignore_index=model.tokenizer.pad_token_id,
        label_smoothing=label_smoothing,
    )
    return loss, logits, labels


def get_git_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def main():
    if args.sweep:
        sweep_id = wandb.sweep(sweep_config,  entity=args.entity, project=args.project)
        wandb.agent(sweep_id, function=run_training)
    else:
        config = dict(hyperparameters)  # makes a shallow copy
        config["git_commit"] = get_git_commit()
        run_training(config)

def run_training(config):
    utils.setup_logging()
    device = utils.get_device()

    run = wandb.init(entity=args.entity, project=args.project,
        # Track hyperparameters and run metadata.
        config=config,
    )

    train_dataset = models.Flickr30kDataset(split="train", data_fraction=config["data_fraction"])
    validation_dataset = models.Flickr30kDataset(split="val", data_fraction=config["data_fraction"])
    test_dataset = models.Flickr30kDataset(split="test", data_fraction=config["data_fraction"])
    logging.info(
        f"Dataset sizes: training {len(train_dataset)} validation: {len(validation_dataset)} test: {len(test_dataset)}"
    )
    training_dataloader = CustomDataLoader(train_dataset, device, train=True, batch_size=config["batch_size"])
    validation_dataloader = CustomDataLoader(validation_dataset, device, batch_size=config["batch_size"])
    test_dataloader = CustomDataLoader(test_dataset, device, batch_size=config["batch_size"])

    # Total optimizer steps = batches per epoch × epochs
    total_steps = len(training_dataloader) * config["epochs"]

    maybe_autocast, scaler = utils.amp_components(device, True)
    model = models.CombinedTransformer(
        model_dim=config["model_dim"],
        ffn_dim=config["ffn_dim"],
        num_heads=config["num_heads"],
        num_decoders=config["num_decoders"],
        dropout=config["dropout"],
    ).to(device)
    wandb.watch(model, log="all", log_freq=100)
    wandb.define_metric("val_loss", summary="min")
    params = [
        {
            "params": model.parameters(),
            "lr": config["learning_rate"],
        },
    ]
    optimizer = optim.Adam(params)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    best_val_loss = float("inf")
    patience_counter = 0
    last_epoch = 0
    grad_norm = 0
    for epoch in range(config["epochs"]):

        total_train_loss = 0.0
        num_train_batches = 0
        for i, batch in enumerate(tqdm(training_dataloader, desc=f"Epoch {epoch + 1}")):
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            optimizer.zero_grad()
            with maybe_autocast:
                loss, logits, labels = loss_fn(batch, model, config["label_smoothing"])

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

        logging.info(f"Epoch {epoch + 1}/{config['epochs']}")
        avg_train_loss = total_train_loss / num_train_batches
        avg_val_loss = validate_model(run, model, validation_dataloader, epoch, device, config)
        scheduler.step()

        run.log(
            {
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "grad_norm": grad_norm,
            },
        )
        last_epoch += 1
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "parameters": params,
                    "model_dim": config["model_dim"],
                    "ffn_dim": config["ffn_dim"],
                    "num_heads": config["num_heads"],
                    "num_decoders": config["num_decoders"],
                    "dropout": config["dropout"],
                },
                utils.MODEL_FILE,
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                run.log({"early_stopping_epochs": epoch + 1})
                break
        if args.check:
            break
    checkpoint = torch.load(utils.MODEL_FILE)
    model.load_state_dict(checkpoint["state_dict"])
    test_loss = validate_model(run, model, test_dataloader, last_epoch + 1, device, config)
    run.log(
        {"test_loss": test_loss},
    )
    artifact = wandb.Artifact(name="basic-decoder-model", type="model")
    artifact.add_file(utils.MODEL_FILE)
    run.log_artifact(artifact)
    run.finish(0)


if __name__ == "__main__":
    main()
