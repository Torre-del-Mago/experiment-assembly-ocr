import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForCausalLM, get_scheduler
from transformers.utils import is_torch_available
from tqdm import tqdm
import wandb
from florence_dataset import FlorenceDataset


class Config:
    train_json = "/workspace/data/splits_ako/training/training_combined.json"
    val_json = "/workspace/data/splits_ako/validation/validation_combined.json"
    model_name_or_path = "microsoft/Florence-2-large-ft"
    output_dir = "/workspace/outputs/florence"
    batch_size = 2
    learning_rate = 5e-5
    epochs = 1
    max_length = 1024
    run_name = "finetune-florence"
    early_stopping_patience = 3
    checkpoint_interval_epochs = 5


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_dataset = FlorenceDataset(cfg.train_json, max_length=cfg.max_length)
    val_dataset = FlorenceDataset(cfg.val_json, max_length=cfg.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    num_training_steps = cfg.epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    wandb.init(project="Florence-train", config=vars(cfg), name=cfg.run_name)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    model.train()
    for epoch in range(cfg.epochs):
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "pixel_values", "labels"]}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())
            wandb.log({"train/loss": loss.item()})

        val_loss = evaluate(model, val_dataloader, device, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.save_pretrained(cfg.output_dir)
            print(f"\nBest model saved with validation loss: {val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stopping_patience:
                print("\nEarly stopping triggered.")
                break

        # Save checkpoint every N epochs
        if (epoch + 1) % cfg.checkpoint_interval_epochs == 0:
            checkpoint_path = os.path.join(cfg.output_dir, f"checkpoint-epoch{epoch+1}")
            model.save_pretrained(checkpoint_path)
            print(f"\nCheckpoint saved at {checkpoint_path}")

    wandb.finish()


def evaluate(model, dataloader, device, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items() if k in ["input_ids", "pixel_values", "labels"]}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    wandb.log({"val/loss": avg_loss, "epoch": epoch})
    print(f"\nValidation loss after epoch {epoch+1}: {avg_loss:.4f}")
    model.train()
    return avg_loss


if __name__ == "__main__":
    cfg = Config()
    train(cfg)
