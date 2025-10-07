import torch
from torch.utils.data import DataLoader
from OCR_dataset import OCRDataset
from transformers import logging
import warnings
import numpy as np
from tqdm import tqdm
import evaluate
from utils import compute_cer
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import wandb
import os

logging.set_verbosity_error()

def train_trOCR(config):
    # Initialize wandb
    wandb.init(
        entity="magisterka_kuchta_geisler",
        project="TrOCR-train",
        name="trOCR_training_run",
        config=config
    )
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(config["device"])

    # Load model and processor
    processor = TrOCRProcessor.from_pretrained(config["processor_path"])
    model = VisionEncoderDecoderModel.from_pretrained(config["model_path"])
    model.to(device)
    
    model.config.encoder.hidden_dropout_prob = config["dropout_rate"]
    model.config.encoder.attention_probs_dropout_prob = config["dropout_rate"]
    model.config.decoder.dropout = config["dropout_rate"]
    model.config.decoder.attention_dropout = config["dropout_rate"]
    model.config.decoder.activation_dropout = config["dropout_rate"]
    
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Select optimizer dynamically
    if config["optimizer"].lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    elif config["optimizer"].lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)
    elif config["optimizer"].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

    # Load datasets
    train_dataset = OCRDataset(
        base_path=config["train_data_path"],
        processor=processor,
        max_target_length=config["max_target_length"]
    )
    val_dataset = OCRDataset(
        base_path=config["val_data_path"],
        processor=processor,
        max_target_length=config["max_target_length"]
    )
    test_dataset = OCRDataset(
        base_path=config["test_data_path"],
        processor=processor,
        max_target_length=config["max_target_length"]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # Early stopping variables
    best_cer = float("inf")
    patience_counter = 0

    # Training loop
    for epoch in range(config["num_epochs"]):
        model.train()
        train_loss = 0.0

        sanity_index = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']} - Training"):
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            sanity_index += 1
            if config["check_sanity"] and sanity_index == 10:
                break

        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss}")

        with torch.no_grad():

            # Validation
            model.eval()
            valid_cer = 0.0
            valid_loss = 0.0
            cer_metric = evaluate.load("cer")

            sanity_index = 0
            
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']} - Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                valid_loss += loss.item()

                preds = model.generate(batch["pixel_values"])
                cer = compute_cer(pred_ids=preds, label_ids=batch["labels"], processor=processor, cer_metric=cer_metric)
                valid_cer += cer

                sanity_index += 1
                if config["check_sanity"] and sanity_index == 10:
                    break

            avg_valid_cer = valid_cer / len(val_dataloader)
            avg_valid_loss = valid_loss / len(val_dataloader)
            print(f"Epoch {epoch+1} - Validation CER: {avg_valid_cer}, Validation Loss: {avg_valid_loss}")

            # Test
            model.eval()
            test_cer = 0.0
            test_loss = 0.0
            cer_metric = evaluate.load("cer")

            sanity_index = 0

            for batch in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']} - Test"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                test_loss += loss.item()

                preds = model.generate(batch["pixel_values"])
                cer = compute_cer(pred_ids=preds, label_ids=batch["labels"], processor=processor, cer_metric=cer_metric)
                test_cer += cer

                sanity_index += 1
                if config["check_sanity"] and sanity_index == 10:
                    break

            avg_test_cer = test_cer / len(test_dataloader)
            avg_test_loss = test_loss / len(test_dataloader)
            print(f"Epoch {epoch+1} - Test CER: {avg_test_cer}, Test Loss: {avg_test_loss}")

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "val_cer": avg_valid_cer,
            "test_cer": avg_test_cer,
            "train_loss": avg_train_loss,
            "val_loss": avg_valid_loss,
            "test_loss": avg_test_loss
        })

        # Save model at interval epochs
        if (epoch + 1) % config["checkpoint_interval_epochs"] == 0:
            checkpoint_dir = os.path.join(config["output_dir"], f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(checkpoint_dir)
            # processor.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # Early stopping
        if avg_valid_cer < best_cer:
            best_cer = avg_valid_cer
            patience_counter = 0
            # Save the best model
            model.save_pretrained(os.path.join(config["output_dir"], "best_model"))
            processor.save_pretrained(os.path.join(config["output_dir"], "best_model"))
            print(f"Epoch {epoch+1}: Best model saved with CER: {best_cer}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1}: No improvement. Patience counter: {patience_counter}")

        if patience_counter >= config["early_stopping_patience"]:
            print("Early stopping triggered.")
            break


            
    wandb.finish()

if __name__ == "__main__":
    
    config = {
        "batch_size": 16,
        "lr": 0.0001,
        "num_epochs": 50,
        "max_target_length": 128,
        "dropout_rate": 0.1,
        "optimizer": "adamw",  # Add optimizer type here (e.g., 'adamw', 'sgd', 'adam'.)
        "momentum": 0.9, # If set to sgd
        "early_stopping": True,
        "checkpoint_interval_epochs": 1,
        "train_data_path": "/workspace/data/one_line/train",
        "val_data_path": "/workspace/data/one_line/val",
        "test_data_path": "/workspace/data/one_line/test",
        "model_path": "microsoft/trocr-base-handwritten",
        "processor_path": "microsoft/trocr-base-handwritten",
        "output_dir": "/workspace/output",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "early_stopping_patience": 10,
        "check_sanity": False  # Set to True for debugging, will limit training to 10 batches
    }
    
    train_trOCR(config)