import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import re
from nltk import edit_distance
from tqdm import tqdm
from transformers import VisionEncoderDecoderConfig, DonutProcessor, VisionEncoderDecoderModel
from donut_dataset import DonutDataset
import wandb
import uuid
import os

MODEL_TOKEN_START = "<ocr>"
MODEL_TOKEN_END = '<ocr/>'

def compute_edit_distance(pred, answer):
    pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
    answer = re.sub(r"<.*?>", "", answer, count=1).replace("</s>", "")
    return edit_distance(pred, answer) / max(len(pred), len(answer))

def train(config):
    # Initialize wandb
    wandb.init(
        entity="magisterka_kuchta_geisler",
        project="Donut-train",
        name="donut_training_run",
        config=config
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    donut_config = VisionEncoderDecoderConfig.from_pretrained(config["pretrained_model_name_or_path"])
    donut_config.encoder.image_size = config["input_size"]
    donut_config.decoder.max_length = config["max_length"]

    # For encoder (Vision Transformer)
    donut_config.encoder.hidden_dropout_prob = config["dropout_rate"]
    donut_config.encoder.attention_probs_dropout_prob = config["dropout_rate"]
    
    # For decoder (BART)
    donut_config.decoder.dropout = config["dropout_rate"]
    donut_config.decoder.attention_dropout = config["dropout_rate"]
    donut_config.decoder.activation_dropout = config["dropout_rate"]

    processor = DonutProcessor.from_pretrained(config["pretrained_model_name_or_path"])
    model = VisionEncoderDecoderModel.from_pretrained(config["pretrained_model_name_or_path"], config=donut_config).to(device)

    processor.image_processor.size = config["input_size"][::-1]
    processor.image_processor.do_align_long_axis = False

    datasets = {}

    for split in ["train", "val", "test"]:
        datasets[split] = DonutDataset(
            model,
            processor,
            config["dataset_name_or_path"] + '/' + split,
            max_length=config["max_length"],
            split=split,
            task_start_token=MODEL_TOKEN_START,
            prompt_end_token=MODEL_TOKEN_END,
            sort_json_key=False,
            check_sanity=config["check_sanity"],
            batch=config["batch_size"]
        )

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([MODEL_TOKEN_START])[0]

    train_loader = DataLoader(datasets["train"], batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(datasets["val"], batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(datasets["test"], batch_size=config["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # Generate unique experiment ID
    experiment_id = str(uuid.uuid4())

    result_path = Path(config["result_path"]) / experiment_id
    result_path.mkdir(parents=True, exist_ok=True)

    processor.save_pretrained(result_path / "processor")

    best_val_loss = float("inf")
    early_stopping_patience = config["early_stopping_patience"]
    early_stopping_counter = 0

    for epoch in range(config["max_epochs"]):
        # --- TRAINING ---
        model.train()
        train_loss = 0
        sanity_index = 0
        for pixel_values, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} - Training"):
            pixel_values, labels = pixel_values.to(device), labels.to(device)

            optimizer.zero_grad()

            loss = model(pixel_values, labels=labels).loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            sanity_index += 1
            if config["check_sanity"] and sanity_index == 10:
                break
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{config['max_epochs']} - Train Loss: {train_loss:.4f}")

        # --- VALIDATION ---
        def run_validation(model, val_loader, processor, config, device, epoch, name="Validation"):
            model.eval()
            total_loss = 0.0
            val_scores = []
            with torch.no_grad():
                sanity_index = 0
                for pixel_values, labels, answers in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} - {name}"):
                    pixel_values, labels = pixel_values.to(device), labels.to(device)
                    batch_size = pixel_values.shape[0]

                    # Loss calculation
                    loss = model(pixel_values, labels=labels).loss
                    total_loss += loss.item()

                    # Generate predictions
                    decoder_input_ids = torch.full((batch_size, 1), model.config.decoder_start_token_id, device=device)

                    outputs = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=config["max_length"],
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                    )

                    preds = []
                    for seq in processor.tokenizer.batch_decode(outputs.sequences):
                        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
                        seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
                        preds.append(seq)

                    if len(preds) != len(answers[0]):
                        print(f"Warning: Number of predictions ({len(preds)}) does not match number of answers ({len(answers[0])})")
                        continue

                    scores = [compute_edit_distance(pred, ans) for pred, ans in zip(preds, answers[0])]
                    val_scores.extend(scores)
                    
                    sanity_index += 1
                    if config["check_sanity"] and sanity_index == 10:
                        break

            total_loss /= len(val_loader)
            val_cer = np.mean(val_scores)

            return total_loss, val_cer

        val_loss, val_cer = run_validation(model, val_loader, processor, config, device, epoch)
        print(f"Epoch {epoch+1}/{config['max_epochs']} - Val Loss: {val_loss:.4f}, Val CER: {val_cer:.4f}")
        
        # --- TESTING ---
        test_loss, test_cer = run_validation(model, test_loader, processor, config, device, epoch, name="Testing")
        print(f"Epoch {epoch+1}/{config['max_epochs']} - Test Loss: {test_loss:.4f}, Test CER: {test_cer:.4f}")

        # Log to wandb
        wandb.log({"epoch": epoch + 1, "test_loss": test_loss, "test_cer": test_cer, "val_loss": val_loss, "val_cer": val_cer, "train_loss": train_loss})

        # Save model at interval epochs
        if (epoch + 1) % config["checkpoint_interval_epochs"] == 0:
            checkpoint_dir = os.path.join(result_path, f"checkpoint_epoch_{epoch+1}")
            model.save_pretrained(checkpoint_dir)
            print(f"Checkpoint saved at epoch {epoch+1}")

        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained(result_path)
            print(f"Model saved to {result_path}")
            early_stopping_counter = 0  # reset the counter if we get a new best model
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")

        # Check for early stopping
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break

    print("Training complete!")
    wandb.finish()


if __name__ == "__main__":
    config = {
        "max_epochs": 20,
        "lr": 1e-4,
        "batch_size": 2,
        "max_length": 256,
        "pretrained_model_name_or_path": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "result_path": "result",
        "dataset_name_or_path": "datasets",
        "input_size": [1280, 960],
        "early_stopping_patience": 5,
        "checkpoint_interval_epochs": 1,
        "check_sanity": False,
        "dropout_rate": 0.1,
    }

    train(config)