import os
import json
import torch
from PIL import Image
from tqdm import tqdm

import requests
import wandb
import time
from transformers import AutoProcessor, AutoModelForCausalLM
from itertools import islice
from jiwer import wer, cer
import editdistance

def evaluate_model_original(max_tokens):
    wandb.init(
        entity="magisterka_kuchta_geisler",
        project="Florence-2-large-ft",
        name=f"zero-shot-tokens-{max_tokens}-masked-original",
        config={
            "model": "microsoft/Florence-2-large-ft",
            "task": "<OCR_WITH_REGION>",
            "prompt": "<OCR_WITH_REGION>",
            "max_new_tokens": max_tokens,
            "num_beams": 3
        }
    )
    error_table = wandb.Table(columns=["file_stem", "image_path", "expected", "predicted"])
    summary_table = wandb.Table(columns=["image_path", "WER", "CER", "LER"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

    prompt = "<OCR_WITH_REGION>"
    task = "<OCR_WITH_REGION>"

    with open("/home/macierz/s184780/MGR/splits/test_original/data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Przekształcenie na słownik {id: {"transcription": ..., "image_path": ...}}
    data_by_id = {str(item["id"]).zfill(4): item for item in data} 
    image_dir = "/home/macierz/s184780/MGR/splits/test_original/masked_output"
    # image_dir = "/home/macierz/s184780/MGR/splits/test_original/images"

    errors_path = "errors.jsonl"
    error_log = open(errors_path, "w", encoding="utf-8")

    wer_total = 0
    cer_total = 0
    ler_total = 0
    count = 0

    for file_name in tqdm(os.listdir(image_dir)):
        if not file_name.endswith(".jpg"):
            print(f"Brak pliku: {file_name}")
            continue
            
        file_stem = os.path.splitext(file_name)[0]

        if file_stem not in data_by_id:
            print(f"Brak danych OCR dla pliku: {file_name}")
            continue

        try:
            expected_transcription = data_by_id[file_stem]["transcription"]
            image_path = os.path.join(image_dir, file_name)
            image = Image.open(image_path)

            inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_tokens,
                do_sample=False,
                num_beams=3
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            generated_text = generated_text.replace("</s>", "").strip()
            parsed = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
            if "labels" in parsed.get(task, {}):
                predicted_lines = parsed[task]["labels"]
            else:
                predicted_lines = []
                print(f"⚠️ Brak `labels` w odpowiedzi modelu dla pliku {file_name}")

            # Jawna konwersja
            joined_pred = " ".join(predicted_lines) if isinstance(predicted_lines, list) else str(predicted_lines)
            joined_target = " ".join(expected_transcription) if isinstance(expected_transcription, list) else str(expected_transcription)

            # Debug output
            print(f"[DEBUG] Target: \"{joined_target}\"")
            print(f"[DEBUG] Prediction: \"{joined_pred}\"")
            print(f"[DEBUG] Target length: {len(joined_target)}")
            print(f"[DEBUG] Prediction length: {len(joined_pred)}")

            # Zabezpieczenie przed pustym targetem
            if not joined_target.strip():
                print(f"⚠️ PUSTY target dla pliku {file_name}, pomijam...")
                continue

            # Oblicz metryki
            try:
                wer_score = min(wer(joined_target, joined_pred), 1.0)
                cer_score = min(cer(joined_target, joined_pred), 1.0)
                ler_raw = editdistance.eval(joined_target, joined_pred)
                ler_score = min(ler_raw / max(len(joined_target), 1), 1.0)

                # Debug
                if wer_score > 0.95 or cer_score > 0.95:
                    print(f"[⚠️] Wysoka metryka:\n  Target: \"{joined_target}\"\n  Prediction: \"{joined_pred}\"\n  WER: {wer_score:.3f}, CER: {cer_score:.3f}, LER: {ler_score:.3f}")
            except Exception as e:
                print(f"❌ Błąd przy liczeniu metryk dla {file_name}: {e}")
                continue

            wer_total += wer_score
            cer_total += cer_score
            ler_total += ler_score
            count += 1

            summary_table.add_data(image_path, wer_score, cer_score, ler_score)

            if predicted_lines != expected_transcription:
                error_table.add_data(
                file_stem,
                image_path,
                "\n".join(expected_transcription),
                "\n".join(predicted_lines)
                )

        except Exception as e:
            print(f"❌ Błąd przetwarzania pliku {file_name}: {e}")
            continue

    if count == 0:
        print("❌ Brak poprawnie przetworzonych przykładów – sprawdź dane wejściowe.")
        error_log.close()
        wandb.finish()
        return
        
    avg_wer = wer_total / count
    avg_cer = cer_total / count
    avg_ler = ler_total / count

    print(f"\nŚrednie metryki dla {count} przykładów:")
    print(f"  WER: {avg_wer:.4f}")
    print(f"  CER: {avg_cer:.4f}")
    print(f"  LER: {avg_ler:.4f}")

    wandb.log({
        "avg_WER": avg_wer,
        "avg_CER": avg_cer,
        "avg_LER": avg_ler,
        "samples_evaluated": count,
        "file_metrics": summary_table,
        "ocr_errors": error_table
    })

    error_log.close()
    wandb.finish() 


def evaluate_model(max_tokens):
    wandb.init(
        entity="magisterka_kuchta_geisler",
        project="Florence-2-large-ft",
        name=f"zero-shot-tokens-{max_tokens}",
        config={
            "model": "microsoft/Florence-2-large-ft",
            "task": "<OCR_WITH_REGION>",
            "prompt": "<OCR_WITH_REGION>",
            "max_new_tokens": max_tokens,
            "num_beams": 3
        }
    )
    # error_table = wandb.Table(columns=["file_id", "image_path", "expected", "predicted"])
    summary_table = wandb.Table(columns=["image_path", "WER", "CER", "LER"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

    prompt = "<OCR_WITH_REGION>"
    task = "<OCR_WITH_REGION>"

    with open("/home/macierz/s184780/MGR/splits/test/test_combined.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # Przekształcenie na słownik {id: {"transcription": ..., "image_path": ...}}
    data_by_id = {item["id"]: item for item in data}
    image_dir = "/home/macierz/s184780/MGR/splits/test/images"

    errors_path = "errors.jsonl"
    error_log = open(errors_path, "w", encoding="utf-8")

    wer_total = 0
    cer_total = 0
    ler_total = 0
    count = 0

    for file_name in tqdm(os.listdir(image_dir)):
        if not file_name.endswith(".png"):
            continue
        
        try:
            file_id = int(file_name.split(".")[0])
        except ValueError:
            continue

        if file_id not in data_by_id:
            print(f"Brak danych OCR dla pliku: {file_name}")
            continue

        expected_transcription = data_by_id[file_id]["transcription"]
        image_path = os.path.join(image_dir, file_name)
        image = Image.open(image_path)

        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=3
        )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        generated_text = generated_text.replace("</s>", "").strip()
        parsed = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
        if "labels" in parsed.get(task, {}):
            predicted_lines = parsed[task]["labels"]
        else:
            predicted_lines = []
            print(f"⚠️ Brak `labels` w odpowiedzi modelu dla pliku {file_name}")

        joined_pred = " ".join(predicted_lines)
        joined_target = " ".join(expected_transcription)

        wer_score = wer(joined_target, joined_pred)
        cer_score = cer(joined_target, joined_pred)
        ler_score = editdistance.eval(joined_target, joined_pred) / max(len(joined_pred), 1)

        wer_total += wer_score
        cer_total += cer_score
        ler_total += ler_score
        count += 1

        summary_table.add_data(image_path, wer_score, cer_score, ler_score)

        # if predicted_lines != expected_transcription:
        #     error_table.add_data(
        #     file_id,
        #     image_path,
        #     "\n".join(expected_transcription),
        #     "\n".join(predicted_lines)
        #     )
        #     json.dump({
        #         "file_id": file_id,
        #         "file_name": file_name,
        #         "wer": wer_score,
        #         "cer": cer_score,
        #         "ler": ler_score,
        #         "expected": expected_transcription,
        #         "predicted": predicted_lines
        #     }, error_log, ensure_ascii=False)
        #     error_log.write("\n")
    
    avg_wer = wer_total / count
    avg_cer = cer_total / count
    avg_ler = ler_total / count

    print(f"\nŚrednie metryki dla {count} przykładów:")
    print(f"  WER: {avg_wer:.4f}")
    print(f"  CER: {avg_cer:.4f}")
    print(f"  LER: {avg_ler:.4f}")

    wandb.log({
        "avg_WER": avg_wer,
        "avg_CER": avg_cer,
        "avg_LER": avg_ler,
        "samples_evaluated": count,
        "file_metrics": summary_table
        # "ocr_errors": error_table
    })

    error_log.close()
    wandb.finish() 


if __name__ == "__main__":
    for max_tokens in [1024, 512, 256]:
        # evaluate_model(max_tokens)
        evaluate_model_original(max_tokens)