import os
import json
import torch
from PIL import Image
from tqdm import tqdm

import requests
import wandb
import time

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from itertools import islice
from jiwer import wer, cer
import editdistance



def evaluate_model_qwen2(max_tokens):
    wandb.init(
        entity="magisterka_kuchta_geisler",
        project="Qwen2.5-VR-72B",
        name=f"zero-shot-tokens-{max_tokens}",
        config={
            "model": "Qwen/Qwen2.5-VL-72B-Instruct",
            "task": "OCR",
            "max_new_tokens": max_tokens,
            "num_beams": 3
        }
    )
    summary_table = wandb.Table(columns=["image_path", "WER", "CER", "LER"])

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-72B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")

    # Load the test data
    with open("/home/macierz/s184780/MGR/splits/test/test_combined.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    data_by_id = {item["id"]: item for item in data}
    image_dir = "/home/macierz/s184780/MGR/splits/test/images"

    errors_path = "errors.jsonl"
    error_log = open(errors_path, "w", encoding="utf-8")

    wer_total = 0
    cer_total = 0
    ler_total = 0
    count = 0

    for file_name in tqdm(islice(os.listdir(image_dir),5)):
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
        # image = Image.open(image_path)

        prompt_text = "Transcribe low-level assembly instructions from image. Code includes mnemonics (mov, add), CPU registers (eax, ecx), 32-bit hex values (e.g., 0x1234), and labels in English (e.g., 'loop', 'end')."
        image_uri = f"file://{image_path}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]

        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(device)

        generated_ids = model.generate(**inputs, max_new_tokens=max_tokens, num_beams=3)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        generated_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        # generated_text = generated_text.strip("\n")

        # Calculate metrics
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
    })

    error_log.close()
    wandb.finish()


if __name__ == "__main__":
    for max_tokens in [1024]:
        evaluate_model_qwen2(max_tokens)
