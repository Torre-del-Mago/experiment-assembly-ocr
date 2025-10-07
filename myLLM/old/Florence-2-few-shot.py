import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
from jiwer import wer, cer
import editdistance
import wandb
from itertools import islice

# Inicjalizacja wandb
wandb.init(
    entity="magisterka_kuchta_geisler",
    project="Florence-2-large-ft",
    name="few-shot-with-bboxes",
    config={
        "model": "microsoft/Florence-2-large-ft",
        "task": "<REGION_TO_DESCRIPTION>",
        "max_new_tokens": 1024,
        "num_beams": 3
    }
)
error_table = wandb.Table(columns=["file_id", "image_path", "expected", "predicted"])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

# Wczytanie danych
with open("/home/macierz/s184780/MGR/splits/test/test_combined.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Przekształcenie na słownik {id: {"transcription": ..., "image_path": ..., "bbox": [...]} }
data_by_id = {item["id"]: item for item in data}
image_dir = "/home/macierz/s184780/MGR/splits/test/images"

# Przygotowanie 5 przykładów few-shot z danych
few_shot_examples = []
for item in data[:5]:  # Pierwsze 5 przykładów
    lines = []
    for bbox, label in zip(item["bbox"], item["transcription"]):
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["width"])
        h = int(bbox["height"])
        bbox_str = f"<bbox_{x}_{y}_{x+w}_{y+h}>"
        lines.append(f"{bbox_str} -> {label}")
    few_shot_examples.append("Przykład:\n" + "\n".join(lines))
few_shot_prompt = "\n\n".join(few_shot_examples)

errors_path = "errors.jsonl"
error_log = open(errors_path, "w", encoding="utf-8")

wer_total = 0
cer_total = 0
ler_total = 0
count = 0

# Maksymalna długość dla modelu to 1024 tokeny
max_tokens = 1024

# Funkcja dzieląca na mniejsze sekwencje, jeśli są zbyt długie
def split_into_chunks(text, max_tokens, processor, image):
    inputs = processor(text=text, images=image, return_tensors="pt", truncation=True, padding="max_length", max_length=max_tokens)
    token_count = inputs["input_ids"].shape[1]  # Liczba tokenów w wejściu
    if token_count > max_tokens:
        # Jeśli tekst po przetworzeniu przekroczył limit, dzielimy go na mniejsze fragmenty
        print(f"Tekst za długi, dzielimy na fragmenty...")
        chunks = [text[i:i+max_tokens] for i in range(0, len(text), max_tokens)]
        return chunks
    else:
        return [text]

for file_name in tqdm(islice(os.listdir(image_dir), 15)):
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

    # Wydobycie regionów bbox
    bboxes = data_by_id[file_id]["bbox"]

    # Tworzenie promptu z aktualnymi bboxami
    region_prompts = []
    for bbox in bboxes:
        x = int(bbox["x"])
        y = int(bbox["y"])
        w = int(bbox["width"])
        h = int(bbox["height"])
        region_prompts.append(f"<bbox_{x}_{y}_{x+w}_{y+h}> ->")

    region_prompts_str = "\n".join(region_prompts)
    prompt = f"""{few_shot_prompt}

Teraz przetwórz regiony obrazu:
{region_prompts_str}
"""

    task = "<REGION_TO_DESCRIPTION>"

    # Sprawdzanie długości i dzielenie na mniejsze fragmenty, jeśli potrzebne
    chunks = split_into_chunks(prompt, max_tokens, processor, image)

    # Teraz przetwarzamy każdy fragment
    for chunk in chunks:
        inputs = processor(text=chunk, images=image, return_tensors="pt").to(device, torch_dtype)

        try:
            # Generowanie odpowiedzi
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3  # Liczba beams pozostaje bez zmian
            )
        except RuntimeError as e:
            print(f"Błąd przy generowaniu tekstu: {e}")
            continue

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

        if predicted_lines != expected_transcription:
            error_table.add_data(
                file_id,
                image_path,
                "\n".join(expected_transcription),
                "\n".join(predicted_lines)
            )
            json.dump({
                "file_id": file_id,
                "file_name": file_name,
                "wer": wer_score,
                "cer": cer_score,
                "ler": ler_score,
                "expected": expected_transcription,
                "predicted": predicted_lines
            }, error_log, ensure_ascii=False)
            error_log.write("\n")

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
    "ocr_errors": error_table
})

error_log.close()
wandb.finish()
