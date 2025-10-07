import torch
import os
import json
from tqdm import tqdm
from craft_hw_ocr import OCR
from craft_text_detector import Craft
from craft_dataset import CraftDataset
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from metrics import calculate_error_rates

def save_results_to_json(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wyniki zapisane do: {output_path}")

def load_dataset(dataset_path):
    """
    Ładuje dataset CraftDataset i zwraca listę (img, transcriptions)
    """
    dataset = CraftDataset(base_path=dataset_path, split="test")
    print(f"Wczytano {len(dataset)} próbek z datasetu")
    return dataset

def evaluate_model(craft, trocr_model, craft_val_dataset, trocr_processor, device, sanity_check=False):
    results = []
    total_processing_time = 0
    cer_list, wer_list, ler_list = [], [], []

    for i, (img, transcriptions) in enumerate(tqdm(craft_val_dataset, desc="Ewaluacja Craft+TrOCR")):
        if sanity_check and i > 0:
            break

        filename = os.path.basename(craft_val_dataset.data[i]["ocr"])

        img = img.to(device) if isinstance(img, torch.Tensor) else img

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        # 1. Detekcja tekstu
        img, detection_results = OCR.detection(img, craft)

        # 2. Rozpoznawanie tekstu z obsługą błędów
        try:
            bboxes, recognized_texts = OCR.recoginition(img, detection_results, trocr_processor, trocr_model, device)
        except Exception as e:
            print(f"Błąd OCR dla pliku {filename}: {e}")
            bboxes, recognized_texts = detection_results.get('boxes', []), [f"ERROR: {str(e)}"]

        end_time.record()
        torch.cuda.synchronize()
        processing_time = start_time.elapsed_time(end_time) / 1000.0  # sekundy
        total_processing_time += processing_time

        metrics = calculate_error_rates(transcriptions, recognized_texts)
        cer_list.append(metrics['CER']['rate'])
        wer_list.append(metrics['WER']['rate'])
        ler_list.append(metrics['LER']['rate'])

        result_entry = {
            "filename": filename,
            "prediction": recognized_texts,
            "expected": transcriptions,
            "processing_time": round(processing_time, 3),
            "metrics": metrics
        }
        results.append(result_entry)

    avg_processing_time = total_processing_time / len(results) if results else 0
    avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0
    avg_wer = sum(wer_list) / len(wer_list) if wer_list else 0
    avg_ler = sum(ler_list) / len(ler_list) if ler_list else 0
    summary = {
        "model_name": trocr_model.name_or_path if hasattr(trocr_model, "name_or_path") else "",
        "sanity_check": sanity_check,
        "total_samples": len(results),
        "total_processing_time": round(total_processing_time, 3),
        "average_processing_time": round(avg_processing_time, 3),
        "average_CER": avg_cer,
        "average_WER": avg_wer,
        "average_LER": avg_ler,
        "results": results
    }
    return summary

def process_eval_test(trocr_model_name, craft_model_path, dataset_path, sanity_check=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    craft = Craft(output_dir=None, crop_type="poly", refiner=True, export_extra=False, link_threshold=0.1, text_threshold=0.3, cuda=torch.cuda.is_available(), weight_path_craft_net=craft_model_path)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained(trocr_model_name)
    model.to(device)
    model.eval()

    craft_val_dataset = load_dataset(dataset_path)

    summary = evaluate_model(craft, model, craft_val_dataset, processor, device, sanity_check=sanity_check)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    prefix = "sanity_" if sanity_check else ""
    output_filename = f"{prefix}results_eval_{trocr_model_name.replace('/', '_')}_{craft_model_path.replace('/', '_') if craft_model_path is not None else 'None'}.json"
    output_path = os.path.join(results_dir, output_filename)
    save_results_to_json(summary, output_path)

    print(f"\n=== STATYSTYKI CZASOWE ===")
    print(f"Łączny czas przetwarzania: {summary['total_processing_time']:.3f} sekund")
    print(f"Średni czas na próbkę: {summary['average_processing_time']:.3f} sekund")
    print(f"Liczba przetworzonych próbek: {summary['total_samples']}")

if __name__ == "__main__":
    model_name = "microsoft/trocr-base-handwritten"
    dataset_path = "data/my_dataset"
    for trocr_model_path in ["microsoft/trocr-base-handwritten"]:
        for craft_model_path in [None]:
            process_eval_test(trocr_model_path, craft_model_path, dataset_path, sanity_check=True)