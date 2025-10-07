import torch
import os
import json
from tqdm import tqdm
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import re
from metrics import calculate_error_rates

def save_results_to_json(results, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_path}")

def load_dataset(dataset_path):
    data_json_path = os.path.join(dataset_path, "data.json")
    
    with open(data_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    file_names = []
    transcriptions = []

    for sample in data:
        filename = sample['ocr'].split('/')[-1]
        file_names.append(os.path.join(dataset_path, "images", filename))
        transcriptions.append(sample['transcription'])

    print(f"Loaded {len(file_names)} samples from dataset")
    return file_names, transcriptions

def evaluate_model(model, processor, dataset_path, device, sanity_check=False):
    results = []
    total_processing_time = 0
    cer_list, wer_list, ler_list = [], [], []
    file_names, total_transcriptions = load_dataset(dataset_path)

    task_prompt = "<ocr>"

    for i, (img, transcriptions) in enumerate(tqdm(zip(file_names, total_transcriptions), desc="Ewaluacja Donut", total=len(file_names))):
        if sanity_check and i > 0:
            break

        image = Image.open(img).convert("RGB")

        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()

        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

        pixel_values = processor(image, return_tensors="pt").pixel_values

        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_position_embeddings,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
        text = processor.token2json(sequence)['text_sequence']
        predictions = [chunk.strip() for chunk in text.split('<sep/>') if chunk.strip()]

        end_time.record()
        torch.cuda.synchronize()
        processing_time = start_time.elapsed_time(end_time) / 1000.0
        total_processing_time += processing_time

        metrics = calculate_error_rates(transcriptions, predictions)
        cer_list.append(metrics['CER']['rate'])
        wer_list.append(metrics['WER']['rate'])
        ler_list.append(metrics['LER']['rate'])

        result_entry = {
            "filename": os.path.basename(img),
            "prediction": predictions,
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
        "model_name": model.name_or_path if hasattr(model, "name_or_path") else "",
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

def process_eval_test(model_name, processor_name, dataset_path, sanity_check=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    processor = DonutProcessor.from_pretrained(processor_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    summary = evaluate_model(model, processor, dataset_path, device, sanity_check=sanity_check)

    results_dir = "result"
    os.makedirs(results_dir, exist_ok=True)
    prefix = "sanity_" if sanity_check else ""
    output_filename = f"{prefix}results_eval_donut_{os.path.basename(model_name)}.json"
    output_path = os.path.join(results_dir, output_filename)
    save_results_to_json(summary, output_path)

    print(f"\n=== STATYSTYKI CZASOWE ===")
    print(f"Łączny czas przetwarzania: {summary['total_processing_time']:.3f} sekund")
    print(f"Średni czas na próbkę: {summary['average_processing_time']:.3f} sekund")
    print(f"Liczba przetworzonych próbek: {summary['total_samples']}")
    print(f"Średni CER: {summary['average_CER']:.3f}")
    print(f"Średni WER: {summary['average_WER']:.3f}")
    print(f"Średni LER: {summary['average_LER']:.3f}")

if __name__ == "__main__":
    processor_name = "result/a3bd0748-f642-4fb2-ac0e-71962fc26016/processor"
    dataset_path = "data/my_dataset/test"
    for model_name in [
        "result/48f75661-77f1-4314-a3f1-f0382bb0ac3f/checkpoint_epoch_2"
    ]:
        process_eval_test(model_name, processor_name, dataset_path, sanity_check=True)