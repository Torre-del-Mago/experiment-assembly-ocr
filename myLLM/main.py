from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from zero_shot import zero_shot_prediction
import os
import json
from tqdm import tqdm
import time
from few_shot import few_shot_prediction
import random
from metrics import calculate_error_rates

def save_results_to_json(results, output_path):
    """
    Saves results to a JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_path}")

def load_dataset(dataset_path):
    """
    Loads the dataset and returns two lists: filenames and transcriptions
    """
    data_json_path = os.path.join(dataset_path, "data.json")
    
    with open(data_json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Two simple lists
    filenames = []
    transcriptions = []
    
    for sample in data:
        # Extract filename from path "images/filename.jpg"
        filename = sample['ocr'].split('/')[-1]
        filenames.append(os.path.join(dataset_path, "images", filename))
        transcriptions.append(sample['transcription'])  # List of strings
    
    print(f"Loaded {len(filenames)} samples from dataset")
    return filenames, transcriptions

def result_to_list(result):
    """
    Converts result (string with assembler code) to a list of strings
    Removes lines containing ```
    """
    if not result:
        return []
    
    lines = []
    for line in result.split('\n'):
        cleaned_line = line.strip()
        # Add only non-empty lines that do not contain ```
        if cleaned_line and '```' not in cleaned_line:
            lines.append(cleaned_line)
    
    return lines

def process_one_shot_test(model_name, dataset_path, sanity_check=False):

    # Model and processor
    if "AWQ" in model_name:
        # For AWQ models, try to force bf16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Model is on device: {model.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")


    filenames, transcriptions = load_dataset(os.path.join(dataset_path, "test"))
    if sanity_check:
        print("=== SANITY CHECK MODE - processing only the first sample ===")
        filenames = filenames[:1]
        transcriptions = transcriptions[:1]

    results = []
    total_processing_time = 0
    cer_list, wer_list, ler_list = [], [], []

    # Zero-shot prediction
    for image_path, transcription in tqdm(zip(filenames, transcriptions), total=len(filenames), desc=f"Zero-shot prediction - {model_name}"):
        start_time = time.time()
        result = zero_shot_prediction(model, processor, image_path, max_new_tokens=2000)
        end_time = time.time()
        processing_time = end_time - start_time
        total_processing_time += processing_time
        predicted_lines = result_to_list(result)

        # Calculate metrics for this sample
        metrics = calculate_error_rates(transcription, predicted_lines)
        cer_list.append(metrics['CER']['rate'])
        wer_list.append(metrics['WER']['rate'])
        ler_list.append(metrics['LER']['rate'])

        result_entry = {
            "filename": os.path.basename(image_path),
            "prediction": predicted_lines,
            "expected": transcription,
            "processing_time": round(processing_time, 3),
            "metrics": metrics
        }
        results.append(result_entry)

    avg_processing_time = total_processing_time / len(results) if results else 0
    avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0
    avg_wer = sum(wer_list) / len(wer_list) if wer_list else 0
    avg_ler = sum(ler_list) / len(ler_list) if ler_list else 0

    summary = {
        "model_name": model_name,
        "sanity_check": sanity_check,
        "total_samples": len(results),
        "total_processing_time": round(total_processing_time, 3),
        "average_processing_time": round(avg_processing_time, 3),
        "average_CER": avg_cer,
        "average_WER": avg_wer,
        "average_LER": avg_ler,
        "results": results

    }

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    prefix = "sanity_" if sanity_check else ""
    output_filename = f"{prefix}results_{model_name.replace('/', '_')}.json"
    output_path = os.path.join(results_dir, output_filename)
    save_results_to_json(summary, output_path)

    print(f"\n=== TIME STATISTICS ===")
    if sanity_check:
        print("*** SANITY CHECK MODE ***")
    print(f"Total processing time: {total_processing_time:.3f} seconds")
    print(f"Average time per sample: {avg_processing_time:.3f} seconds")
    print(f"Number of processed samples: {len(results)}")
    print(f"Average CER: {avg_cer:.4f}, WER: {avg_wer:.4f}, LER: {avg_ler:.4f}")

def process_few_shot_test(model_name, dataset_path, few_shot_count=5, few_shot_set="train", sanity_check=False):
    """
    Tests the model with few-shot learning
    
    Args:
        model_name: name of the model to load
        dataset_path: path to the dataset
        few_shot_count: number of examples for few-shot learning
        few_shot_set: set to take examples from ("train" or "val")
        sanity_check: if True, processes only the first sample
    """
    
    if "AWQ" in model_name:
        # For AWQ models, try to force bf16
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Model is on device: {model.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")

    # Load test set
    test_filenames, test_transcriptions = load_dataset(os.path.join(dataset_path, "test"))
    
    # Load set for few-shot examples
    few_shot_filenames, few_shot_transcriptions = load_dataset(os.path.join(dataset_path, few_shot_set))
    
    # Sanity check - only first test sample
    if sanity_check:
        print("=== SANITY CHECK MODE - processing only the first test sample ===")
        test_filenames = test_filenames[:1]
        test_transcriptions = test_transcriptions[:1]
    
    # Select first few_shot_count examples
    print(f"Using {few_shot_count} random examples from set '{few_shot_set}' for few-shot learning")

    results = []
    total_processing_time = 0
    cer_list, wer_list, ler_list = [], [], []

    # Few-shot prediction
    for image_path, transcription in tqdm(zip(test_filenames, test_transcriptions), 
                                         total=len(test_filenames), 
                                         desc=f"Few-shot prediction ({few_shot_count} random examples) - {model_name}"):

        # Select random examples for each iteration, making sure image_path is not in path_example_images
        while True:
            idxs = random.sample(range(len(few_shot_filenames)), few_shot_count)
            path_example_images = [few_shot_filenames[i] for i in idxs]
            if image_path not in path_example_images:
                break
            else:
                print(f"Trying again because {image_path} is among the few-shot examples")
        example_transcription = [few_shot_transcriptions[i] for i in idxs]

        start_time = time.time()
        result = few_shot_prediction(model, processor, image_path, path_example_images, example_transcription, max_new_tokens=2000)
        end_time = time.time()
        processing_time = end_time - start_time
        total_processing_time += processing_time
        predicted_lines = result_to_list(result)

        # Calculate metrics for this sample
        metrics = calculate_error_rates(transcription, predicted_lines)
        cer_list.append(metrics['CER']['rate'])
        wer_list.append(metrics['WER']['rate'])
        ler_list.append(metrics['LER']['rate'])

        result_entry = {
            "filename": os.path.basename(image_path),
            "prediction": predicted_lines,
            "expected": transcription,
            "processing_time": round(processing_time, 3),
            "metrics": metrics
        }
        results.append(result_entry)

    avg_processing_time = total_processing_time / len(results) if results else 0
    avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0
    avg_wer = sum(wer_list) / len(wer_list) if wer_list else 0
    avg_ler = sum(ler_list) / len(ler_list) if ler_list else 0

    summary = {
        "model_name": model_name,
        "few_shot_count": few_shot_count,
        "few_shot_set": few_shot_set,
        "sanity_check": sanity_check,
        "total_samples": len(results),
        "total_processing_time": round(total_processing_time, 3),
        "average_processing_time": round(avg_processing_time, 3),
        "average_CER": avg_cer,
        "average_WER": avg_wer,
        "average_LER": avg_ler,
        "results": results
    }

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    model_safe_name = model_name.replace('/', '_')
    prefix = "sanity_" if sanity_check else ""
    output_filename = f"{prefix}results_few_shot_{few_shot_count}_{few_shot_set}_{model_safe_name}.json"
    output_path = os.path.join(results_dir, output_filename)
    save_results_to_json(summary, output_path)

    print(f"\n=== TIME STATISTICS (FEW-SHOT) ===")
    if sanity_check:
        print("*** SANITY CHECK MODE ***")
    print(f"Number of few-shot examples: {few_shot_count}")
    print(f"Few-shot set: {few_shot_set}")
    print(f"Total processing time: {total_processing_time:.3f} seconds")
    print(f"Average time per sample: {avg_processing_time:.3f} seconds")
    print(f"Number of processed samples: {len(results)}")
    print(f"Average CER: {avg_cer:.4f}, WER: {avg_wer:.4f}, LER: {avg_ler:.4f}")

if __name__ == "__main__":
    dataset_path = "dataset/my_dataset"

    for model_name in [
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ]:
        process_one_shot_test(model_name, dataset_path, sanity_check=True)
        process_few_shot_test(model_name, dataset_path, few_shot_count=5, few_shot_set="train", sanity_check=True)

