import json
import os
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor

class FlorenceDataset(Dataset):
    def __init__(self, json_path, processor_name="microsoft/Florence-2-large-ft", task_token="<OCR_WITH_REGION>", max_length=1024):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
        self.task_token = task_token
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["ocr"]  # zakładamy że pole "ocr" to ścieżka do obrazu
        image = Image.open(image_path).convert("RGB")

        # Spłaszczamy transkrypcję do pojedynczego ciągu (z enterami)
        transcription_lines = item["transcription"]
        transcription_text = "\n".join(transcription_lines)

        # Tokenizacja wejścia (prompt tylko z tokenem zadania)
        inputs = self.processor(
            text=self.task_token,
            images=image,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length
        )

        # Tokenizacja etykiety
        with self.processor.as_target_processor():
            labels = self.processor(
                text=transcription_text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length
            ).input_ids

        # Przekształć dane do formatu batchowego (usuń pierwszy wymiar [1, ...] -> [...])
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        labels = labels.squeeze(0)

        return {
            **inputs,
            "labels": labels,
            "id": item.get("id", -1),
            "path": image_path,
            "text": transcription_text
        }
