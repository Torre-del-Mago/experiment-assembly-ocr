import os
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, base_path, processor, max_target_length=128):
        """
        OCR Dataset class that loads images from a directory and transcriptions from a CSV file.

        Args:
            base_path (str): Base path containing the images and the CSV file.
            processor (TrOCRProcessor): TrOCR processor for processing images and transcriptions.
            max_target_length (int): Maximum length for target sequences (default: 128).
        """
        self.base_path = base_path
        self.image_dir = os.path.join(base_path, "images")
        self.csv_file = os.path.join(base_path, "labels.csv")
        
        self.processor = processor
        self.max_target_length = max_target_length

        # Load data from the CSV file
        self.data = []
        with open(self.csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter=',')
            for row in reader:
                image_path, transcription = row
                self.data.append({
                    "image_path": os.path.join(self.image_dir, image_path),
                    "transcription": transcription.strip()
                })

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns an image and its transcription in a format suitable for the model.

        Args:
            idx (int): Index of the sample in the dataset.

        Returns:
            dict: Dictionary containing pixel values and labels.
        """
        # Get the image file path and transcription
        entry = self.data[idx]
        image_path = entry["image_path"]
        transcription = entry["transcription"]

        # Prepare the image (resize and normalize)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Encode the transcription into input IDs
        labels = self.processor.tokenizer(
            transcription,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True
        ).input_ids

        # Ensure PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        
        # Return as a dictionary
        encoding = {
            "pixel_values": pixel_values.squeeze(),  # Remove extra batch dimension
            "labels": torch.tensor(labels)  # Convert labels to tensor
        }
        return encoding