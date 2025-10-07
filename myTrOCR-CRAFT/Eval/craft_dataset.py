import os
import json
from torch.utils.data import Dataset
from craft_hw_ocr.OCR import * 

class CraftDataset(Dataset):
    def __init__(self, base_path, split="training"):
        """
        Dataset do trenowania modelu CRAFT, przyjmującego obrazy i dane z pliku JSON.
        
        Args:
            base_path (str): Ścieżka do bazowego folderu, z którego tworzymy ścieżki.
            split (str, optional): Wartość "train" lub "val" określająca, czy używamy danych treningowych czy walidacyjnych.
            transform (callable, optional): Funkcja transformująca obrazy, np. augmentacja.
        """
        self.base_path = base_path
        self.split = split

        self.image_dir = os.path.join(self.base_path, split, "images")
        self.json_file = os.path.join(self.base_path, split, f"data.json")

        with open(self.json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = os.path.basename(self.data[idx]["ocr"])
        img = load_image(os.path.join(self.image_dir, image_path))
        
        if img is None:
            raise ValueError(f"Image at index {idx} is None. Check your data loading process. Path to image {image_path}. My base path: {self.image_dir} ")

        transcription = self.data[idx]["transcription"]

        return img, transcription
