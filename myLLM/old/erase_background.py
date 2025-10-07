import os
import json
from PIL import Image
from tqdm import tqdm

# Ścieżki
image_dir = "/home/macierz/s184780/MGR/splits/test_original/images"
json_path = "/home/macierz/s184780/MGR/splits/test_original/data.json"
output_dir = "/home/macierz/s184780/MGR/splits/test_original/masked_output"
os.makedirs(output_dir, exist_ok=True)

# Wczytanie danych JSON
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Indeksowanie po ID
data_by_id = {str(item["id"]).zfill(4): item for item in data} 

# Główna pętla po obrazach
for file_name in tqdm(os.listdir(image_dir)):
    if not file_name.endswith(".jpg"):
        continue

    file_stem = os.path.splitext(file_name)[0]

    if file_stem not in data_by_id:
        print(f"Brak danych OCR dla pliku: {file_name}")
        continue

    try:
        # Wczytaj obraz
        img_path = os.path.join(image_dir, file_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Utwórz czarny obraz o tych samych rozmiarach
        masked_img = Image.new("RGB", (width, height), (255, 255, 255))

        # Iteracja po bboxach
        for bbox in data_by_id[file_stem]["bbox"]:
            # Zamiana współrzędnych procentowych na piksele
            x = int((bbox["x"] / 100) * width)
            y = int((bbox["y"] / 100) * height)
            w = int((bbox["width"] / 100) * width)
            h = int((bbox["height"] / 100) * height)

            # Wycięcie i wklejenie fragmentu
            region = img.crop((x, y, x + w, y + h))
            masked_img.paste(region, (x, y))

        # Zapis wyniku
        masked_img.save(os.path.join(output_dir, file_name))

    except Exception as e:
        print(f"Błąd przetwarzania {file_name}: {e}")