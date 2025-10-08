import gradio as gr
import torch
import gc
import os
import sys
import numpy as np
from PIL import Image
import re
import tempfile
from pathlib import Path
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    DonutProcessor,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor
)

# Dodaj ścieżki do modułów projektu
sys.path.append(os.path.join(os.path.dirname(__file__), 'myTrOCR-CRAFT/Eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'myTrOCR-CRAFT/Eval/craft_text_detector'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'myTrOCR-CRAFT/Eval/craft_hw_ocr'))

try:
    from craft_text_detector import Craft
    from craft_hw_ocr import OCR
    CRAFT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: CRAFT modules not available: {e}")
    CRAFT_AVAILABLE = False

try:
    from qwen_vl_utils import process_vision_info
    QWEN_AVAILABLE = True
except ImportError:
    print("Warning: qwen_vl_utils not available. Qwen functionality will be limited.")
    QWEN_AVAILABLE = False

# Ścieżki do modeli
CRAFT_MODEL_PATH = "models/CRAFT/CRAFT_clr_amp_25.pth"
DONUT_MODEL_PATH = "models/Donut"
TROCR_MODEL_NAME = "microsoft/trocr-base-handwritten"
QWEN_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Globalne modele (cache)
models_cache = {
    "craft": None,
    "trocr": None,
    "trocr_processor": None,
    "donut": None,
    "donut_processor": None,
    "qwen": None,
    "qwen_processor": None
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Używane urządzenie: {device}")


def load_trocr_craft_models():
    """Ładuje modele TrOCR i CRAFT"""
    if not CRAFT_AVAILABLE:
        return None, None, None
    
    if models_cache["trocr"] is None:
        print("Ładowanie modeli TrOCR + CRAFT...")
        models_cache["trocr_processor"] = TrOCRProcessor.from_pretrained(TROCR_MODEL_NAME)
        models_cache["trocr"] = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_NAME)
        models_cache["trocr"].to(device)
        models_cache["trocr"].eval()
        
        models_cache["craft"] = Craft(
            output_dir=None,
            crop_type="poly",
            refiner=True,
            export_extra=False,
            link_threshold=0.1,
            text_threshold=0.3,
            cuda=torch.cuda.is_available(),
            weight_path_craft_net=CRAFT_MODEL_PATH
        )
        print("Modele TrOCR + CRAFT załadowane!")
    
    return models_cache["craft"], models_cache["trocr"], models_cache["trocr_processor"]


def _move_model_to_cpu(key):
    """Move a model in models_cache to CPU and delete if possible to free GPU memory."""
    try:
        m = models_cache.get(key)
        if m is None:
            return
        # If it's a HuggingFace model or torch.nn.Module, move it to cpu
        if hasattr(m, 'to'):
            try:
                m.to('cpu')
            except Exception:
                pass
        # Delete reference
        models_cache[key] = None
    except Exception:
        pass


def load_donut_model():
    """Ładuje model Donut"""
    if models_cache["donut"] is None:
        print("Ładowanie modelu Donut...")
        models_cache["donut_processor"] = DonutProcessor.from_pretrained(DONUT_MODEL_PATH)
        models_cache["donut"] = VisionEncoderDecoderModel.from_pretrained(DONUT_MODEL_PATH)
        models_cache["donut"].to(device)
        models_cache["donut"].eval()
        print("Model Donut załadowany!")
    
    return models_cache["donut"], models_cache["donut_processor"]


def load_qwen_model():
    """Ładuje model Qwen"""
    if models_cache["qwen"] is None:
        print("Ładowanie modelu Qwen (może to chwilę potrwać)...")
        models_cache["qwen"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        models_cache["qwen_processor"] = AutoProcessor.from_pretrained(QWEN_MODEL_NAME)
        print("Model Qwen załadowany!")
    
    return models_cache["qwen"], models_cache["qwen_processor"]


def process_with_trocr_craft(image):
    """Przetwarzanie obrazu z użyciem TrOCR + CRAFT"""
    try:
        if not CRAFT_AVAILABLE:
            return "Błąd: Moduły CRAFT nie są dostępne. Zainstaluj wymagane zależności.", None
        
        craft, trocr_model, trocr_processor = load_trocr_craft_models()
        
        # Konwersja obrazu PIL do numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Detekcja tekstu
        img, detection_results = OCR.detection(img_array, craft)
        
        # Rozpoznawanie tekstu
        bboxes, recognized_texts = OCR.recoginition(
            img, 
            detection_results, 
            trocr_processor, 
            trocr_model, 
            device
        )
        
        # Przygotowanie wyniku
        result_text = "\n".join(recognized_texts)
        
        # Wizualizacja bbox-ów
        img_with_boxes = OCR.visualize(img.copy(), detection_results)
        
        return result_text, img_with_boxes
    
    except Exception as e:
        return f"Błąd podczas przetwarzania: {str(e)}", None


def process_with_donut(image):
    """Przetwarzanie obrazu z użyciem Donut"""
    try:
        model, processor = load_donut_model()
        
        # Konwersja do PIL Image jeśli potrzeba
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        image = image.convert("RGB")
        
        task_prompt = "<ocr>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).input_ids
        
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
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(
            processor.tokenizer.pad_token, ""
        )
        sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
        
        text = processor.token2json(sequence)['text_sequence']
        predictions = [chunk.strip() for chunk in text.split('<sep/>') if chunk.strip()]
        
        result_text = "\n".join(predictions)
        
        return result_text, image
    
    except Exception as e:
        return f"Błąd podczas przetwarzania: {str(e)}", None


def process_with_qwen(image):
    """Przetwarzanie obrazu z użyciem Qwen (zero-shot)"""

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
    
    try:
        if not QWEN_AVAILABLE:
            return "Błąd: qwen_vl_utils nie jest dostępny. Zainstaluj qwen-vl-utils.", None
        
        model, processor = load_qwen_model()
        # Konwersja do PIL Image i zapis tymczasowy (bezpieczny, przenośny)
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_image_path = os.path.abspath(temp_file.name)
        temp_file.close()
        image.save(temp_image_path)
        file_uri = Path(temp_image_path).as_uri()

        # Przygotowanie promptu
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": file_uri},
                {"type": "text", "text": 
                    "Tylko odczytaj kod assemblera architektury x86, nie komentuj go. W kodzie mogą znajdować się \n" +
                    "nazwy funkcji zdefiniowanych i zadeklarowanych w kodzie.\n" +
                    "Wszystkie mnemoniki (instrukcje asemblera), np. `mov`, `jmp`, `cmp`, itp.\n" +
                    "Wszystkie etykiety (np. nazwy funkcji, punkty skoków, etykiety BEGIN/END, itp.)\n" +
                    "Wszystkie występujące rejestry, np. `eax`, `ebx`, `xmm0`, `st3`, itp.\n" +
                    "Słowa kluczowe asemblera, np. `ptr`, `byte`, `dword`, `near`, `extern`, `public`, `PROC`, `ENDP`, `END`.\n" +
                    "Kod powinien być w formacie np:\n" +
                    "```\n"
                    "mov ebx, ecx\n" +
                    "sub esp, 32\n" +
                    "etykieta:\n" +
                    "add esp, 8\n" +
                    ".itd\n" +
                    "```\n" +
                    "Wypisz kod, bez dodatkowych informacji.\n"
                }
            ]}
        ]
        
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)
        
        generated_ids = model.generate(**inputs, max_new_tokens=2000)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        output = "\n".join(result_to_list(output))
        
        try:
            return output, image
        finally:
            if os.path.exists(temp_image_path):
                try:
                    os.remove(temp_image_path)
                except Exception:
                    pass
    
    except Exception as e:
        return f"Błąd podczas przetwarzania: {str(e)}", None


def process_image(image, model_choice):
    """Główna funkcja przetwarzania obrazu"""
    if image is None:
        return "Proszę wczytać obraz.", None

    # Przed załadowaniem nowego modelu - zwolnij GPU z innych modeli
    try:
        # Move other models to CPU / delete them to free GPU
        if model_choice == "TrOCR + CRAFT":
            _move_model_to_cpu('donut')
            _move_model_to_cpu('donut_processor')
            _move_model_to_cpu('qwen')
            _move_model_to_cpu('qwen_processor')
        elif model_choice == "Donut":
            _move_model_to_cpu('trocr')
            _move_model_to_cpu('trocr_processor')
            _move_model_to_cpu('craft')
            _move_model_to_cpu('qwen')
            _move_model_to_cpu('qwen_processor')
        elif model_choice == "Qwen (Zero-shot)":
            _move_model_to_cpu('trocr')
            _move_model_to_cpu('trocr_processor')
            _move_model_to_cpu('craft')
            _move_model_to_cpu('donut')
            _move_model_to_cpu('donut_processor')
    except Exception:
        pass

    # Force garbage collection and empty CUDA cache to free GPU memory
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    
    if model_choice == "TrOCR + CRAFT":
        return process_with_trocr_craft(image)
    elif model_choice == "Donut":
        return process_with_donut(image)
    elif model_choice == "Qwen (Zero-shot)":
        return process_with_qwen(image)
    else:
        return "Nieznany model.", None


# Interfejs Gradio
with gr.Blocks(title="Assembly OCR - Rozpoznawanie kodu Assembly") as demo:
    gr.Markdown("# 🔍 Assembly OCR - Rozpoznawanie kodu Assembly")
    gr.Markdown("Wybierz model OCR i wczytaj obraz z kodem assembly do rozpoznania.")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Obraz wejściowy", 
                type="pil",
                height=400
            )
            model_choice = gr.Radio(
                choices=["TrOCR + CRAFT", "Donut", "Qwen (Zero-shot)"],
                label="Wybierz model",
                value="TrOCR + CRAFT"
            )
            process_btn = gr.Button("🚀 Rozpoznaj tekst", variant="primary")
        
        with gr.Column():
            text_output = gr.Textbox(
                label="Rozpoznany tekst",
                lines=20,
                placeholder="Tutaj pojawi się rozpoznany kod assembly..."
            )
            image_output = gr.Image(
                label="Obraz z oznaczeniami",
                height=400
            )
    
    gr.Markdown("""
    ### ℹ️ Informacje o modelach:
    - **TrOCR + CRAFT**: Detekcja tekstu za pomocą CRAFT + rozpoznawanie TrOCR
    - **Donut**: End-to-end model transformerowy bez OCR
    - **Qwen (Zero-shot)**: Multimodalny model językowy w trybie zero-shot
    
    ### 📝 Uwagi:
    - Pierwsze użycie modelu może trwać dłużej (ładowanie modelu)
    - Qwen wymaga najwięcej pamięci GPU
    - Dla najlepszych wyników używaj obrazów o dobrej jakości
    """)
    
    process_btn.click(
        fn=process_image,
        inputs=[image_input, model_choice],
        outputs=[text_output, image_output]
    )
    
    # Przykłady
    gr.Markdown("### 📸 Przykłady")
    gr.Markdown("Możesz użyć własnych obrazów lub przetestować na przykładowych plikach z datasetu.")

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
