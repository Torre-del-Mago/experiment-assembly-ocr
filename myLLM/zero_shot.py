from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def zero_shot_prediction(model, processor, image_path, max_new_tokens=512):
    """
    Performs zero-shot prediction on an image containing assembler code.
    
    Args:
        model: Qwen2.5-VL model
        processor: AutoProcessor
        image_path: Path to the image
        max_new_tokens: Maximum number of new tokens to generate
    
    Returns:
        str: Recognized assembler code
    """
    # Zapytanie multimodalne
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": "Tylko odczytaj kod assemblera architektury x86, nie komentuj go. W kodzie mogą znajdować się \n" +
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
            "Wypisz kod, bez dodatkowych informacji.\n"}
        ]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return output

if __name__ == "__main__":
    model_name = "Qwen/Qwen2.5-VL-32B-Instruct"

    # Model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Model jest na urządzeniu: {model.device}")
    print(f"CUDA dostępne: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Aktualne urządzenie CUDA: {torch.cuda.current_device()}")
        print(f"Nazwa karty graficznej: {torch.cuda.get_device_name()}")

    # path to image
    image_path = "/home/dyplom/Dataset/my_dataset/test/images/fd4e5d78-189507_zadanie_5.jpg"

    text = zero_shot_prediction(model, processor, image_path, max_new_tokens=2000)

    print(text)