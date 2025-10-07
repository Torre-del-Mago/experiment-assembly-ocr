from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def few_shot_prediction(model, processor, image_path, path_example_images, example_transcription, max_new_tokens=512):
    """
    Performs few-shot prediction on an image containing assembler code using examples.
    
    Args:
        model: Qwen2.5-VL model
        processor: AutoProcessor
        image_path: Path to the image to analyze
        path_example_images: List of paths to example images
        example_transcription: List of transcriptions corresponding to the example images
        max_new_tokens: Maximum number of new tokens to generate
    
    Returns:
        str: Recognized assembler code
    """
    # Check if the number of examples matches
    if len(path_example_images) != len(example_transcription):
        raise ValueError("The number of example images must be equal to the number of transcriptions")
    
    # Build messages with examples
    messages = []
    
    # Add examples
    for i, (example_image_path, transcription) in enumerate(zip(path_example_images, example_transcription)):
        # Example - user image
        messages.append({
            "role": "user", 
            "content": [
                {"type": "image", "image": f"file://{example_image_path}"},
                {"type": "text", "text": "Odczytaj kod assemblera architektury x86 z tego obrazu."}
            ]
        })
        
    # Example - assistant's response
        messages.append({
            "role": "assistant",
            "content": "\n".join(transcription) if isinstance(transcription, list) else transcription
        })
    
    # Add the main query
    messages.append({
        "role": "user", 
        "content": [
            {"type": "image", "image": f"file://{image_path}"},
            {"type": "text", "text": "Tylko odczytaj kod assemblera architektury x86, nie komentuj go. W kodzie mogą znajdować się \n" +
            "nazwy funkcji zdefiniowanych i zadeklarowanych w kodzie.\n" +
            "Wszystkie mnemoniki (instrukcje asemblera), np. `mov`, `jmp`, `cmp`, itp.\n" +
            "Wszystkie etykiety (np. nazwy funkcji, punkty skoków, etykiety BEGIN/END, itp.)\n" +
            "Wszystkie występujące rejestry, np. `eax`, `ebx`, `xmm0`, `st3`, itp.\n" +
            "Słowa kluczowe asemblera, np. `ptr`, `byte`, `dword`, `near`, `extern`, `public`, `PROC`, `ENDP`, `END`.\n" +
            "Kod powinien być w formacie np:\n" +
            "```\n"
            ".code\n" +
            "sub esp, 32\n" +
            "etykieta:\n" +
            "add esp, 8\n" +
            ".itd\n" +
            "```\n" +
            "Wypisz kod, bez dodatkowych informacji.\n"}
        ]
    })

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
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

    # Model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    print(f"Model is on device: {model.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")

    # Path to image
    image_path = "/home/dyplom/Dataset/my_dataset/test/images/f05b2abd-197998_zadanie_7.jpg"

    path_example_images = [
        "/home/dyplom/Dataset/my_dataset/test/images/1ab97781-198025_zadanie_4.jpg",
        "/home/dyplom/Dataset/my_dataset/test/images/1ca6a782-226C.jpg",
        "/home/dyplom/Dataset/my_dataset/test/images/fe783173-197920_zadanie4.jpg",
        "/home/dyplom/Dataset/my_dataset/test/images/f931c162-197703_zadanie9.jpg",
        "/home/dyplom/Dataset/my_dataset/test/images/f6d5f4c1-5.jpg",
        "/home/dyplom/Dataset/my_dataset/test/images/e42b1584-197637_zadanie_4.jpg",
        "/home/dyplom/Dataset/my_dataset/test/images/d563396d-197584_zadanie_7.jpg",
        "/home/dyplom/Dataset/my_dataset/test/images/25454343-273.jpg",
    ]

    example_transcription = [
        [
                "_mul_24 PROC",
                "push ebp",
                "mov ebp, esp",
                "sub esp, 32",
                "push ebx",
                "push edi",
                "push esi",
                "mov esi, [ebp + 8]",
                "mov eax, [esi]",
                "mov ebx, [esi + 4]",
                "mov ecx, [esi + 8]",
                "mov edx, [esi + 12]",
                "mov [ebp - 16], eax",
                "mov [ebp - 12], ebx",
                "mov [ebp - 8], ecx",
                "mov [ebp - 4], edx",
                "mov [ebp - 32], eax",
                "mov [ebp - 28], ebx",
                "mov [ebp - 24], ecx",
                "mov [ebp - 20], edx",
                "mov ecx, 3",
                "shl_loop1:",
                "mov eax, [ebp - 16]",
                "mov ebx, [ebp - 12]",
                "mov edx, [ebp - 8]",
                "mov edi, [ebp - 4]",
                "sal eax, 1",
                "rcl ebx, 1",
                "rcl edx, 1",
                "rcl edi, 1",
                "mov [ebp - 16], eax",
                "mov [ebp - 12], ebx",
                "mov [ebp - 8], edx",
                "mov [ebp - 4], edi",
                "loop shl_loop1",
                "mov ecx, 4",
                "shl_loop2:",
                "mov eax, [ebp - 32]",
                "mov ebx, [ebp - 28]",
                "mov edx, [ebp - 24]",
                "mov edi, [ebp - 20]",
                "sal eax, 1",
                "rcl ebx, 1",
                "rcl edx, 1",
                "rcl edi, 1",
                "mov [ebp - 32], eax",
                "mov [ebp - 28], ebx",
                "mov [ebp - 24], edx",
                "mov [ebp - 20], edi",
                "loop shl_loop2",
                "mov eax, [ebp - 32]",
                "mov ebx, [ebp - 28]",
                "mov ecx, [ebp - 24]",
                "mov edx, [ebp - 20]",
                "add [ebp - 16], eax",
                "adc [ebp - 12], ebx",
                "adc [ebp - 8], ecx",
                "adc [ebp - 4], edx",
                "mov esi, [ebp + 12]",
                "mov eax, [ebp - 16]",
                "mov ebx, [ebp - 12]",
                "mov ecx, [ebp - 8]",
                "mov edx, [ebp - 4]",
                "mov [esi], eax",
                "mov [esi + 4], ebx",
                "mov [esi + 8], ecx",
                "mov [esi + 12], edx",
                "pop esi",
                "pop edi",
                "pop ebx",
                "add esp, 32",
                "pop ebp",
                "ret",
                "_mul_24 ENDP",
                "END"
            ],
            [
                "movq xmm0, [ebx]",
                "movq xmm1, [edx]",
                "mov ecx, 15",
                "mov eax, 0",
                "ptl: movq xmm0, [ebx + 8 * ecx]",
                "movq xmm1, [edx + 8 * ecx]",
                "psadbw xmm1, xmm0",
                "movd esi, xmm1",
                "add eax, esi",
                "cmp ecx, 0",
                "jnb ptl",
                "pop esi",
                "pop edx",
                "pop ebx",
                "pop ebp",
                "_mad ENDP"
            ],
            [
                "_mul_24 PROC",
                "mov esi, [ebp + 8]",
                "mov edi, [ebp + 12]",
                "mov eax, [esi]",
                "mov ebx, [esi + 4]",
                "mov ecx, [esi + 8]",
                "mov edx, [esi + 12]",
                "mov [edi], eax",
                "mov [edi + 4], ebx",
                "mov [edi + 8], ecx",
                "mov [edi + 12], edx",
                "mov eax, [esi]",
                "mov ebx, [esi + 4]",
                "mov ecx, [esi + 8]",
                "mov edx, [esi + 12]",
                "shl eax, 3",
                "rcl ebx, 3",
                "rcl ecx, 3",
                "rcl edx, 3",
                "add [edi], eax",
                "adc [edi + 4], ebx",
                "adc [edi + 8], ecx",
                "adc [edi + 12], edx",
                "shl eax, 4",
                "rcl ebx, 4",
                "rcl ecx, 4",
                "rcl edx, 4",
                "add [edi], eax",
                "adc [edi + 4], ebx",
                "adc [edi + 8], ecx",
                "adc [edi + 12], edx",
                "_mul_24 ENDP"
            ],
            [
                "_zad_9 PROC",
                "push ebp",
                "mov ebp, esp",
                "pusha",
                "add esp, 3",
                "mov edx, eax",
                "mov esi, 7",
                "add eax, eax",
                "imul ebx, 7",
                "push ecx",
                "cmp esi, 7",
                "ja e1",
                "jmp e2",
                "e1:",
                "e2:",
                "petla:",
                "loop petla",
                "cmp eax, 3",
                "jl eax_mniejszy",
                "eax_mniejszy",
                "mov [ebp - 4], edx",
                "call CALCULATE",
                "shr eax, 2",
                "mov edi, ebx",
                "xor ecx, ecx",
                "mov eax, 8",
                "shl edx, 7",
                "sub ebx, 1",
                "push esi",
                "pop esi",
                "cmp eax, ebx",
                "je FINISHED:",
                "inc edi",
                "dec ecx",
                "mov ebp, 3",
                "or ebx, 1h",
                "mov edx, 0",
                "xchag eax, esi",
                "popa",
                "pop ebp",
                "ret",
                "_zad_9 ENDP",
                "END"
            ],
            [
                "_search PROC",
                "push ebp",
                "mov ebp, esp",
                "sub esp, 4",
                "pusha",
                "mov edi, [edi + 8]",
                "xor ecx, ecx",
                "text_len_loop:",
                "cmp word ptr [edi], 0",
                "je text_len_done",
                "add edi, 2",
                "inc ecx",
                "jmp text_len_loop",
                "text_len_done:",
                "mov esi, ecx",
                "mov edi, [ebp + 12]",
                "xor ecx, ecx",
                "patt_len_loop:",
                "cmp word ptr [edi], 0",
                "je patt_len_done",
                "add edi, 2",
                "inc ecx",
                "jmp patt_len_loop",
                "patt_len_done:",
                "mov ebx, ecx",
                "push [ebp + 12]",
                "push [ebp + 8]",
                "call _create_last",
                "add esp, 8",
                "mov [ebp - 4], eax",
                "xor edx, edx",
                "search_loop:",
                "mov ecx, edx",
                "add ecx, ebx",
                "cmp ecx, esi",
                "ja search_not_found",
                "mov edi, ebx",
                "dec edi",
                "compare_loop:",
                "mov ecx, [ebp + 8]",
                "mov eax, edx",
                "add eax, edi",
                "shl eax, 1",
                "mov ax, [ecx + eax]",
                "mov ecx, [ebp + 12]",
                "mov eax, edi",
                "shl eax, 1",
                "mov cx, [ecx + eax]",
                "cmp ax, cx",
                "jne mismatch",
                "test edi, edi",
                "jz found_match",
                "dec edi",
                "jmp compare_loop",
                "mismatch:",
                "mov ecx, [ebp + 8]",
                "mov eax, edx",
                "add eax, edi",
                "shl eax, 1",
                "movsx eax, word ptr [ecx + eax]",
                "push eax",
                "push dword ptr [ebp - 4]",
                "call _find_in_last",
                "add esp, 8",
                "mov ecx, edi",
                "sub ecx, eax",
                "cmp ecx, 1",
                "jge shift",
                "mov ecx, 1",
                "shift:",
                "add edx, ecx",
                "jmp search_loop",
                "found_match:",
                "inc edx",
                "mov eax, edx",
                "jmp _exit",
                "search_not_found:",
                "mov eax, -1",
                "_exit:",
                "popa",
                "mov esp, ebp",
                "pop ebp",
                "ret",
                "_search ENDP"
            ],
            [
                "_mul_24 PROC",
                "push ebp",
                "mov ebp, esp",
                "push eax",
                "push ebx",
                "push ecx",
                "push edx",
                "push esi",
                "push edi",
                "mov edi, [ebp + 12]",
                "mov esi, [ebp + 8]",
                "mov eax, [esi]",
                "mov ebx, [esi + 4]",
                "mov ecx, [esi + 8]",
                "mov edx, [esi + 12]",
                "mov [edi], eax",
                "mov [edi + 4], ebx",
                "mov [edi + 8], ecx",
                "mov [edi + 12], edx",
                "shl eax, 3",
                "rcl ebx, 3",
                "rcl ecx, 3",
                "rcl edx, 3",
                "add [edi], eax",
                "add [edi + 4], ebx",
                "add [edi + 8], ecx",
                "add [edi + 12], edx",
                "shr edx, 1",
                "rcr ecx, 1",
                "rcr ebx, 1",
                "rcr eax, 1",
                "add [edi], eax",
                "add [edi + 4], ebx",
                "add [edi + 8], ecx",
                "add [edi + 12], edx",
                "pop edi",
                "pop esi",
                "pop edx",
                "pop ecx",
                "pop ebx",
                "pop eax",
                "pop ebp",
                "ret",
                "_mul_24 ENDP"
            ],
            [
                "_create_last PROC",
                "push ebp",
                "mov ebp, esp",
                "push ebx",
                "push esi",
                "push edi",
                "mov esi, [ebp + 8]",
                "mov edi, [ebp + 12]",
                "xor eax, eax",
                "xor ebx, ebx",
                "xor ecx, ecx",
                "mov al, [edi]",
                "test al, al",
                "jz wzorzec_empty",
                "xor edx, edx",
                "wzorzec_len:",
                "cmp byte ptr [edi + edx], 0",
                "je wzorzec_len_end",
                "inc edx",
                "jmp wzorzec_len",
                "wzorzec_len_end:",
                "mov ecx, edx",
                "push 0",
                "push 0",
                "push 0",
                "push 0",
                "call _VirtualAlloc@16",
                "test eax, eax",
                "jz alloc_error",
                "mov ebx, eax",
                "xor eax, eax",
                "xor edx, edx",
                "mov al, [esi]",
                "test al, al",
                "jz tablica_end",
                "push ebx",
                "push esi",
                "push edi",
                "mov edi, ebx",
                "mov ecx, 256",
                "mov al, -1",
                "rep stosb",
                "pop edi",
                "pop esi",
                "pop ebx",
                "xor ecx, ecx",
                "xor edx, edx",
                "mov al, [edi + ecx]",
                "test al, al",
                "jz tablica_end",
                "wzorzec_index:",
                "mov al, [edi + ecx]",
                "mov [ebx + eax], cl",
                "inc ecx",
                "cmp byte ptr [edi + ecx], 0",
                "jne wzorzec_index",
                "mov eax, ebx",
                "jmp end",
                "wzorzec_empty:",
                "xor eax, eax",
                "jmp end",
                "alloc_error:",
                "xor eax, eax",
                "jmp end",
                "tablica_end:",
                "mov eax, ebx",
                "end:",
                "pop edi",
                "pop esi",
                "pop ebx",
                "pop ebp",
                "ret",
                "_create_last ENDP"
            ],
            [
                "nowy_ja _PROC",
                "push ebp",
                "mov ebp, esp",
                "pusha",
                "mov eax, [ebp + 8]",
                "jne dalej",
                "popa",
                "pop ebp",
                "ret 4",
                "dalej:",
                "jnc dalej2",
                "popa",
                "pop ebp",
                "ret 4",
                "dalej2:",
                "mov dword ptr [esp], eax",
                "popa",
                "pop ebp",
                "ret 4",
                "nowy_ja _ENDP"
            ]
    ]

    text = few_shot_prediction(model, processor, image_path, path_example_images, example_transcription, max_new_tokens=2000)

    print(text)

    