"""
Gradio OCR Example
This example demonstrates how to use Gradio to create an interactive OCR interface.
"""

import gradio as gr
import pytesseract
from PIL import Image
import numpy as np


def perform_ocr(image):
    """
    Perform OCR on the input image using pytesseract.
    
    Args:
        image: Input image (numpy array or PIL Image)
    
    Returns:
        str: Extracted text from the image
    """
    if image is None:
        return "No image provided"
    
    try:
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        
        if not text.strip():
            return "No text detected in the image"
        
        return text
    except Exception as e:
        return f"Error during OCR: {str(e)}"


def create_gradio_interface():
    """
    Create and configure the Gradio interface for OCR.
    
    Returns:
        gr.Interface: Configured Gradio interface
    """
    # Create the interface
    interface = gr.Interface(
        fn=perform_ocr,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=gr.Textbox(label="Extracted Text", lines=10),
        title="OCR Demo with Gradio",
        description="Upload an image containing text, and the OCR model will extract the text from it.",
        examples=[
            # You can add example image paths here
            # ["path/to/example1.jpg"],
            # ["path/to/example2.png"],
        ],
        theme=gr.themes.Soft(),
    )
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_gradio_interface()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
    )
