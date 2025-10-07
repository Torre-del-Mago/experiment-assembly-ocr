"""
Simple Gradio Example
This is a minimal example demonstrating Gradio interface creation without external dependencies.
"""

import gradio as gr
from PIL import Image
import numpy as np


def analyze_image(image):
    """
    Analyze the input image and return basic information.
    This is a simplified example that doesn't require OCR libraries.
    
    Args:
        image: Input image (numpy array or PIL Image)
    
    Returns:
        str: Analysis results
    """
    if image is None:
        return "No image provided"
    
    try:
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Get image information
        width, height = pil_image.size
        mode = pil_image.mode
        
        result = f"""Image Analysis Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📐 Dimensions: {width} x {height} pixels
🎨 Color Mode: {mode}
📊 Aspect Ratio: {width/height:.2f}
💾 Total Pixels: {width * height:,}
"""
        return result
    except Exception as e:
        return f"Error during analysis: {str(e)}"


def create_simple_interface():
    """
    Create and configure a simple Gradio interface.
    
    Returns:
        gr.Interface: Configured Gradio interface
    """
    interface = gr.Interface(
        fn=analyze_image,
        inputs=gr.Image(type="pil", label="Upload Image"),
        outputs=gr.Textbox(label="Analysis Results", lines=8),
        title="Simple Image Analysis with Gradio",
        description="Upload an image to see its basic properties. This example demonstrates Gradio without requiring external OCR dependencies.",
        theme=gr.themes.Soft(),
    )
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_simple_interface()
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,  # Default Gradio port
    )
