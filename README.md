# experiment-assembly-ocr
It was project for master thesis. Training and Evaluation models on custom-build dataset.

## Gradio OCR Example

This repository includes an example of using Gradio to create an interactive OCR (Optical Character Recognition) interface.

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine installed on your system

#### Installing Tesseract

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
Download and install from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Torre-del-Mago/experiment-assembly-ocr.git
cd experiment-assembly-ocr
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Full OCR Example (requires Tesseract)

Run the Gradio OCR demo:
```bash
python gradio_ocr_example.py
```

#### Option 2: Simple Image Analysis Example (no external dependencies)

If you don't have Tesseract installed, you can try the simpler example:
```bash
python gradio_simple_example.py
```

The application will start a local web server (default: http://localhost:7860). Open this URL in your web browser to access the interactive interface.

### Features

**OCR Example (`gradio_ocr_example.py`):**
- Upload images containing text
- Automatically extract text using OCR
- Interactive web interface powered by Gradio
- Support for various image formats (PNG, JPEG, etc.)

**Simple Example (`gradio_simple_example.py`):**
- Upload images to analyze
- Get image dimensions, color mode, and other properties
- No external dependencies required (works out of the box)
- Great starting point for learning Gradio
