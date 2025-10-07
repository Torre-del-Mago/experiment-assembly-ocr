# TrOCR - Transformer-based Optical Character Recognition

A PyTorch implementation of TrOCR (Transformer-based Optical Character Recognition) for training and inference on handwritten text recognition tasks.

## 🔍 Overview

This TrOCR implementation provides an end-to-end solution for handwritten text recognition using Microsoft's TrOCR architecture. The system combines a vision encoder (based on ViT) with a text decoder (based on RoBERTa) for accurate OCR performance.

## 🏗️ Project Structure

```
TrOCR/
├── train.py                    # Main training script
├── OCR_dataset.py             # Custom dataset class
├── utils.py                   # Utility functions (CER computation)
├── output/                    # Model checkpoints and outputs
└── README.md                  # This file
```

## 🚀 Features

- **Pre-trained Model Fine-tuning**: Built on Microsoft's TrOCR base models
- **Custom Dataset Support**: Train on your own handwritten text data
- **Multiple Optimizers**: Support for AdamW, SGD, and Adam optimizers
- **Early Stopping**: Prevent overfitting with patience-based stopping
- **Checkpoint Management**: Save and resume training from specific epochs
- **Comprehensive Evaluation**: CER (Character Error Rate) metrics
- **Weights & Biases Integration**: Track experiments and metrics
- **GPU/CPU Support**: Automatic device detection and configuration

## 📋 Requirements

```txt
torch
torchvision
transformers
datasets
evaluate
tqdm
wandb
PIL
```

## 🛠️ Installation

1. **Install dependencies**:
```bash
pip install torch transformers datasets evaluate tqdm wandb pillow
```

2. **Set up Weights & Biases** (optional):
```bash
wandb login
```

## 📊 Dataset Format

The dataset should be organized as follows:

```
Dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels.csv
├── val/
│   ├── images/
│   └── labels.csv
└── test/
    ├── images/
    └── labels.csv
```

### Labels Format
The `labels.csv` file should contain image filenames and corresponding transcriptions:
```csv
image1.jpg,Hello world
image2.jpg,Sample text
image3.jpg,Another example
```

## 🎯 Usage

### Training

Configure your training parameters and run:

```python
from train import train_trOCR

config = {
    "batch_size": 16,
    "lr": 0.0001,
    "num_epochs": 100,
    "max_target_length": 128,
    "dropout_rate": 0.1,
    "optimizer": "adamw",
    "early_stopping_patience": 5,
    "checkpoint_interval_epochs": 5,
    "train_data_path": "/path/to/train",
    "val_data_path": "/path/to/val", 
    "test_data_path": "/path/to/test",
    "model_path": "microsoft/trocr-base-handwritten",
    "processor_path": "microsoft/trocr-base-handwritten",
    "output_dir": "./output",
    "device": "cuda:0" if torch.cuda.is_available() else "cpu"
}

train_trOCR(config)
```

### Quick Start

```bash
python train.py
```

The script will automatically:
- Load the pre-trained TrOCR model
- Initialize datasets from the configured paths
- Start training with progress tracking
- Save the best model based on validation CER
- Log metrics to Weights & Biases

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Training batch size | 16 |
| `lr` | Learning rate | 0.0001 |
| `num_epochs` | Maximum training epochs | 100 |
| `max_target_length` | Maximum text sequence length | 128 |
| `dropout_rate` | Dropout probability | 0.1 |
| `optimizer` | Optimizer type (`adamw`, `sgd`, `adam`) | "adamw" |
| `early_stopping_patience` | Early stopping patience | 5 |
| `checkpoint_interval_epochs` | Save interval (epochs) | 5 |
| `check_sanity` | Debug mode (limits to 10 batches) | False |

## 📈 Model Performance

The training script tracks:
- **Training Loss**: Cross-entropy loss on training data
- **Validation Loss**: Cross-entropy loss on validation data
- **Test Loss**: Cross-entropy loss on test data
- **Validation CER**: Character Error Rate on validation set
- **Test CER**: Character Error Rate on test set

All metrics are automatically logged to Weights & Biases for visualization.

## 🔧 Model Architecture

The implementation uses:
- **Vision Encoder**: DeiT (Data-efficient Image Transformer)
- **Text Decoder**: RoBERTa-based decoder
- **Base Model**: `microsoft/trocr-base-handwritten`
- **Processor**: TrOCR processor for image and text preprocessing

## 💾 Model Checkpoints

The system saves:
- **Best Model**: Saved when validation CER improves (`output/best_model/`)
- **Interval Checkpoints**: Saved every N epochs as configured
- **Processor**: Tokenizer and image processor saved with model

## 🧮 Custom Dataset Class

The `OCRDataset` class handles:
- Image loading and preprocessing
- Text tokenization with padding/truncation
- Proper label formatting for loss computation
- Batch preparation for training

```python
from OCR_dataset import OCRDataset
from transformers import TrOCRProcessor

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
dataset = OCRDataset(
    base_path="/path/to/data",
    processor=processor,
    max_target_length=128
)
```

## 🔍 Evaluation Metrics

### Character Error Rate (CER)
```python
from utils import compute_cer

cer = compute_cer(predictions, references)
```

CER measures character-level accuracy and is the primary metric for OCR evaluation.

## 🚀 Advanced Usage

### Custom Optimizer Configuration
```python
# SGD with momentum
config["optimizer"] = "sgd"
config["momentum"] = 0.9

# Adam with different learning rate
config["optimizer"] = "adam"
config["lr"] = 0.001
```

### Early Stopping
```python
config["early_stopping_patience"] = 10  # Stop after 10 epochs without improvement
```

### Debug Mode
```python
config["check_sanity"] = True  # Limit to 10 batches per epoch for testing
```

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce `batch_size`
2. **Import Errors**: Ensure all dependencies are installed
3. **Dataset Loading**: Check file paths and CSV format
4. **Model Loading**: Verify internet connection for downloading pre-trained models

### Performance Tips

- Use GPU for faster training (`device: "cuda:0"`)
- Increase `batch_size` if memory allows
- Adjust `max_target_length` based on your text length
- Use mixed precision training for memory efficiency

## 📝 License

This project uses the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Microsoft Research for the TrOCR architecture
- Hugging Face for the Transformers library
- PyTorch team for the deep learning framework

---

**Note**: This implementation is designed for research and educational purposes. For production use, consider additional optimization and error handling.