# Donut OCR Fine-tuning Project

This project implements fine-tuning of the Donut (Document Understanding Transformer) model for OCR tasks. Donut is an end-to-end document understanding model that processes document images directly without requiring separate OCR preprocessing.

## Overview

The project contains a complete pipeline for training a Donut model on custom OCR datasets, including:
- Custom dataset preparation and loading
- Model training with validation and testing
- Support for data augmentation
- Integration with Weights & Biases for experiment tracking

## Features

- **Custom Dataset Support**: Load and process your own OCR datasets
- **Data Augmentation**: Built-in support for augmented training images
- **Early Stopping**: Prevents overfitting with configurable patience
- **Checkpointing**: Regular model checkpoints during training
- **Metrics Tracking**: Character Error Rate (CER) and loss monitoring
- **GPU Support**: CUDA acceleration for faster training

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have CUDA installed if you want to use GPU acceleration.

## Dataset Preparation

### Dataset Structure
Your dataset should follow this structure:
```
dataset/
├── train/
│   ├── images/
│   └── donut_data.json
├── val/
│   ├── images/
│   └── donut_data.json
└── test/
    ├── images/
    └── donut_data.json
```

### Creating the Dataset
Use the `dataset_create.py` script to convert your JSON annotations into HuggingFace dataset format:

```python
python dataset_create.py
```

The script expects JSON files with the following format:
```json
{
  "image_code": "image_filename.jpg",
  "gt_parse": {...},  // Ground truth annotations
  "aug": []  // Augmentation variants (optional)
}
```

## Training

### Configuration
Edit the configuration in `train.py`:

```python
config = {
    "max_epochs": 20,
    "lr": 1e-4,
    "batch_size": 1,
    "max_length": 256,
    "pretrained_model_name_or_path": "naver-clova-ix/donut-base-finetuned-cord-v2",
    "input_size": [1280, 960],
    "dropout_rate": 0.1,
    "early_stopping_patience": 5,
    "checkpoint_interval_epochs": 1,
    "check_sanity": True  # Set to False for full training
}
```

### Starting Training
```bash
python train.py
```

### Monitoring
The training process will:
- Display progress bars for each epoch
- Print training, validation, and test metrics
- Save the best model based on validation loss
- Log metrics to Weights & Biases (if configured)

## Model Architecture

The project uses the Donut model with:
- **Encoder**: Vision Transformer (ViT) for image processing
- **Decoder**: BART for text generation
- **Special Tokens**: `<ocr>` and `</ocr>` for task specification

## Key Components

### DonutDataset (`donut_dataset.py`)
- Custom PyTorch Dataset class for Donut
- Handles image preprocessing and tokenization
- Supports JSON-to-token conversion for structured outputs
- Manages special tokens for the model

### Training Script (`train.py`)
- Complete training loop with validation and testing
- Early stopping mechanism
- Model checkpointing
- Metrics computation (CER)

### Dataset Creation (`dataset_create.py`)
- Utility functions for preparing HuggingFace datasets
- Handles augmented images
- Converts JSON annotations to required format

## Output

The trained model will be saved in the `result/` directory with a unique experiment ID. The directory contains:
- Model weights and configuration
- Tokenizer files
- Training checkpoints (if enabled)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers
- Datasets
- CUDA (optional, for GPU acceleration)

## Notes

- Set `check_sanity=False` in the config for full dataset training
- Adjust `batch_size` based on your GPU memory
- The model supports various input image sizes - adjust `input_size` as needed
- Training time depends on dataset size and hardware configuration

## Troubleshooting

- **Out of Memory**: Reduce batch size or input image size
- **Slow Training**: Enable CUDA if available, increase batch size
- **Poor Performance**: Increase training epochs, adjust learning rate, or check data quality