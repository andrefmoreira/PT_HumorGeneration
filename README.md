# PT_HumorGeneration
T5 fine-tuned model to generate Portuguese humor (training code only, model not provided).

---

## About
This repository contains code to fine-tune a **T5-based Portuguese model** for humor generation. The repository **does not include the trained model**, so users need to train it themselves using the provided scripts and datasets.

---

## Getting Started

### Prerequisites
- Python (≥3.10 recommended)
- PyTorch
- Hugging Face Transformers
- Datasets library
- GPU recommended for training

### Installation
```bin
git clone https://github.com/andrefmoreira/PT_HumorGeneration
cd PT_HumorGeneration
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Training the model

1. Make sure the preprocessed datasets are available in ./processed_datasets/ with train, validation, and test folders.

2. Run the training script:
   ```bin
   python train_t5_humor.py
   ```
This will train the model and save it in ./t5_humor_pt_model.

### Usage

After training the model you can either run the code using either the Test_input_PTT5.py or the Test_dataset_PTT5.py.
