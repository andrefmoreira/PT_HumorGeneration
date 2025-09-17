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
