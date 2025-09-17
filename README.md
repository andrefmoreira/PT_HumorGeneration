# PT Humor Generation

T5 fine-tuned model to generate Portuguese humor.

---

## Table of Contents

- [About](#about)  
- [Features](#features)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)
  
---

## About

This repository contains a **T5-based** model fine-tuned for **Portuguese Humor generation**. The aim is to generate humorous text in Portuguese (jokes, playful remarks, etc.) using modern NLP methods.

---

## Features

- Fine-tuned T5 model in Portuguese  
- Generates humorous content with style in Portuguese  
- (Optional: you can add features like “adjustable humor level”, “context-aware prompts”, etc.)

---

## Getting Started

### Prerequisites

You’ll need the following tools or resources:

- Python (version ≥ 3.x)  
- PyTorch / Transformers library (Hugging Face)  
- GPU (recommended for training/fine-tuning/inference for speed)  
- (Optional) Tokenizer files, pre-trained T5 checkpoints  

### Installation

Here’s how to set up the project locally:

1. Clone this repo  
   ```bash
   git clone https://github.com/andrefmoreira/PT_HumorGeneration.git
   cd PT_HumorGeneration

2. (Optional) Create a virtual environment
   ```bash
    python3 -m venv venv
    source venv/bin/activate

4. Install dependencies
    ```bash
    pip install -r requirements.txt

5. Download or prepare any required model checkpoints / data if needed.


## Usage

Here's the code you need to run to be able to use the model
  ```bash
  from model import HumorGenerator  # adjust according to your module structure
  
  # load model
  gen = HumorGenerator.from_pretrained('path/to/fine_tuned_t5_model')
  
  # generate humor
  prompt = "Insira aqui o seu prompt engraçado em português"
  output = gen.generate(prompt, max_length=100, temperature=0.9)
  print(output)
