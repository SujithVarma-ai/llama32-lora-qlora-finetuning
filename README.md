# 🚀 Llama 3.2 LoRA & QLoRA Fine-Tuning

A comprehensive hands-on project demonstrating Parameter-Efficient Fine-Tuning (PEFT) of **Llama 3.2 3B** using **LoRA** and **QLoRA** for product price prediction.

This project covers the complete open-source LLM fine-tuning workflow, including quantization, LoRA adapter configuration, prompt dataset preparation, baseline evaluation, training, hyperparameter tuning, and final model evaluation.

---

## 📌 Project Overview

The goal of this project is to fine-tune an open-source Large Language Model (LLM) to predict product prices from product descriptions while minimizing computational requirements using LoRA and QLoRA.

The project follows a structured workflow:

### 1. LoRA & QLoRA Configuration

- Explored Llama 3.2 3B architecture
- Identified target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- Applied LoRA adapters for parameter-efficient fine-tuning
- Implemented QLoRA using 4-bit quantization
- Compared trainable and total parameters
- Optimized memory usage for efficient training

### 2. Dataset Preparation & Baseline Evaluation

- Loaded product pricing datasets
- Analyzed token distributions
- Created prompt-completion pairs
- Generated training-ready prompt datasets
- Evaluated the base Llama 3.2 model
- Established baseline performance metrics (MAE, MSE, R²)

### 3. Training Pipeline Setup

- Configured Supervised Fine-Tuning (SFT)
- Defined batch size, epochs, optimizer, and learning rate
- Configured gradient accumulation
- Integrated Weights & Biases experiment tracking
- Prepared Hugging Face Hub integration

### 4. Hyperparameter Tuning & Model Training

- Trained LoRA adapters on prompt datasets
- Monitored training and validation loss
- Tuned learning rate and training parameters
- Saved and evaluated model checkpoints

### 5. Fine-Tuned Model Evaluation

- Generated final fine-tuned model
- Evaluated model performance on unseen data
- Compared results against baseline model
- Analyzed MAE, MSE, and R² metrics
- Assessed prediction quality and model improvements

---

## 🔄 Workflow

```text
LoRA / QLoRA Configuration
        ↓
Prompt Dataset Preparation
        ↓
Base Model Evaluation
        ↓
Training Configuration
        ↓
LoRA Fine-Tuning
        ↓
Final Evaluation
```

---

## 🧠 Technologies Used

* Python
* PyTorch
* Transformers
* Hugging Face Hub
* PEFT (LoRA)
* QLoRA
* TRL (SFTTrainer)
* BitsAndBytes
* Weights & Biases (W&B)
* Matplotlib
* NumPy
* Pandas

---

## 🔥 LoRA & QLoRA

### LoRA (Low-Rank Adaptation)

Instead of updating all 3 billion parameters of Llama 3.2, LoRA freezes the base model and trains only a small number of adapter parameters.

Benefits:

* Reduced memory consumption
* Faster training
* Smaller checkpoints
* Efficient fine-tuning

### QLoRA

QLoRA combines:

* 4-bit Quantization
* LoRA Adapters

Benefits:

* Significant GPU memory reduction
* Ability to fine-tune larger models on limited hardware
* Maintains strong performance

---

## 📊 Evaluation Metrics

The model is evaluated using:

### MAE (Mean Absolute Error)

Measures average prediction error.

Lower is better.

### MSE (Mean Squared Error)

Measures squared prediction error.

Lower is better.

### R² Score

Measures how well the model explains price variation.

Higher is better.

---

## 📈 Results

| Metric | Value |
|----------|----------|
| MAE | 39.88 |
| MSE | 4636 |
| R² Score | 78.9% |

The model achieved an MAE of 39.88, MSE of 4636, and R² score of 78.9% on the evaluation dataset.

```

## ⚙️ Installation

```bash
git clone https://github.com/SujithVarma-ai/llama32-lora-qlora-finetuning.git

cd llama32-lora-qlora-finetuning

pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

---

## ✨ Key Features

- Fine-tuning Llama 3.2 3B using LoRA and QLoRA
- 4-bit quantization for memory-efficient training
- Prompt-completion dataset generation
- Supervised Fine-Tuning (SFT) pipeline
- Hyperparameter tuning and experiment tracking
- Automated evaluation using MAE, MSE, and R² metrics
- Hugging Face Hub integration
- Weights & Biases experiment monitoring

---

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

* Open-source LLM fine-tuning
* LoRA adapter architecture
* QLoRA quantization techniques
* Prompt engineering for fine-tuning
* Hyperparameter tuning
* Model evaluation
* Experiment tracking with W&B

---

## 📜 License

This project is intended for educational and research purposes.
