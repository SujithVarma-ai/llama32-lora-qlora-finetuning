# 🚀 Llama 3.2 LoRA & QLoRA Fine-Tuning

A comprehensive hands-on project demonstrating Parameter-Efficient Fine-Tuning (PEFT) of **Llama 3.2 3B** using **LoRA** and **QLoRA** for product price prediction.

This project covers the complete open-source LLM fine-tuning workflow, including quantization, LoRA adapter configuration, prompt dataset preparation, baseline evaluation, training, hyperparameter tuning, and final model evaluation.

---

## 📌 Project Overview

The goal of this project is to fine-tune an open-source Large Language Model (LLM) to predict product prices from product descriptions while minimizing computational requirements using LoRA and QLoRA.

The project follows a structured workflow:

### Day 1: LoRA & QLoRA Fundamentals

* Explored Llama 3.2 3B architecture
* Identified target modules (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
* Applied LoRA adapters
* Implemented QLoRA with 4-bit quantization
* Compared trainable vs total parameters
* Reduced GPU memory requirements

### Day 2: Prompt Dataset Creation & Baseline Evaluation

* Loaded lite and full datasets
* Analyzed token distributions
* Created prompt-completion pairs
* Generated training-ready prompt datasets
* Evaluated the base Llama 3.2 model
* Measured baseline metrics:

  * MAE (Mean Absolute Error)
  * MSE (Mean Squared Error)
  * R² Score

### Day 3: Training Configuration

* Configured SFT training pipeline
* Defined training parameters
* Configured batch size and epochs
* Set learning rate scheduling
* Enabled experiment tracking with Weights & Biases

### Day 4: Training & Hyperparameter Tuning

* Trained LoRA adapters
* Monitored training and validation loss
* Tuned learning rate and training parameters
* Saved checkpoints during training

### Day 5: Fine-Tuning & Evaluation

* Generated final fine-tuned model
* Evaluated model performance
* Compared results against baseline model
* Analyzed prediction quality

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

## 📁 Repository Structure

```text
.
├── pricer/
├── FTOpenSourceBaseModel.ipynb
├── FinetuningOpenSource_QLoRA.ipynb
├── FinetuningOpenSource_Evaluation.ipynb
├── FinetuningOpenSource_Hyperparameters_Training.ipynb
├── FinetuningOpenSource_(small_dataset).ipynb
├── FinetuningOpenSource_(large_dataset).ipynb
├── results.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

---

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

## 🎯 Learning Outcomes

Through this project, I gained practical experience in:

* Open-source LLM fine-tuning
* LoRA adapter architecture
* QLoRA quantization techniques
* Prompt engineering for fine-tuning
* Hyperparameter tuning
* Model evaluation and benchmarking
* Hugging Face ecosystem
* Experiment tracking with W&B

---

## 📜 License

This project is intended for educational and research purposes.
