<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/c3ff5854-cbfd-4bc9-9b9d-2259162da87b" />Sure! Here's a professional and detailed `README.md` file tailored for your GPT-2-based TinyStories project. You can copy and paste it directly into your GitHub repository:

---

````markdown
# 🧠 TinyStories GPT-2 - Minimal Transformer Language Model

This project implements a simplified GPT-2 architecture using PyTorch to train on a subset of the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories). It covers data loading, model implementation from scratch (including self-attention and positional encoding), training with gradient accumulation, evaluation using perplexity, and text generation.

---

## 📌 Features

- ✅ Custom Transformer Decoder architecture (GPT-style)
- ✅ Multi-head self-attention and positional encoding
- ✅ Tokenization using HuggingFace GPT-2 tokenizer
- ✅ Train/test split from local TinyStories dataset
- ✅ Cross-entropy loss with padding token masking
- ✅ Gradient accumulation support
- ✅ Perplexity evaluation
- ✅ Greedy text generation

---

## 🧪 Dataset

We use the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories), a cleaned corpus of short, simple stories suitable for language modeling experiments.

You should first download and cache it locally using:

```python
from datasets import load_dataset

# Run this before using load_from_disk
dataset = load_dataset("roneneldan/TinyStories", split="train")
dataset.save_to_disk("./tinystories_local")
```

Then in the main script, it is loaded using:

```python
from datasets import load_from_disk
dataset = load_from_disk("./tinystories_local")
```

---

## 🚀 Training

To train the model on 10% of the dataset (90% train / 10% test split), simply run:

```bash
python gpt2_tinystories.py
```

Training parameters:

* Epochs: 5
* Batch Size: 8
* Learning Rate: 3e-4
* Gradient Accumulation Steps: 2
* Loss: Cross-entropy with padding mask

After each epoch:

* Model checkpoint is saved
* Generated text sample is saved
* Training loss and accuracy are plotted and saved

---

## 📈 Evaluation

The model is evaluated using **perplexity** on the test set:

```python
def compute_perplexity(model, test_loader):
    ...
```

This provides a numeric measure of how well the model predicts text (lower is better).

---

## ✍️ Text Generation

You can generate text using either greedy decoding or sampling:

```python
def generate_text(model, prompt, max_length=50, method="greedy"):
    ...
```

Example:

```python
Prompt: "Once upon a time, in a small village"
Generated Text: "Once upon a time, in a small village, there was a little boy named Max who loved to build robots..."
```

---

## 🧠 Model Architecture

* Embedding Layer
* Positional Encoding (Sinusoidal)
* 2 Transformer Decoder Layers:

  * Multi-Head Self-Attention
  * Feed-Forward Network
  * LayerNorm + Residual Connections
* Linear Output Layer

Total Parameters: \~86M (based on GPT-2 tiny configuration)

---

## 🧰 Dependencies

Make sure you have the following Python packages installed:

```bash
pip install torch transformers datasets matplotlib
```

---

## 💾 Loading and Generating from Saved Model

Use `gpt2_generate.py` to load a saved model and generate new text:

```bash
python gpt2_generate.py
```

Edit the path to your `.pth` model and set the desired prompts in the script.

---

## 📊 Results

Training metrics (loss and accuracy per epoch) are visualized in:

```
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/30eea5df-4427-4099-aac3-662a0f9f110d" />

```

You can also inspect generated text samples after each epoch for qualitative evaluation.

---

## 📌 Notes

* This implementation is educational and **not optimized for production**.
* For large-scale or more efficient implementations, consider using the HuggingFace `transformers` library directly.
* You can easily extend this project with additional decoder layers, larger datasets, or sampling techniques (e.g. top-k, nucleus).

---

## 📚 References

* [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762)
* [GPT-2 Paper](https://openai.com/research/language-unsupervised)
* [TinyStories Dataset](https://huggingface.co/datasets/roneneldan/TinyStories)
* [HuggingFace Transformers](https://huggingface.co/transformers/)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

## ✨ Author

**Mohamed Ashraf**
Computer and Data Sciences | Alexandria University
AI/ML Enthusiast • Deep Learning • NLP • Transformers
[GitHub]([github.com/mohamedashraff22](https://github.com/mohamedashraff22)) | [LinkedIn]([https://linkedin.com/](https://www.linkedin.com/in/mohameed-ashraf/))

---



Let me know if you'd like to include images (e.g., a training plot or model diagram), Hugging Face model integration, or Colab support.
```
