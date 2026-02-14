# 📝 Amazon Review Synthesizer: Fine-Tuning T5

This repository contains a complete pipeline for fine-tuning a **T5 (Text-to-Text Transfer Transformer)** model to generate realistic product reviews based on a product title and a desired star rating.

By leveraging the **McAuley-Lab Amazon Reviews 2023** dataset, this project demonstrates how to preprocess large-scale e-commerce data and train a sequence-to-sequence model for conditional text generation.

## 🚀 Overview

The goal is to provide the model with a prompt like:
`review: Adobe Photoshop, 5 Stars!`

And have the model generate a coherent, contextually relevant review:
`"This software is the industry standard for a reason. The new AI features in this version saved me hours of work..."`

## 🛠️ Tech Stack

* **Language:** Python
* **Framework:** Hugging Face Transformers
* **Model:** `t5-base`
* **Dataset:** [Amazon Reviews 2023](https://github.com/hyp1231/AmazonReviews2023)
* **Libraries:** `datasets`, `numpy`, `pandas`, `torch`

## 📊 Dataset Pipeline

Since the legacy `amazon_us_reviews` dataset was deprecated, this project utilizes the **Software** category from the 2023 McAuley-Lab release.

1. **Data Merging:** Inner joins metadata (titles) with raw reviews via `parent_asin`.
2. **Filtering:** Only includes **Verified Purchases** and reviews longer than **100 characters** to ensure quality.
3. **Stratification:** Uses `class_encode_column` to prepare for balanced sampling across star ratings.
4. **Tokenization:** Implements a custom preprocessing function to format inputs into the T5-standard "prefix" format.

## 🏗️ Model Training

The training utilizes the `Trainer` API for efficient execution:

| Parameter | Value |
| --- | --- |
| **Base Model** | `t5-base` |
| **Epochs** | 3 |
| **Batch Size** | 12 |
| **Max Length** | 128 tokens |
| **Optimizer** | AdamW (Default) |

### Training Logic:

The model is trained using a prompt-response pattern:

* **Input:** `review: [Product Title], [Rating] Stars!`
* **Target:** `[Review Body]`
* **Loss Calculation:** Padding tokens in the labels are replaced with `-100` so they are ignored by the loss function (Standard Seq2Seq practice).

## 🖥️ Usage

To run this project, ensure you have the required dependencies installed:

```bash
pip install datasets transformers sentencepiece torch

```

### Generating a Review

After training, you can generate synthetic reviews by passing a custom string to the `generate_review` function:

```python
# Example Usage
print(generate_review("Microsoft Office 365, 5 Stars!"))

```

## 📈 Inference Parameters

To ensure high-quality output, the generation function uses:

* **Beam Search:** `num_beams=6` for better sequence probability.
* **N-gram Blocking:** `no_repeat_ngram_size=3` to prevent repetitive "looping" text.
* **Early Stopping:** To cease generation once a coherent ending is reached.

---

### 🤝 Contributing

Feel free to fork this repository and experiment with different product categories (e.g., "Electronics" or "Home_and_Kitchen") by changing the `dataset_category` variable in the script.

---
