"""
Twitter BERT Sentiment Classifier Training Pipeline
Assignment 03 - NLP Lab

An enterprise-ready training script that includes dataset cleanup, tokenization packing,
evaluation metrics calculation (Accuracy + Macro F1), and automated device fallback (CUDA/CPU).
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    TrainerCallback
)

# --------------------------------------------------------------------------
# 1. Configuration & Hyperparameters
# --------------------------------------------------------------------------
MODEL_NAME = "bert-base-uncased"
DATASET_PATH = r"C:\Users\Dell\Desktop\python\twitter_training.csv"
OUTPUT_DIR = r"C:\Users\Dell\Desktop\python\bert_sentiment_model"
LOGS_DIR = r"C:\Users\Dell\Desktop\python\logs"
RESULTS_DIR = r"C:\Users\Dell\Desktop\python\results"

# Label matrix mappings matching dataset unique distributions
LABEL_MAP = {"Irrelevant": 0, "Negative": 1, "Neutral": 2, "Positive": 3}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# --------------------------------------------------------------------------
# 2. PyTorch Dataset Wrapper
# --------------------------------------------------------------------------
class TwitterDataset(torch.utils.data.Dataset):
    """
    Efficient data wrapper to feed tokenized tensor frames to the BERT network graph.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# --------------------------------------------------------------------------
# 3. Helper Metrics Computation
# --------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """
    Computes professional classification matrix scores during validation cycles.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro")
    
    return {
        "accuracy": acc,
        "f1_macro": f1
    }


# --------------------------------------------------------------------------
# 4. Main Training Engine
# --------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("        INITIALIZING TWITTER BERT FINE-TUNING PIPELINE")
    print("=" * 70)

    # Device Diagnostic Check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Target Computing Core detected: Core Engine running on [{device.upper()}]")
    if device == "cpu":
        print("[!] WARNING: Training BERT on a CPU context will be exceptionally slow.")
        print("    Consider reducing training epochs or using a machine with an active NVIDIA CUDA GPU.")

    # Data Ingestion and Validation Frame Cleanup
    if not os.path.exists(DATASET_PATH):
        print(f"[X] FATAL ERROR: Target file not found at path: {DATASET_PATH}")
        sys.exit(1)

    print("[*] Ingesting CSV Matrix Data Stream...")
    # The dataset has no implicit headers, assign mapping frameworks manually
    df = pd.read_csv(DATASET_PATH, header=None, names=["id", "entity", "sentiment", "text"])
    
    print(f"[*] Raw Dataset Records Shape: {df.shape}")
    
    # Prune null strings or unmappable rows
    df = df.dropna(subset=["text", "sentiment"])
    df["label"] = df["sentiment"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print(f"[*] Cleaned Dataset Records Shape: {df.shape}")
    print("[*] Target Label Distributions:\n", df["sentiment"].value_counts())

    # Split into Train and Validation matrices (80% / 20% split)
    X_texts = df["text"].astype(str).tolist()
    y_labels = df["label"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        X_texts, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    print(f"[*] Arrays split successfully: Train={len(train_texts)} | Validation={len(val_texts)}")

    # Tokenizer Initialization Pass
    print(f"[*] Fetching Transformer Tokenizer pipeline: [{MODEL_NAME}]...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("[*] Tokenizing text data arrays into structural tensors...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=False, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=False, max_length=128)

    # Packaging datasets
    train_dataset = TwitterDataset(train_encodings, train_labels)
    val_dataset = TwitterDataset(val_encodings, val_labels)

    # Instantiating the sequence classifier core weights
    print(f"[*] Initializing Sequence Classifier weights from: [{MODEL_NAME}]")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_MAP),
        id2label=INV_LABEL_MAP,
        label2id=LABEL_MAP
    )

    # Define Dynamic Padding for optimized RAM footprint
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Establish training runtime tracking arguments
    print("[*] Setting up optimization runtime hyperparameters...")
    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=3,                     # Optimize sequence 3 times across entire matrix
        per_device_train_batch_size=16,         # Adjust down to 8 if encountering Out-Of-Memory (OOM) errors
        per_device_eval_batch_size=32,
        warmup_steps=500,                       # Learning rate scheduling steps
        weight_decay=0.01,                      # Regularization term penalty
        logging_dir=LOGS_DIR,
        logging_steps=100,                      # Telemetry logs terminal step cycle interval
        evaluation_strategy="epoch",            # Evaluate at the completion of each epoch sequence
        save_strategy="epoch",                  # Checkpoint save behavior
        load_best_model_at_end=True,            # Fallback model to absolute best weights configuration
        metric_for_best_model="accuracy",
        fp16=torch.cuda.is_available(),         # Enable 16-bit Mixed Precision computation if GPU is live
        report_to="none"                        # Disables tracking overhead connections (wandb/tensorboard)
    )

    # Initialize the Trainer interface context
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Run Optimization Loop
    print("\n" + "=" * 50)
    print("          EXECUTING STOCHASTIC GRADIENT DESCENT PASS")
    print("=" * 50)
    trainer.train()

    # Finalization and Persistent Storage Export Sequence
    print("\n" + "=" * 50)
    print("          SAVING PRODUCTION-READY GRAPH ARTIFACTS")
    print("=" * 50)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print(f"[✓] Optimization cycle terminated smoothly.")
    print(f"[✓] Compiled binary weights saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()