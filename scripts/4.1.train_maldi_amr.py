# train_maldi_amr.py

# === Import libraries ===

import random
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments
from collections import Counter




from maldi_nn.models import MaldiTransformer  # from installed maldi-nn package

import transformers
#print("Transformers version:", transformers.__version__)
from transformers import TrainingArguments
#print("TrainingArguments class loaded from:", TrainingArguments.__module__)



# === Custom PyTorch Dataset ===
class MaldiPeakDataset(Dataset):
    """
    Wraps a .npy peak matrix and a CSV label file into a Dataset object.
    Each sample is a structured dictionary with (mz, intensity) arrays.
    """
    def __init__(self, npy_path, label_csv_path):
        self.data = np.load(npy_path)  # shape: (N, 200, 2)
        self.labels_df = pd.read_csv(label_csv_path)
        self.labels = self.labels_df["label"].astype(int).values

        # Align lengths safely
        min_len = min(self.data.shape[0], len(self.labels))
        self.data = self.data[:min_len]
        self.labels = self.labels[:min_len]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        peak_array = self.data[idx]  # shape (200, 2)
        spectrum = {
            "mz": torch.tensor(peak_array[:, 0], dtype=torch.float32),
            "intensity": torch.tensor(peak_array[:, 1], dtype=torch.float32),
        }
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": spectrum, "labels": label_tensor}

# === Custom classifier model ===
class MaldiClassifier(nn.Module):
    def __init__(self, encoder, embedding_dim, class_weights, hidden_dim=256, num_labels=2):
        super().__init__()
        self.encoder = encoder
        self.class_weights = class_weights
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, input_ids, labels=None):
        emb = self.encoder(input_ids).mean(dim=1)
        logits = self.classifier(emb)
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fn(logits, labels)
        return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}


# === Load encoder from checkpoint ===
ckpt_path = "../checkpoints/MaldiTransformerM.ckpt"
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(ckpt_path, map_location=device)
base_model = MaldiTransformer(**checkpoint['hyper_parameters'])
base_model.load_state_dict(checkpoint['state_dict'], strict=False)
base_model.to(device)
encoder = base_model.transformer

# Infer embedding dimension by passing dummy input
with torch.no_grad():
    dummy_spectrum = {
        "mz": torch.randn(1, 200).to(device),
        "intensity": torch.randn(1, 200).to(device),
    }
    dummy_output = encoder(dummy_spectrum)  # shape: (1, seq_len, emb_dim)
    #print("Encoder output shape:", dummy_output.shape)    
    #embedding_dim = dummy_output.shape[1]
    embedding_dim = dummy_output.shape[-1]




# === Load dataset ===
npy_path = "../aumc_peaks200.npy"
label_csv_path = "../aumc_cleaned_labels_esbl.csv"
full_dataset = MaldiPeakDataset(npy_path, label_csv_path)


label_counts = Counter(full_dataset.labels)
total = sum(label_counts.values())
class_weights = [
    total / label_counts[i] for i in range(2)  # weight for class 0, class 1
]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)


# Initialize classifier
#model = MaldiClassifier(encoder, embedding_dim).to(device)
model = MaldiClassifier(encoder, embedding_dim, class_weights).to(device)


# === Train/validation split ===
train_indices, val_indices = train_test_split(
    list(range(len(full_dataset))),
    test_size=0.2,
    random_state=42,
    stratify=full_dataset.labels
)
train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

# === Evaluation metrics ===
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions  # shape: (batch_size, num_classes)

    # Convert logits to probabilities
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)

    acc = accuracy_score(labels, preds)

    # Binary classification: use probability of class 1
    try:
        auroc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auroc = float('nan')  # fallback if only one class in batch

    return {"accuracy": acc, "auroc": auroc}



# === Custom collate function for nested input ===
def custom_collate_fn(batch):
    mz = [item["input_ids"]["mz"] for item in batch]
    intensity = [item["input_ids"]["intensity"] for item in batch]
    labels = [item["labels"] for item in batch]

    batch_input = {
        "mz": torch.stack(mz),
        "intensity": torch.stack(intensity),
    }
    batch_labels = torch.tensor(labels)
    return {"input_ids": batch_input, "labels": batch_labels}

# === Training configuration ===
import os
os.environ["ACCELERATE_DISABLE_MIXED_PRECISION"] = "true"
os.environ["ACCELERATE_USE_CPU"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Critical additions to avoid distributed initialization
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["LOCAL_RANK"] = "0"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"



training_args = TrainingArguments(
    output_dir="../models/esbl_transformer",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="../logs",
    logging_steps=10,
    load_best_model_at_end=False,
    metric_for_best_model="auroc",
    greater_is_better=True,
    ddp_find_unused_parameters=True,
    save_total_limit=2,
)

# === HuggingFace Trainer wrapper ===
class HuggingFaceWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids=None, labels=None, **kwargs):
        output = self.model(input_ids, labels)
        return {
            "loss": output["loss"] if "loss" in output else None,
            "logits": output["logits"]
        }

trainer = Trainer(
    model=HuggingFaceWrapper(model),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=custom_collate_fn,  # ← Add this line
)

# === Entry point ===
if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    trainer.save_model = lambda *args, **kwargs: None

    import collections
    print(collections.Counter(full_dataset.labels))


    trainer.train()
    trainer.evaluate()

    # === Save the trained model weights ===
    torch.save(model.state_dict(), "../models/maldi_classifier.pth")
    print("Model weights saved to ../models/maldi_classifier.pth")
