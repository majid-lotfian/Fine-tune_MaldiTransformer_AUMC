# train_maldi_amr.py
# Fine-tune a pre-trained MaldiTransformer on MALDI-TOF spectra (binary AMR)

import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

from transformers import Trainer, TrainingArguments, TrainerCallback

from maldi_nn.models import MaldiTransformer  # from installed maldi-nn package

# ----------------------------
# Config (adjust as you like)
# ----------------------------
FREEZE_EPOCHS = 0        # freeze encoder for the first N epochs, then unfreeze
USE_FOCAL = False         # use focal loss instead of weighted CE
FOCAL_GAMMA = 1.5        # focal loss gamma (1.0–2.0 is a good sweep)
DROPOUT_P = 0.20         # classifier dropout
SEED = 42

# ----------------------------
# Dataset
# ----------------------------
class MaldiPeakDataset(Dataset):
    """
    Wraps a .npy peak matrix and a CSV label file into a Dataset object.
    Each sample is a dict with (mz, intensity) arrays and an integer label.
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
        mz = torch.tensor(peak_array[:, 0], dtype=torch.float32)
        intensity = torch.tensor(peak_array[:, 1], dtype=torch.float32)

        # --- Normalize safely ---
        # 1) Clip outliers (99th percentile) to prevent huge values
        mz = torch.clamp(mz, min=mz.quantile(0.001), max=mz.quantile(0.999))
        intensity = torch.clamp(intensity, min=intensity.quantile(0.001), max=intensity.quantile(0.999))

        # 2) Optionally log-transform intensity (stabilizes dynamic range)
        intensity = torch.log1p(intensity - intensity.min())

        # 3) Standardize to mean 0 / std 1 per spectrum
        mz = (mz - mz.mean()) / (mz.std() + 1e-6)
        intensity = (intensity - intensity.mean()) / (intensity.std() + 1e-6)

        spectrum = {"mz": mz, "intensity": intensity}
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return {"input_ids": spectrum, "labels": label_tensor}


def custom_collate_fn(batch):
    mz = [item["input_ids"]["mz"] for item in batch]
    intensity = [item["input_ids"]["intensity"] for item in batch]
    labels = [item["labels"] for item in batch]

    batch_input = {
        "mz": torch.stack(mz),
        "intensity": torch.stack(intensity),
    }
    batch_labels = torch.tensor(labels, dtype=torch.long)
    return {"input_ids": batch_input, "labels": batch_labels}

# ----------------------------
# Focal loss (logits expected)
# ----------------------------
class FocalLoss(nn.Module):
    """
    Multi-class focal loss with per-class alpha (weights).
    - logits: (B, C)
    - target: (B,) with class indices
    """
    def __init__(self, alpha=None, gamma=1.5, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, target):
        # CE per-sample
        ce = F.cross_entropy(logits, target, reduction="none")
        # Prob of the true class
        pt = torch.exp(-ce)  # pt = softmax(logits)[range(B), target]
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            focal = alpha_t * focal
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal

# ----------------------------
# Model head
# ----------------------------
class MaldiClassifier(nn.Module):
    def __init__(self, encoder, embedding_dim, use_focal=True, alpha=None,
                 hidden_dim=256, num_labels=2, dropout_p=0.2, focal_gamma=1.5):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_labels),
        )
        self.use_focal = use_focal
        self.focal = FocalLoss(alpha=alpha, gamma=focal_gamma) if use_focal else None
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None):
        # encoder returns (B, seq_len, D); first token is CLS
        seq = self.encoder(input_ids)            # (B, 201, D)
        cls = seq[:, 0, :]                       # (B, D)
        logits = self.classifier(cls)            # (B, 2)
        if labels is None:
            return {"logits": logits}
        
        
        #if not torch.isfinite(logits).all():
         #   print("NaN or Inf detected in logits during forward pass!")
          #  print("Sample logits:", logits[:5])
           # print("Any NaN in encoder output:", not torch.isfinite(seq).all())

        
        loss = self.focal(logits, labels) if self.use_focal else self.ce(logits, labels)
        return {"logits": logits, "loss": loss}

# ----------------------------
# Metrics
# ----------------------------
def compute_metrics(pred):
    labels = np.asarray(pred.label_ids)
    logits = np.asarray(pred.predictions)            # (N, 2)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=1)

    acc = accuracy_score(labels, preds)
    try:
        auroc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auroc = float("nan")
    try:
        auprc = average_precision_score(labels, probs[:, 1])
    except ValueError:
        auprc = float("nan")

    return {"accuracy": acc, "auroc": auroc, "auprc": auprc}

# ----------------------------
# Callback: unfreeze encoder
# ----------------------------
class UnfreezeEncoderCallback(TrainerCallback):
    def __init__(self, freeze_epochs: int = 0):
        self.freeze_epochs = freeze_epochs
        self.unfroze = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self.freeze_epochs > 0:
            # freeze encoder at the very start
            model = kwargs["model"].model  # HuggingFaceWrapper(model).model
            for p in model.encoder.parameters():
                p.requires_grad = False

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.freeze_epochs == 0 or self.unfroze:
            return
        # state.epoch is float; unfreeze when we've *completed* freeze_epochs
        if state.epoch is not None and state.epoch >= self.freeze_epochs:
            model = kwargs["model"].model
            for p in model.encoder.parameters():
                p.requires_grad = True
            self.unfroze = True
            print(f"🔓 Unfroze encoder after epoch {int(state.epoch)}.")

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Environment knobs to keep Trainer single-process on Snellius login node
    os.environ["ACCELERATE_DISABLE_MIXED_PRECISION"] = "true"
    os.environ["ACCELERATE_USE_CPU"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    # Repro
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Load encoder from checkpoint ---
    ckpt_path = "../checkpoints/MaldiTransformerM.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    base_model = MaldiTransformer(**checkpoint["hyper_parameters"])
    base_model.load_state_dict(checkpoint["state_dict"], strict=False)
    base_model.to(device)
    encoder = base_model.transformer

    # Infer embedding dimension
    with torch.no_grad():
        dummy_spectrum = {"mz": torch.randn(1, 200).to(device),
                          "intensity": torch.randn(1, 200).to(device)}
        dummy_output = encoder(dummy_spectrum)      # (1, seq_len, emb_dim)
        embedding_dim = int(dummy_output.shape[-1])

    # --- Load dataset ---
    npy_path = "../aumc_peaks200.npy"
    label_csv_path = "../aumc_cleaned_labels_esblV2.csv"
    full_dataset = MaldiPeakDataset(npy_path, label_csv_path)

    # Class stats & focal alpha (normalized inverse frequency)
    label_counts = Counter(full_dataset.labels)
    total = sum(label_counts.values())
    print(dict(label_counts))

    freq = torch.tensor([label_counts.get(0, 0) / total,
                         label_counts.get(1, 0) / total], device=device, dtype=torch.float32)
    inv_freq = 1.0 / torch.clamp(freq, min=1e-6)
    alpha = inv_freq / inv_freq.sum()   # normalized, sums to 1.0

    # --- Split ---
    train_idx, val_idx = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2,
        random_state=SEED,
        stratify=full_dataset.labels
    )
    train_labels = full_dataset.labels[train_idx]
    val_labels   = full_dataset.labels[val_idx]
    print("Full:", Counter(full_dataset.labels))
    print("Train:", Counter(train_labels))
    print("Val:", Counter(val_labels))

    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    # --- Build model ---
    model_core = MaldiClassifier(
        encoder=encoder,
        embedding_dim=embedding_dim,
        use_focal=USE_FOCAL,
        alpha=alpha if USE_FOCAL else None,
        hidden_dim=256,
        num_labels=2,
        dropout_p=DROPOUT_P,
        focal_gamma=FOCAL_GAMMA,
    ).to(device)

    # HF wrapper so Trainer expects dict(loss, logits)
    class HuggingFaceWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, input_ids=None, labels=None, **kwargs):
            out = self.model(input_ids, labels)
            return {"loss": out.get("loss", None), "logits": out["logits"]}

    wrapped = HuggingFaceWrapper(model_core)

    # --- Training args ---
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
        report_to=[],  # disable wandb/hf logging if not configured
    )

    trainer = Trainer(
        model=wrapped,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=custom_collate_fn,
        callbacks=[UnfreezeEncoderCallback(FREEZE_EPOCHS)],
    )

    # avoid saving mid-epoch/safetensors issues
    trainer.save_model = lambda *args, **kwargs: None

    # --- Train/Eval ---
    trainer.train()
    eval_out = trainer.evaluate()
    print(eval_out)

    # --- Save final head+encoder weights for downstream use ---
    torch.save(model_core.state_dict(), "../models/maldi_classifierV2.pth")
    print("Model weights saved to ../models/maldi_classifierV2.pth")
