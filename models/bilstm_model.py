import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    PROCESSED_DIR, METADATA_DIR, CLUSTERING_FEATURES
)
from utils.logger import get_logger

log = get_logger("bilstm_model")

MODEL_PATH = METADATA_DIR / "bilstm_model.pt"

# Sequence parameters — from proposal
SEQ_LEN    = 60    # 60-timestep session sequences
BATCH_SIZE = 256
EPOCHS     = 15
LR         = 1e-3

# Features for BiLSTM — temporal behaviour signals
LSTM_FEATURES = [
    "duration", "src_bytes", "dst_bytes",
    "count", "srv_count",
    "serror_rate", "rerror_rate",
    "same_srv_rate", "diff_srv_rate",
    "error_rate_combined", "service_diversity_score",
    "bytes_ratio", "connection_rate",
    "peer_outlier_score", "auth_failure_rate",
    "privilege_abuse_score",
]


class SequenceDataset(Dataset):
    """
    Builds overlapping sequences of length SEQ_LEN from the DataFrame.
    Each sample = (sequence of SEQ_LEN rows, label of last row).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
        self.X = torch.FloatTensor(np.array(X, copy=True))
        self.y = torch.FloatTensor(np.array(y, copy=True))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]
        label = self.y[idx + self.seq_len - 1]
        return x_seq, label


class BiLSTMClassifier(nn.Module):
    """
    2-layer Bidirectional LSTM classifier.
    Architecture from proposal: 128 hidden units, dropout=0.3
    """
    def __init__(
        self,
        input_size : int,
        hidden_size: int = 128,
        num_layers : int = 2,
        dropout    : float = 0.3,
    ):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size   = input_size,
            hidden_size  = hidden_size,
            num_layers   = num_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        # bidirectional doubles hidden size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.bilstm(x)
        # Use last timestep output
        last    = out[:, -1, :]
        last    = self.dropout(last)
        return self.classifier(last).squeeze(1)


def normalise(
    X_train: np.ndarray,
    X_test : np.ndarray,
) -> tuple:
    """Min-max normalise using train statistics."""
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    rng      = max_vals - min_vals + 1e-9

    X_train_n = (X_train - min_vals) / rng
    X_test_n  = (X_test  - min_vals) / rng

    return X_train_n, X_test_n, min_vals, rng
 

def train_bilstm(
    model      : BiLSTMClassifier,
    train_loader: DataLoader,
    val_loader  : DataLoader,
    epochs      : int = EPOCHS,
    lr          : float = LR,
) -> BiLSTMClassifier:
    """Train with binary cross-entropy, early stopping on val loss."""
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training on: {device}")
    model     = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0
    patience_max  = 5

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds   = model(X_batch)
                val_loss += criterion(preds, y_batch).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        log.info(f"Epoch {epoch:>2}/{epochs} | "
                 f"train_loss={train_loss:.4f} | "
                 f"val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience_max:
                log.info(f"Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate_bilstm(
    model      : BiLSTMClassifier,
    loader     : DataLoader,
    split      : str = "test",
    threshold  : float = 0.5,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    model.eval()

    all_proba, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device))
            all_proba.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    proba  = np.array(all_proba)
    labels = np.array(all_labels)
    y_pred = (proba >= threshold).astype(int)

    f1  = f1_score(labels, y_pred, zero_division=0)
    auc = roc_auc_score(labels, proba)
    fpr = (
        ((y_pred == 1) & (labels == 0)).sum() /
        ((labels == 0).sum() + 1e-9)
    )

    log.info(f"\n--- BiLSTM [{split}] ---")
    log.info(f"F1      : {f1:.4f}")
    log.info(f"ROC-AUC : {auc:.4f}")
    log.info(f"FPR     : {fpr:.4f} ({fpr*100:.1f}%)")
    log.info(f"\n{classification_report(labels, y_pred, target_names=['normal','attack'])}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(labels, y_pred)}")

    return {"f1": f1, "roc_auc": auc, "fpr": fpr,
            "proba": proba, "y_pred": y_pred, "labels": labels}


def run_bilstm() -> dict:
    """Master function — prepare sequences, train, evaluate, save."""
    # Load data
    train_df = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")

    feats    = [f for f in LSTM_FEATURES if f in train_df.columns]
    X_train  = train_df[feats].fillna(0).values
    y_train  = train_df["label_binary"].values
    X_test   = test_df[feats].fillna(0).values
    y_test   = test_df["label_binary"].values

    log.info(f"Features: {len(feats)} | "
             f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Normalise
    X_train, X_test, min_vals, rng = normalise(X_train, X_test)

    # Split train into train/val
    split_idx = int(len(X_train) * 0.85)
    X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    # Build datasets
    train_ds  = SequenceDataset(X_tr,  y_tr)
    val_ds    = SequenceDataset(X_val, y_val)
    test_ds   = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    log.info(f"Sequences — train: {len(train_ds):,} | "
             f"val: {len(val_ds):,} | test: {len(test_ds):,}")

    # Build model
    model = BiLSTMClassifier(input_size=len(feats))
    log.info(f"BiLSTM parameters: "
             f"{sum(p.numel() for p in model.parameters()):,}")

    # Train
    model = train_bilstm(model, train_loader, val_loader)

    # Evaluate
    test_results = evaluate_bilstm(model, test_loader, "test")

    # Save
    torch.save({
        "model_state": model.state_dict(),
        "features"   : feats,
        "min_vals"   : min_vals,
        "rng"        : rng,
        "input_size" : len(feats),
    }, MODEL_PATH)
    log.info(f"Model saved -> {MODEL_PATH}")

    return test_results


if __name__ == "__main__":
    results = run_bilstm()
    print(f"\nFinal Test Results:")
    print(f"  F1      : {results['f1']:.4f}")
    print(f"  ROC-AUC : {results['roc_auc']:.4f}")
    print(f"  FPR     : {results['fpr']*100:.1f}%")