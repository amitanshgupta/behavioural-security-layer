# ============================================================
# BiLSTM Training on CERT Data — Google Colab
# Run this in Colab with GPU runtime enabled
# Runtime → Change runtime type → T4 GPU
# ============================================================

# Cell 1 — Install dependencies
# !pip install torch pandas scikit-learn

# Cell 2 — Mount Drive and upload cert_features.csv
# from google.colab import drive
# drive.mount('/content/drive')

# Cell 3 — Imports
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import os

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Cell 4 — Config
SEQ_LEN    = 30     # 30-day rolling window per user
BATCH_SIZE = 512
EPOCHS     = 20
LR         = 1e-3
HIDDEN     = 128
N_LAYERS   = 2
DROPOUT    = 0.3

# CERT temporal features for BiLSTM
LSTM_FEATURES = [
    "total_logons", "after_hours_logons", "unique_pcs",
    "usb_connects", "file_events", "executable_access",
    "after_hours_files", "unique_files",
    "http_requests", "suspicious_domains",
    "external_emails", "high_risk_emails",
    "logon_hour_mean", "logon_hour_std",
    "total_logons_role_zscore", "file_events_role_zscore",
    "usb_connects_role_zscore",
    "psych_risk_score", "N", "C",
]


# Cell 5 — Load data
def load_cert_sequences(path: str) -> tuple:
    """
    Load CERT feature matrix and build per-user sequences.
    Each user's daily records become a time series.
    Sequences of SEQ_LEN days are created with sliding window.
    """
    print("Loading data...")
    df = pd.read_csv(path, low_memory=False)

    features = [f for f in LSTM_FEATURES if f in df.columns]
    print(f"Using {len(features)} features")

    # Sort by user and date for temporal ordering
    df["date_only"] = pd.to_datetime(df["date_only"])
    df = df.sort_values(["user", "date_only"]).reset_index(drop=True)

    # Normalise features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features].fillna(0))

    # Build sequences per user
    X_seqs, y_seqs = [], []
    for user, group in df.groupby("user"):
        X_user = group[features].values
        y_user = group["label_binary"].values

        if len(X_user) < SEQ_LEN:
            continue

        for i in range(len(X_user) - SEQ_LEN):
            X_seqs.append(X_user[i : i + SEQ_LEN])
            y_seqs.append(y_user[i + SEQ_LEN - 1])

    X = np.array(X_seqs, dtype=np.float32)
    y = np.array(y_seqs, dtype=np.float32)
    print(f"Sequences: {len(X):,} | Shape: {X.shape}")
    print(f"Attack rate: {y.mean()*100:.1f}%")
    return X, y, features, scaler


# Cell 6 — Dataset
class CERTSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(np.array(X, copy=True))
        self.y = torch.FloatTensor(np.array(y, copy=True))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Cell 7 — Model
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden=HIDDEN,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if n_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _  = self.bilstm(x)
        last    = out[:, -1, :]
        last    = self.dropout(last)
        return self.classifier(last).squeeze(1)


# Cell 8 — Training loop
def train(model, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    best_val_loss = float("inf")
    best_state    = None
    patience_ctr  = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds    = model(X_batch.to(DEVICE))
                val_loss += criterion(
                    preds, y_batch.to(DEVICE)
                ).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:>2}/{EPOCHS} | "
              f"train={train_loss:.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= 5:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# Cell 9 — Evaluate
def evaluate(model, loader):
    model.eval()
    all_proba, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(DEVICE))
            all_proba.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    proba  = np.array(all_proba)
    labels = np.array(all_labels)
    y_pred = (proba >= 0.5).astype(int)

    f1  = f1_score(labels, y_pred, zero_division=0)
    auc = roc_auc_score(labels, proba)
    fpr = (
        ((y_pred == 1) & (labels == 0)).sum() /
        ((labels == 0).sum() + 1e-9)
    )
    print(f"F1={f1:.4f} | AUC={auc:.4f} | FPR={fpr:.4f}")
    print(classification_report(
        labels, y_pred, target_names=["normal", "attack"]
    ))
    return f1, auc, fpr


# Cell 10 — Main
if __name__ == "__main__":
    # Load data
    # In Colab: path = "/content/drive/MyDrive/cert_features.csv"
    path = "data/processed/cert_features.csv"
    X, y, features, scaler = load_cert_sequences(path)

    # Split
    split = int(len(X) * 0.80)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    val_split = int(len(X_train) * 0.85)
    X_tr  = X_train[:val_split]
    X_val = X_train[val_split:]
    y_tr  = y_train[:val_split]
    y_val = y_train[val_split:]

    # Datasets
    train_ds = CERTSequenceDataset(X_tr,   y_tr)
    val_ds   = CERTSequenceDataset(X_val,  y_val)
    test_ds  = CERTSequenceDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False
    )

    print(f"Train: {len(train_ds):,} | "
          f"Val: {len(val_ds):,} | "
          f"Test: {len(test_ds):,}")

    # Build and train model
    model = BiLSTMClassifier(input_size=len(features)).to(DEVICE)
    print(f"Parameters: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    model = train(model, train_loader, val_loader)

    # Evaluate
    print("\n--- Test Results ---")
    f1, auc, fpr = evaluate(model, test_loader)

    # Save
    torch.save({
        "model_state": model.state_dict(),
        "features"   : features,
        "input_size" : len(features),
        "scaler"     : scaler,
        "f1"         : f1,
        "auc"        : auc,
        "fpr"        : fpr,
    }, "bilstm_cert.pt")
    print("Model saved -> bilstm_cert.pt")
    print("Download and move to data/metadata/bilstm_cert.pt")