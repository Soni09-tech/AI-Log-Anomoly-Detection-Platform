"""
Sequence-based model: LSTM Autoencoder.

Unlike the Isolation Forest (which only looks at event COUNTS per
session, losing order), this model looks at the actual EVENT SEQUENCE.
It learns to reconstruct normal sequences; sessions it reconstructs
badly (high reconstruction error) are flagged as anomalous.

Requires torch, which isn't installed in this sandbox (no internet).
Run this on your own machine:
    pip install torch pandas scikit-learn

This is the natural "level up" from the Isolation Forest baseline for
your report: compare count-based vs sequence-based detection.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT
OUTPUTS_DIR = ROOT / "outputs" if (ROOT / "outputs").exists() else ROOT
MODELS_DIR = ROOT / "models" if (ROOT / "models").exists() else ROOT

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
except ImportError as e:
    raise ImportError(
        "This module needs PyTorch. Install it with: pip install torch"
    ) from e


PAD_VALUE = -1  # padding id, distinct from real event ids (which start at 0)


def build_sequences(parsed_csv):
    """Turns parsed_logs.csv into one ordered list-of-event-ids per session."""
    df = pd.read_csv(parsed_csv).sort_values(["block_id", "date", "time"])
    seqs = df.groupby("block_id")["event_id"].apply(list).to_dict()
    return seqs


class SeqDataset(Dataset):
    def __init__(self, sequences, max_len, n_event_types):
        self.sequences = sequences
        self.max_len = max_len
        self.n_event_types = n_event_types

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx][: self.max_len]
        padded = seq + [PAD_VALUE] * (self.max_len - len(seq))
        # one-hot encode; padding rows are all-zero
        x = np.zeros((self.max_len, self.n_event_types), dtype=np.float32)
        for i, eid in enumerate(padded):
            if eid != PAD_VALUE:
                x[i, eid] = 1.0
        return torch.from_numpy(x)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_event_types, hidden_dim=32, latent_dim=16):
        super().__init__()
        self.encoder = nn.LSTM(n_event_types, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, n_event_types)

    def forward(self, x):
        seq_len = x.size(1)
        _, (h, _) = self.encoder(x)
        z = self.to_latent(h[-1])                     # (batch, latent_dim)
        h0 = self.from_latent(z).unsqueeze(1)          # (batch, 1, hidden_dim)
        h0 = h0.repeat(1, seq_len, 1)                  # feed same context at every step
        dec_out, _ = self.decoder(h0)
        recon = self.output(dec_out)                   # (batch, seq_len, n_event_types)
        return recon


def train_autoencoder(parsed_csv, labels_csv, epochs=15, batch_size=32, max_len=10):
    seqs = build_sequences(parsed_csv)
    labels = pd.read_csv(labels_csv).set_index("block_id")["label"].to_dict()

    block_ids = list(seqs.keys())
    n_event_types = max(max(s) for s in seqs.values()) + 1

    # train only on sessions labeled normal (label == 0) -- the model
    # should never see anomalous sequences during training
    train_ids = [b for b in block_ids if labels.get(b, 0) == 0]
    test_ids = block_ids  # evaluate reconstruction error on everything

    train_seqs = [seqs[b] for b in train_ids]
    train_ds = SeqDataset(train_seqs, max_len, n_event_types)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder(n_event_types)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(train_ds):.4f}")

    # score every session by reconstruction error
    model.eval()
    test_seqs = [seqs[b] for b in test_ids]
    test_ds = SeqDataset(test_seqs, max_len, n_event_types)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for batch in test_loader:
            recon = model(batch)
            err = ((recon - batch) ** 2).mean(dim=(1, 2))
            errors.extend(err.tolist())

    result = pd.DataFrame({
        "block_id": test_ids,
        "reconstruction_error": errors,
        "label": [labels.get(b, 0) for b in test_ids],
    })

    # pick threshold as the 95th percentile of TRAIN reconstruction error
    # (i.e. "normal" sessions), anything above it on the full set = anomaly
    with torch.no_grad():
        train_errors = []
        for batch in DataLoader(train_ds, batch_size=batch_size, shuffle=False):
            recon = model(batch)
            err = ((recon - batch) ** 2).mean(dim=(1, 2))
            train_errors.extend(err.tolist())
    threshold = float(np.percentile(train_errors, 95))
    result["predicted_anomaly"] = (result["reconstruction_error"] > threshold).astype(int)

    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(
        result["label"], result["predicted_anomaly"], average="binary", zero_division=0
    )
    print(f"\nThreshold (95th pct of train error): {threshold:.4f}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")

    output_path = OUTPUTS_DIR / "lstm_autoencoder_results.csv"
    model_path = MODELS_DIR / "lstm_autoencoder.pt"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_path, index=False)
    torch.save(model.state_dict(), model_path)
    print(f"Saved predictions -> {output_path}")
    print(f"Saved model       -> {model_path}")
    return model, result


if __name__ == "__main__":
    train_autoencoder(
        parsed_csv=DATA_DIR / "parsed_logs.csv",
        labels_csv=DATA_DIR / "labels.csv",
    )
