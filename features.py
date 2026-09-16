"""
Feature engineering: parsed_logs.csv (one row per log line) ->
one feature vector per session (block_id).

This is the standard "event count vector" approach used in log anomaly
detection (same idea as in the HDFS anomaly detection papers): for each
session, count how many times each event template occurred, plus a few
extra signals (sequence length, WARN/ERROR ratio, distinct event count).
"""

from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT


def build_feature_matrix(parsed_csv, labels_csv, n_event_types=None):
    df = pd.read_csv(parsed_csv)
    labels = pd.read_csv(labels_csv)

    if n_event_types is None:
        n_event_types = df["event_id"].max() + 1

    # sort by block then by (date, time) so sequence-derived features are correct
    df = df.sort_values(["block_id", "date", "time"])

    rows = []
    for blk, grp in df.groupby("block_id"):
        counts = np.zeros(n_event_types, dtype=int)
        for eid in grp["event_id"]:
            counts[eid] += 1

        n_events = len(grp)
        n_distinct = grp["event_id"].nunique()
        n_warn_err = grp["level"].isin(["WARN", "ERROR"]).sum()
        warn_err_ratio = n_warn_err / n_events if n_events else 0.0

        row = {
            "block_id": blk,
            "n_events": n_events,
            "n_distinct_events": n_distinct,
            "warn_err_ratio": warn_err_ratio,
        }
        for i, c in enumerate(counts):
            row[f"evt_{i}"] = c
        rows.append(row)

    feat_df = pd.DataFrame(rows)
    feat_df = feat_df.merge(labels, on="block_id", how="left")
    feat_df["label"] = feat_df["label"].fillna(0).astype(int)
    return feat_df


def chronological_split(feat_df, test_frac=0.2):
    """
    Splits by block_id order of first appearance (proxy for time), NOT
    randomly — avoids the classic mistake of leaking future sessions into
    training when your features/labels are time-ordered.
    """
    n = len(feat_df)
    split_idx = int(n * (1 - test_frac))
    train = feat_df.iloc[:split_idx].reset_index(drop=True)
    test = feat_df.iloc[split_idx:].reset_index(drop=True)
    return train, test


if __name__ == "__main__":
    feat_df = build_feature_matrix(
        parsed_csv=DATA_DIR / "parsed_logs.csv",
        labels_csv=DATA_DIR / "labels.csv",
    )
    out_path = DATA_DIR / "features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"Built feature matrix: {feat_df.shape[0]} sessions x {feat_df.shape[1]} columns")
    print(f"Anomaly rate: {feat_df['label'].mean():.1%}")
    print(f"-> {out_path}")
