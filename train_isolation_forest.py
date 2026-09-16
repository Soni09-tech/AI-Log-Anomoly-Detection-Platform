"""
Baseline model: Isolation Forest (unsupervised).

Trains on the (mostly normal) training split, scores every session in
the test split, and reports precision/recall/F1 — NOT accuracy, since
anomalies are rare and accuracy would look great even from a model that
predicts "normal" every time.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

from features import build_feature_matrix, chronological_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" if (ROOT / "data").exists() else ROOT
OUTPUTS_DIR = ROOT / "outputs" if (ROOT / "outputs").exists() else ROOT
MODELS_DIR = ROOT / "models" if (ROOT / "models").exists() else ROOT


def train_and_evaluate(features_path=None, contamination=0.08):
    if features_path:
        feat_df = pd.read_csv(features_path)
    else:
        feat_df = build_feature_matrix(
            parsed_csv=DATA_DIR / "parsed_logs.csv",
            labels_csv=DATA_DIR / "labels.csv",
        )

    train_df, test_df = chronological_split(feat_df, test_frac=0.25)

    feature_cols = [c for c in feat_df.columns if c not in ("block_id", "label")]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    # Train only on data the model treats as "mostly normal" -- Isolation
    # Forest doesn't need labels, but `contamination` tells it roughly what
    # fraction of training data to expect as outliers.
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train)

    # decision_function: higher = more normal. Flip sign so higher = more anomalous.
    anomaly_score = -model.decision_function(X_test)
    pred = (model.predict(X_test) == -1).astype(int)  # -1 = outlier, 1 = inlier

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_test, anomaly_score)
    except ValueError:
        auc = float("nan")  # only one class present in test split

    cm = confusion_matrix(y_test, pred)

    print("=== Isolation Forest results (test split) ===")
    print(f"Test sessions: {len(test_df)}  |  Actual anomalies: {y_test.sum()}")
    print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  ROC-AUC: {auc:.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(cm)

    output_path = OUTPUTS_DIR / "isolation_forest_results.csv"
    model_path = MODELS_DIR / "isolation_forest.joblib"
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    out_df = test_df[["block_id", "label"]].copy()
    out_df["anomaly_score"] = anomaly_score
    out_df["predicted_anomaly"] = pred
    out_df.to_csv(output_path, index=False)

    joblib.dump({"model": model, "feature_cols": feature_cols}, model_path)
    print(f"\nSaved predictions -> {output_path}")
    print(f"Saved model       -> {model_path}")

    return model, out_df


if __name__ == "__main__":
    train_and_evaluate(features_path=DATA_DIR / "features.csv")
