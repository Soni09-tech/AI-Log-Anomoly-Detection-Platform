"""
Baseline model: Isolation Forest (unsupervised).

Trains on the (mostly normal) training split, scores every session in
the test split, and reports precision/recall/F1 — NOT accuracy, since
anomalies are rare and accuracy would look great even from a model that
predicts "normal" every time.
"""

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix

from features import build_feature_matrix, chronological_split


def train_and_evaluate(features_path=None, contamination=0.08):
    if features_path:
        feat_df = pd.read_csv(features_path)
    else:
        feat_df = build_feature_matrix(
            parsed_csv="/home/claude/log-anomaly-detector/data/parsed_logs.csv",
            labels_csv="/home/claude/log-anomaly-detector/data/labels.csv",
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

    out_df = test_df[["block_id", "label"]].copy()
    out_df["anomaly_score"] = anomaly_score
    out_df["predicted_anomaly"] = pred
    out_df.to_csv("/home/claude/log-anomaly-detector/outputs/isolation_forest_results.csv", index=False)

    joblib.dump(
        {"model": model, "feature_cols": feature_cols},
        "/home/claude/log-anomaly-detector/models/isolation_forest.joblib",
    )
    print("\nSaved predictions -> outputs/isolation_forest_results.csv")
    print("Saved model       -> models/isolation_forest.joblib")

    return model, out_df


if __name__ == "__main__":
    train_and_evaluate(features_path="/home/claude/log-anomaly-detector/data/features.csv")
