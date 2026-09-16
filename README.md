# AI Log Anomaly Detection Platform

Detects anomalous system/application log sessions using ML — a count-based
unsupervised baseline (Isolation Forest) and an optional sequence-based
deep learning model (LSTM Autoencoder), with a Streamlit dashboard for
inspection.

## How it works

```
raw_logs.log ──► log_parser.py ──► parsed_logs.csv (event templates)
                                          │
                                          ▼
                                   features.py ──► features.csv
                                          │           (one row per session:
                                          │            event counts + stats)
                                          ▼
                          train_isolation_forest.py ──► results + model
                          lstm_autoencoder.py (optional, sequence-aware)
                                          │
                                          ▼
                              app/dashboard.py (Streamlit UI)
```

**1. Log parsing (`src/log_parser.py`)** — masks variable tokens
(numbers, IPs, block IDs) in each log line so structurally identical
lines collapse into the same "event template," the way Drain/Drain3
works. Ships a dependency-free implementation so the whole pipeline runs
without internet access; swap in the real `drain3` package for
production use (see comments in the file).

**2. Feature engineering (`src/features.py`)** — groups log lines by
session (block ID) and builds one feature vector per session: how many
times each event template occurred, sequence length, distinct event
count, and WARN/ERROR ratio. Splits chronologically (not randomly) to
avoid leaking future sessions into training.

**3. Baseline model (`src/train_isolation_forest.py`)** — Isolation
Forest, unsupervised, trained on the count vectors. Reports
precision/recall/F1/ROC-AUC (not accuracy — anomalies are rare, so
accuracy is a misleading metric here).

**4. Sequence model (`src/lstm_autoencoder.py`, optional)** — an LSTM
Autoencoder trained only on normal sessions; it learns to reconstruct
normal event sequences, and sessions with high reconstruction error are
flagged as anomalous. This is the natural "level up" for your report:
count-based vs. order-aware detection. Requires `torch`.

**5. Dashboard (`app/dashboard.py`, optional)** — Streamlit app showing
flagged sessions, anomaly score distribution, and a drill-down view into
each session's raw events. Requires `streamlit`.

## Setup

```bash
pip install -r requirements.txt
```

Everything in `pandas`/`numpy`/`scikit-learn` works offline. `drain3`,
`torch`, and `streamlit` are optional upgrades — install them once you
have internet access.

## Run the pipeline

```bash
python data/generate_logs.py          # synthetic HDFS-style logs + labels
python src/log_parser.py              # raw logs -> event templates
python src/features.py                # event templates -> feature vectors
python src/train_isolation_forest.py  # train + evaluate baseline model

# optional
python src/lstm_autoencoder.py        # train + evaluate sequence model
streamlit run app/dashboard.py        # launch dashboard
```

## Swapping in a real dataset

Replace `data/generate_logs.py`'s output with a real dataset from
[Loghub](https://github.com/logpai/loghub) (HDFS and BGL both ship
ground-truth anomaly labels per block/session). Keep the same file
layout:
- `data/raw_logs.log` — raw log lines
- `data/labels.csv` — columns `block_id,label` (1 = anomalous)

Everything downstream (`log_parser.py` onward) works unchanged, since
Loghub's HDFS format matches the `date time pid level component: content`
layout this parser expects. For other datasets (BGL, Linux, Hadoop),
you'll need to adjust `LOG_LINE_RE` in `log_parser.py` to match their
line format.

## Results on the synthetic dataset (Isolation Forest baseline)

| Metric | Score |
|---|---|
| Precision | 0.90 |
| Recall | 0.97 |
| F1 | 0.94 |
| ROC-AUC | 0.98 |

(Synthetic anomalies are somewhat easier to separate than real-world
ones — expect these numbers to drop, especially recall, on a real
dataset like Loghub HDFS. Report both.)

## Suggested next steps for your report

- Compare Isolation Forest vs. LSTM Autoencoder head-to-head on the same
  test split — this comparison alone is a strong "results" section.
- Try One-Class SVM as a second baseline.
- Add a live "streaming" simulation: replay `raw_logs.log` line-by-line
  with a delay, feeding a rolling window into the model, to demo
  near-real-time detection.
- If you want a DeepLog-style approach instead of an autoencoder: train
  an LSTM to predict the *next* event given the previous k events;
  flag a session anomalous if the actual next event isn't in the
  model's top-k predicted events.
