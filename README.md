<<<<<<< HEAD
# 🚀 AI Log Anomaly Detection Platform

Detect anomalous system and application log sessions using Machine Learning. This project combines an unsupervised anomaly detection baseline (**Isolation Forest**) with an optional sequence-aware deep learning model (**LSTM Autoencoder**) and provides a **Streamlit dashboard** for interactive analysis.

## 🌐 Live Demo

**Streamlit App:**
https://ai-log-anomoly-detection-platform-kd6zpsyshzngbyx3by8jtg.streamlit.app/

---

## 📌 Overview

Large-scale systems generate massive volumes of logs every day. Identifying abnormal behavior manually is difficult, time-consuming, and error-prone.

This platform automates the process by:

* Parsing raw logs into structured event templates
* Extracting session-level features
* Detecting anomalies using Machine Learning
* Visualizing results through an interactive dashboard

---

## 🏗️ Architecture

```text
raw_logs.log ──► log_parser.py ──► parsed_logs.csv (event templates)
                                          │
                                          ▼
                                   features.py ──► features.csv
                                          │
                                          ▼
                          train_isolation_forest.py ──► results + model
                          lstm_autoencoder.py (optional)
                                          │
                                          ▼
                              app/dashboard.py (Streamlit UI)
```

---

## 📂 Project Structure

```text
AI-Log-Anomoly-Detection-Platform/
│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── generate_logs.py
│   ├── raw_logs.log
│   └── labels.csv
│
├── src/
│   ├── log_parser.py
│   ├── features.py
│   ├── train_isolation_forest.py
│   └── lstm_autoencoder.py
│
├── models/
├── outputs/
├── requirements.txt
└── README.md
```

---

## ⚙️ Features

### 1. Log Parsing

`src/log_parser.py`

* Converts raw logs into event templates
* Masks variable values such as:

  * Numbers
  * IP Addresses
  * Block IDs
* Groups structurally similar log messages

Output:

```text
parsed_logs.csv
```

---

### 2. Feature Engineering

`src/features.py`

Creates one feature vector per session containing:

* Event template counts
* Sequence length
* Number of distinct events
* WARN ratio
* ERROR ratio

Uses chronological splitting to prevent future-data leakage.

Output:

```text
features.csv
```

---

### 3. Isolation Forest (Baseline Model)

`src/train_isolation_forest.py`

An unsupervised anomaly detection algorithm trained on session-level feature vectors.

Evaluation Metrics:

* Precision
* Recall
* F1 Score
* ROC-AUC

---

### 4. LSTM Autoencoder (Optional)

`src/lstm_autoencoder.py`

A sequence-aware deep learning model that:

1. Learns normal event sequences
2. Reconstructs log sequences
3. Calculates reconstruction error
4. Flags high-error sessions as anomalous

Advantages:

* Captures event order
* Learns temporal dependencies
* Detects complex anomalies

Requires:

```bash
pip install torch
```

---

### 5. Streamlit Dashboard

`app/dashboard.py`

Interactive dashboard for:

* Viewing detected anomalies
* Exploring anomaly score distributions
* Investigating individual sessions
* Reviewing raw log events

Launch:

```bash
streamlit run app/dashboard.py
```

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/Soni09-tech/AI-Log-Anomoly-Detection-Platform.git
cd AI-Log-Anomoly-Detection-Platform
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Pipeline

### Generate Synthetic Logs

```bash
python data/generate_logs.py
```

### Parse Logs

```bash
python src/log_parser.py
```

### Generate Features

```bash
python src/features.py
```

### Train Isolation Forest

```bash
python src/train_isolation_forest.py
```

### Train LSTM Autoencoder (Optional)

```bash
python src/lstm_autoencoder.py
```

### Launch Dashboard

```bash
streamlit run app/dashboard.py
```

---

## 📊 Results

### Isolation Forest Performance (Synthetic Dataset)

| Metric    | Score |
| --------- | ----- |
| Precision | 0.90  |
| Recall    | 0.97  |
| F1 Score  | 0.94  |
| ROC-AUC   | 0.98  |

> Note: Synthetic datasets are easier than real-world logs. Performance may decrease when evaluated on production datasets.

---

## 🔄 Using Real Datasets

You can replace the synthetic dataset with datasets from:

* HDFS
* BGL
* Linux Logs
* Hadoop Logs

Recommended source:

**Loghub Dataset Repository**

https://github.com/logpai/loghub

Required files:

```text
data/raw_logs.log
data/labels.csv
```

Example labels format:

```csv
block_id,label
blk_123,0
blk_456,1
```

Where:

* 0 = Normal
* 1 = Anomalous

---

## 🚀 Future Improvements

* Compare Isolation Forest vs LSTM Autoencoder
* Add One-Class SVM baseline
* Real-time log streaming detection
* REST API integration
* Docker deployment
* Kubernetes deployment
* Alerting and monitoring system

---

## 🧰 Technology Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* PyTorch
* Streamlit

---

## 🎯 Use Cases

* Infrastructure Monitoring
* Distributed Systems
* Cloud Platforms
* Security Monitoring
* Application Performance Analysis
* IT Operations Analytics

---

## 👨‍💻 Author

**Soni Kumar**

GitHub Repository:
https://github.com/Soni09-tech/AI-Log-Anomoly-Detection-Platform

Live Demo:
https://ai-log-anomoly-detection-platform-kd6zpsyshzngbyx3by8jtg.streamlit.app/

---

## ⭐ Support

If you found this project useful, consider giving the repository a star on GitHub.
=======
# AI-Powered Log Anomaly Detection Platform

An end-to-end log anomaly detection project that parses HDFS-style logs,
extracts session features, scores sessions with an Isolation Forest, and
provides a Streamlit dashboard with both API and local-model inference.

**Authors:** Soni Kumari and Pranab Kumar

## Project structure

```text
.
├── backend/
│   └── main.py                 FastAPI API and streaming prediction pipeline
├── frontend/
│   ├── app.py                  SentinelLog AI SOC dashboard
│   ├── dashboard.py            Lightweight offline dashboard
│   └── requirements.txt        Pinned dashboard dependencies
├── features.py                 Session feature extraction
├── log_parser.py               Raw-log parsing and template extraction
├── streaming_inference.py      Bounded-memory parser and model inference
├── train_isolation_forest.py   Baseline training/evaluation
├── lstm_autoencoder.py         Optional sequence-model training
├── main.py                     Vercel/FastAPI entry point
├── vercel.json                 Vercel Python routing
├── requirements.txt            API dependencies (no torch or Streamlit)
├── raw_logs.log                Existing source log data
├── labels.csv                  Existing session labels
├── parsed_logs.csv             Existing parsed log data
├── templates.csv               Existing event templates
├── features.csv                Existing session features
├── isolation_forest.joblib     Existing trained model
└── isolation_forest_results.csv Existing evaluation results
```

The pipeline scripts resolve their input/output files relative to their own
location, not the current shell directory. The checked-in data and model are
used as-is by the API and dashboard; inference writes intermediate files only
to a temporary directory.

## Local setup

Use Python 3.10 or newer. In the repository root, install API dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r frontend/requirements.txt
```

Run the API and dashboard in separate PowerShell terminals from the repository
root. The API accepts `.log` and `.txt` files up to 200 MB, processes text in
25,000-line batches, keeps only bounded result previews, and uses temporary
storage for the uploaded stream. Streamlit itself receives uploads in its
server process, so this configuration is deliberately size-limited rather than
claiming to accept arbitrarily large or multi-gigabyte files.

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

```powershell
$env:API_URL = "http://127.0.0.1:8001"
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
```

The repository also includes a `.streamlit/config.toml` file so Streamlit launches in headless mode without the interactive email/setup prompt, which is required in non-interactive terminals such as VS Code and containers.

Open the dashboard URL printed by Streamlit (normally
`http://localhost:8501`). The API is available at `http://127.0.0.1:8001/`,
with interactive documentation at `http://127.0.0.1:8001/docs`.

The dashboard checks `GET /` with a two-second timeout and uses the local model
automatically when the backend cannot be reached. Binary documents, empty
files, malformed requests, and over-limit uploads receive visible errors.

For a lightweight offline-only dashboard, run:

```powershell
streamlit run frontend/dashboard.py
```

It uses the checked-in `isolation_forest.joblib` when the API is unavailable.

## Streamlit Cloud

1. Push this repository to GitHub and create a Streamlit Community Cloud app.
2. Set the app's **Main file path** to `frontend/app.py`.
3. Configure the app's Python dependencies from `frontend/requirements.txt`
   (Streamlit Cloud also accepts a root `requirements.txt`; this repository
   keeps that file API-only).
4. In **App settings → Secrets**, add the deployed backend URL:

   ```toml
   API_URL = "https://your-api-deployment.example"
   ```

5. Deploy. The Streamlit dashboard uses the API URL for remote predictions and
   falls back to the repository's local model if the backend cannot be reached.
   Ensure the deployment contains `isolation_forest.joblib`.

## Vercel API deployment

The repository root contains `main.py`, `vercel.json`, and API-only
`requirements.txt`. Import the repository in Vercel and deploy with the
included configuration. The health endpoint is `GET /`; inference is `POST
/predict` and accepts a `.log` or `.txt` multipart file, a text form field,
or a raw UTF-8 text body.
For Streamlit Cloud, set `API_URL` to the deployed URL in the app secrets.

## Safe full-pipeline test before the demo

These commands test the existing checked-in dataset in a disposable copy so
the original data, model, and result files are not overwritten. Run in
PowerShell from the repository root:

```powershell
$demo = Join-Path $env:TEMP "ai-log-anomaly-demo"
New-Item -ItemType Directory -Force $demo | Out-Null
Copy-Item raw_logs.log, labels.csv, log_parser.py, features.py, train_isolation_forest.py $demo
Push-Location $demo
python log_parser.py
python features.py
python train_isolation_forest.py
Pop-Location
```

This validates parsing, feature extraction, training, and evaluation while
keeping all generated files inside the temporary demo folder.

Then test live API and dashboard inference:

```powershell
# Terminal 1 (repository root)
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

```powershell
# Terminal 2 (repository root)
Invoke-RestMethod http://127.0.0.1:8001/
curl.exe -F "file=@raw_logs.log" http://127.0.0.1:8001/predict
.\.venv\Scripts\python.exe -m streamlit run frontend\app.py --server.address 127.0.0.1 --server.port 8501
```

The health request should return `status: online`; the prediction response
should include `status`, `total_sessions`, `anomalies_detected`, and `results`.
Upload `raw_logs.log` in the dashboard and test both API and local-model
inference. Text-based files are accepted regardless of extension; CSV, JSON,
and other line-oriented text are supported. Files without session IDs are
analyzed as a single file-level session.

## Pipeline details

1. `log_parser.py` masks variable tokens and maps each valid log line to a
   block/session ID and event-template ID.
2. `features.py` aggregates event counts and session statistics into one row
   per session.
3. `train_isolation_forest.py` trains and evaluates the count-based baseline.
4. `lstm_autoencoder.py` is an optional sequence-based experiment and requires
   PyTorch; it is not required by the API or dashboard.

The API and dashboard only read the existing model for prediction. To run
training on your working copy, first back up the source data/model or run the
training pipeline in a disposable copy as shown above.
>>>>>>> 10a25f3 (Fix website loading and backend issues)
