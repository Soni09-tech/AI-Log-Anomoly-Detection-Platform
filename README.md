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
