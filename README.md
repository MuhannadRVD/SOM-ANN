# 🧠 Hybrid Fraud Detection Model — SOM + ANN

A hybrid deep learning pipeline that combines **unsupervised** and **supervised** learning to detect fraudulent credit card applications. A Self-Organizing Map (SOM) first identifies anomalies, then an Artificial Neural Network (ANN) is trained on those findings to produce a ranked fraud probability score for every customer.

---

## 📌 How It Works

```
Credit Card Data
      │
      ▼
 [SOM] — Unsupervised anomaly detection
      │
      ▼
 Fraud labels generated from SOM clusters
      │
      ▼
 [ANN] — Supervised binary classifier
      │
      ▼
 Fraud probability score per customer
```

The key idea: the SOM acts as an **automatic labeler** — it spots suspicious outlier neurons on the U-Matrix, and those customers become the positive class for the ANN. This turns an unsupervised problem into a supervised one without any manual labeling.

---

## 📂 Dataset

**`Credit_Card_Applications.csv`**

- Each row is a credit card application
- Features: 15 customer attributes (anonymized)
- Last column (`Class`): whether the application was approved (used only for visualization, not training)

---

## 🔧 Tech Stack

| Library | Purpose |
|---|---|
| `numpy`, `pandas` | Data handling |
| `sklearn` | MinMaxScaler, StandardScaler |
| `minisom` | Self-Organizing Map |
| `matplotlib` | U-Matrix visualization |
| `keras` | ANN classifier |

---

## 🚀 Pipeline Breakdown

### 1. Feature Scaling
All features normalized to `[0, 1]` using `MinMaxScaler` before feeding into the SOM.

### 2. SOM Training
- Grid size: `10×10`
- Input length: `15` features
- Sigma: `1.0` | Learning rate: `0.5`
- Iterations: `100`

### 3. U-Matrix Visualization
The SOM outputs a distance map where:
- ⬛ **Dark regions** → low inter-neuron distance → dense normal clusters
- ⬜ **Bright/white regions** → high inter-neuron distance → anomalies / cluster boundaries
- 🔴 **Red circles** → customers whose application was NOT approved
- 🟩 **Green squares** → customers whose application WAS approved

### 4. Fraud Extraction
Customers mapped to suspicious neurons (white/bright BMUs) are flagged as potential frauds.

### 5. ANN Training (Unsupervised → Supervised)
- Input: all 15 customer features (StandardScaled)
- Architecture: `Dense(2, relu)` → `Dense(1, sigmoid)`
- Loss: `binary_crossentropy` | Optimizer: `adam`
- Output: fraud probability per customer `[0.0 – 1.0]`

### 6. Ranked Output
Customers are sorted by fraud probability score, highest risk at the top.

---

## 📊 Output Example

```
Index      Prediction      Score     
-----------------------------------
15423      1               0.9821
8731       1               0.9754
3302       0               0.1023
...
```

---

## ▶️ How to Run

```bash
# Install dependencies
pip install minisom keras scikit-learn pandas matplotlib

# Run the notebook
jupyter notebook SOM___ANN.ipynb
```

> Make sure `Credit_Card_Applications.csv` is in the same directory as the notebook.

---

## 📁 Project Structure

```
├── SOM___ANN.ipynb               # Main notebook
├── Credit_Card_Applications.csv  # Dataset
└── README.md
```

---

## 💡 Key Concept

> This project demonstrates how unsupervised learning (SOM) can bootstrap supervised learning (ANN) — useful in domains where labeled fraud data is scarce or expensive to obtain.
