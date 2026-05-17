# 🛡️ Multiclass Cyber-Physical Attack Detection on Smart IP Cameras

> Extending IALSTM with Focal Loss for fine-grained 9-class attack classification on the TON_IoT network dataset. First multiclass extension of the binary IALSTM framework, with a novel CNN-BiLSTM-Attention hybrid as a comparative methodology.

---

## 📌 Overview

Smart IP cameras are high-value targets for cyber-physical attacks — from ransomware encrypting stored footage to MITM interception of live streams. Existing ML-based IDS solutions detect *whether* an attack is happening but not *what kind*. This project solves that:

- Transforms a **binary** (Normal / Attack) detection system into a **9-class fine-grained attack classifier**
- Applies **Focal Loss** to handle extreme class imbalance in real IoT traffic (Normal = 75%, MITM < 1%)
- Proposes and compares **two deep learning architectures** on the TON_IoT benchmark

| | Method 1 | Method 2 |
|---|---|---|
| **Model** | IALSTM + Focal Loss | CNN-BiLSTM-Attention + Focal Loss |
| **Accuracy** | 74.94% | **93.25%** |
| **Macro F1** | 0.5696 | **0.9267** |
| **Weighted F1** | 0.7076 | **0.9339** |
| **Parameters** | ~22,000 (student) | ~51,000 |
| **Classes with F1 = 0** | 3 (MITM, Password, XSS) | **None** |

---

## 🧠 Key Contributions

1. **First multiclass IALSTM extension** — modifies the binary Hazarika et al. (2025) framework to classify 9 distinct attack types using Focal Loss and a 10-class softmax output.
2. **Focal Loss for IoT IDS** — applies `FL(pt) = -α·(1−pt)^γ·log(pt)` (γ=2.0, α=0.25) to suppress easy majority-class gradient dominance without any data augmentation or oversampling.
3. **CNN-BiLSTM-Attention hybrid** — three-stage architecture combining local feature extraction (CNN), bidirectional temporal modelling (BiLSTM), and dynamic timestep weighting (Multi-Head Attention).
4. **Knowledge Distillation** (Method 1) — 8× model compression (180K → 22K params) from a teacher LSTM to an edge-deployable student IALSTM.
5. **Ablation & reproducibility** — 5-run statistical validation; Method 2 Macro F1 std dev = 0.0013 across seeds 42, 123, 256, 512, 1024.

---

## 🎯 Attack Classes

| Class | Raw Samples | Test Sequences | Description |
|-------|:-----------:|:--------------:|-------------|
| Normal | 50,000 | 8,408 | Benign network traffic |
| DoS / DDoS | 40,000 | 7,797 | Denial-of-Service / Distributed DoS |
| Backdoor | 20,000 | 3,737 | Persistent unauthorized remote access |
| Injection | 20,000 | 3,993 | SQL / command injection |
| Password | 20,000 | 3,972 | Brute-force / credential attacks |
| Ransomware | 20,000 | 2,947 | File encryption malware |
| Scanning | 20,000 | 4,000 | Network reconnaissance / port scanning |
| XSS | 20,000 | 3,027 | Cross-Site Scripting |
| MITM | 1,043 | 208 | Man-in-the-Middle (rarest — <1% of data) |
| **Total** | **211,043** | **38,089** | |

---

## 📁 Project Structure

```
cyber-attack-detection/
├── notebooks/
│   └── smart_camera_multiclass_TON_IoT.ipynb   # Full experiment notebook (14 cells)
├── data/
│   └── raw/
│       └── train_test_network.csv               # TON_IoT dataset (not in repo — see below)
├── models/
│   ├── ialstm_multiclass_*.h5                   # Method 1: trained IALSTM student
│   ├── cnn_bilstm_multiclass_*.h5               # Method 2: trained CNN-BiLSTM-Attention
│   ├── method2_rf_*.pkl                         # Random Forest baseline
│   ├── scaler.pkl                               # MinMaxScaler (fit on train set)
│   └── label_encoder.pkl                        # LabelEncoder for attack classes
├── checkpoints/                                 # Best validation-loss weights per run
├── plots/
│   ├── cnn_bilstm_training_history.png          # Loss/accuracy/precision/recall curves
│   ├── ialstm_training_history.png
│   ├── confusion_matrices.png                   # Side-by-side normalized CMs
│   ├── confusion_matrix_normalized_FIXED.png
│   ├── confusion_matrix_raw_FIXED.png
│   ├── per_class_f1_comparison.png              # Bar chart: Method 1 vs 2 per class
│   ├── focal_loss_explanation.png               # FL vs CE curve (γ = 0.5/1/2/5)
│   ├── model_compression_analysis.png           # Teacher→Student compression scatter
│   └── overall_comparison.png                  # Accuracy / Macro F1 / Weighted F1 bars
├── results/
│   └── final_results_*.json                     # Full metrics summary
├── logs/
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone

```bash
git clone https://github.com/<your-username>/cyber-attack-detection.git
cd cyber-attack-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

```
tensorflow==2.15.0          # Deep learning backbone
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
imbalanced-learn==0.11.0
matplotlib==3.7.2
seaborn==0.12.2
shap==0.42.1                # Explainability analysis
optuna==3.3.0               # Hyperparameter search
plotly==5.16.1
joblib==1.3.2
tensorflow-model-optimization==0.7.5
```

**Hardware used:** NVIDIA RTX 3050 6GB (CUDA 11.8) — training time ~20–35 min per model.

### 3. Download the dataset

TON_IoT is freely available for academic research from UNSW Sydney:

👉 **[https://research.unsw.edu.au/projects/toniot-datasets](https://research.unsw.edu.au/projects/toniot-datasets)**

Download the **Network dataset** folder and place the CSV at:

```
data/raw/train_test_network.csv
```

> ⚠️ The dataset (~28.5 MB compressed, ~500 MB–1 GB in memory during training) is **not included** in this repository.

---

## 🚀 Usage

```bash
jupyter notebook notebooks/smart_camera_multiclass_TON_IoT.ipynb
```

| Cell | What it does |
|------|-------------|
| 1 | Installs dependencies, imports, sets working directory |
| 2 | GPU / CUDA configuration |
| 3 | **Focal Loss** implementation (`γ=2.0, α=0.25`) |
| 4 | `CONFIG` dict — all hyperparameters in one place |
| 5 | Load TON_IoT dataset, handle multiple filename variants |
| 6 | Explore labels, map raw attack names → canonical classes |
| 7 | Preprocessing pipeline (dedup → median impute → MinMax scale → sequences) |
| 8 | Build Teacher LSTM + Student IALSTM (Method 1) |
| 9 | Train student via Knowledge Distillation (soft + hard targets) |
| 10 | Evaluate Method 1 — per-class report + confusion matrix |
| 11 | Build + train CNN-BiLSTM-Attention (Method 2) |
| 12 | Evaluate Method 2 — per-class report + confusion matrix |
| 13 | Side-by-side comparison table |
| 14+ | Training history plots, per-class F1 bar chart, SHAP analysis |

---

## 🏗️ Architectures

### Method 1 — IALSTM + Focal Loss (Knowledge Distillation)

```
TEACHER (~180K params)
  Input(30, 41) → LSTM(128) → BN → LSTM(64) → BN → Dense(64, ReLU) → Dropout(0.3) → Dense(9, softmax)
                                    ↓
                        Knowledge Distillation
                        Temperature T = 4.0  |  α_KD = 0.7
                        L = (1-α)·Focal(hard) + α·KL(soft)
                                    ↓
STUDENT / IALSTM (~22K params — 8× compression)
  Input(30, 41) → LSTM(32) → BN → LSTM(16) → BN → Dense(32, ReLU) → Dropout(0.2) → Dense(9, softmax)
```

### Method 2 — CNN-BiLSTM-Attention + Focal Loss

```
Input(30, 41)
  → [Conv1D(32, k=3) → BN → ReLU → Dropout(0.3)]     # (batch, 30, 32)
  → [Conv1D(64, k=3) → BN → ReLU → Dropout(0.3)]     # (batch, 30, 64)
  → MaxPool1D(2)                                       # (batch, 15, 64)
  → BiLSTM(32 per dir → 64 total, return_seq=True)    # (batch, 15, 64)
  → BiLSTM(16 per dir → 32 total, return_seq=True)    # (batch, 15, 32)
  → MultiHeadAttention(heads=2, key_dim=16)
  → Residual: x = x + Attention(x)
  → GlobalAveragePooling1D                             # (batch, 32)
  → Dense(64, ReLU) → Dropout(0.3)
  → Dense(9, softmax)                                  # ~51K params total
```

**Why bidirectional?** Forward LSTM captures chronological attack development; reverse LSTM captures patterns more distinctive when viewed retrospectively (especially useful for scanning and MITM flows).

**Why attention?** Ablation study showed removing Multi-Head Attention drops Macro F1 from 0.9267 → ~0.88, with the largest degradation on MITM and XSS.

---

## 📊 Results

### Per-Class F1-Score Comparison

| Attack Class | IALSTM + FL | CNN-BiLSTM + FL | Test Support |
|---|:-----------:|:---------------:|:------------:|
| Backdoor | 0.9647 | **1.0000** | 3,737 |
| DoS / DDoS | 0.8446 | **0.9256** | 7,797 |
| Injection | 0.4875 | **0.7354** | 3,993 |
| MITM | 0.0000 ❌ | **0.9417** ✅ | 208 |
| Normal | **0.9996** | 0.9999 | 8,408 |
| Password | 0.0000 ❌ | **0.9200** ✅ | 3,972 |
| Ransomware | 0.8636 | **0.9927** | 2,947 |
| Scanning | 0.9666 | **0.9995** | 4,000 |
| XSS | 0.0000 ❌ | **0.8258** ✅ | 3,027 |
| **Macro F1** | 0.5696 | **0.9267** | 38,089 |
| **Accuracy** | 74.94% | **93.25%** | 38,089 |

### Class Collapse in IALSTM (Method 1)

The IALSTM student (22K params) hits **complete class collapse** on MITM, Password, and XSS despite Focal Loss training. Root cause: insufficient representational capacity — 32/16-unit LSTM layers cannot simultaneously maintain 10 distinct decision boundaries. The confusion matrix shows MITM flows predicted as DoS/DDoS, Password flows as Normal/DoS, and XSS flows as Normal. This is a capacity problem, not a loss function problem.

### Focal Loss Gamma Ablation

| γ | Observation |
|---|-------------|
| 0.5 | Too weak — MITM and Password recall degrade 8–12% vs baseline |
| **2.0** | **Optimal** — best Macro F1, stable training |
| 3.0 | Too aggressive — training instability in early epochs |

### Statistical Reproducibility (Method 2, 5 runs)

| Seed | Macro F1 |
|------|----------|
| 42 | 0.9267 |
| 123 | 0.9241 |
| 256 | 0.9278 |
| 512 | 0.9259 |
| 1024 | 0.9263 |
| **Mean ± Std** | **0.9262 ± 0.0013** |

---

## 🔬 Dataset — TON_IoT

| Property | Detail |
|---|---|
| **Source** | UNSW Canberra Cyber / Nour Moustafa, 2021 |
| **Domain** | Network traffic from IoT/IIoT edge devices |
| **Features** | 44 total (connection, statistics, user, violation) |
| **Used features** | 41 numeric (after preprocessing) |
| **Train split** | 133,310 sequences (70%) |
| **Val split** | 19,045 sequences (12.5%) |
| **Test split** | 38,089 sequences (17.5%) |
| **Sequence length** | 30 timesteps |
| **Class imbalance** | Normal ≈ 75%, MITM < 1% |
| **Preprocessing** | Dedup → median impute → MinMax [0,1] → label encode → stratified split |

---

## 🔮 Future Work

- **CNN-BiLSTM as teacher + IALSTM-scale student** — Knowledge Distillation using the better-performing Method 2 as the teacher, targeting edge-deployable multiclass detection without class collapse.
- **Multimodal fusion** — combining network traffic features with device telemetry (health metrics, environmental sensors) for detection of physical tampering scenarios.
- **Federated learning** — privacy-preserving collaborative training across distributed camera networks without raw data sharing.
- **Real-time edge evaluation** — hardware testbed benchmarking for latency and energy consumption on embedded devices (e.g., Jetson Orin Nano).

---

## 📄 References

```bibtex
@article{hazarika2025ialstm,
  title   = {Real-Time Detection of Cyber-Physical Attacks on Smart-IP-Camera
             Using Network and Telemetry Data},
  author  = {Hazarika, A. and Choudhury, N. and Shu, L. and Su, Q.},
  journal = {IEEE Transactions on Industrial Cyber-Physical Systems},
  volume  = {3},
  pages   = {251--261},
  year    = {2025}
}

@inproceedings{lin2017focal,
  title     = {Focal Loss for Dense Object Detection},
  author    = {Lin, T.-Y. and Goyal, P. and Girshick, R. and He, K. and Doll{\'a}r, P.},
  booktitle = {Proc. IEEE ICCV},
  pages     = {2980--2988},
  year      = {2017}
}

@article{moustafa2021toniot,
  title   = {A New Distributed Architecture for Evaluating AI-Based Security Systems
             at the Edge: Network TON\_IoT Datasets},
  author  = {Moustafa, N.},
  journal = {Sustainable Cities and Society},
  volume  = {72},
  year    = {2021}
}
```

---

## 📜 License

For academic and research use. Dataset subject to UNSW TON_IoT terms of use.
