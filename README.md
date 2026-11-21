# RanSMAP Ransomware Classification & Detection Suite

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![License: CC BY-ND 4.0](https://img.shields.io/badge/License-CC%20BY--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nd/4.0/)

An end-to-end machine learning and deep learning ransomware detection suite built on low-level storage and memory access telemetry from the **RanSMAP** benchmark.

This repository features:
1. **A Complete Python 6-CSV Application & Web Dashboard** ([app.py](file:///home/nomad/Documents/Work/Projects/checkPS/app.py)): Ingests, preprocesses, and evaluates multi-channel hypervisor memory and disk telemetry in real time.
2. **A Breakthrough Deep Learning Model** ([temporal_transformer_model.pkl](file:///home/nomad/Documents/Work/Projects/checkPS/temporal_transformer_model.pkl)): Single Optimized Transformer with multi-scale harmonic temporal positional encodings achieving **96.50% Weighted F1 score** and **97.24% Binary F1 score** on 356,476 test operations.
3. **Multi-Model Support**: Seamless side-by-side switching between the baseline ensemble model and the temporal transformer model in both the web UI and CLI.
4. **Archived Experimental Progression** ([experimental/](file:///home/nomad/Documents/Work/Projects/checkPS/experimental/)): Full historical checkpoints detailing the evolution from sequential LSTMs to hybrid networks, nudged architectures, and the final temporal-aware transformer.

---

## Origin & Attribution

This project was initialized by cloning **[@gmh5225](https://github.com/gmh5225)**'s [RanSMAP-Ransomware-Binary-Classification](https://github.com/gmh5225/RanSMAP-Ransomware-Binary-Classification) repository.

The dataset and underlying behavioral telemetry originate from the seminal RanSMAP benchmark by **Manabu Hirano** and **Ryotaro Kobayashi**:
- **Dataset**: [RanSMAP: Open Dataset of Ransomware Storage and Memory Access Patterns](https://www.kaggle.com/datasets/hiranomanabu/ransmap-2024-ransomware-behavioral-features)
- **Primary Citation**: 
  > M. Hirano and R. Kobayashi, *"RanSMAP: Open Dataset of Ransomware Storage and Memory Access Patterns for Creating Deep Learning Based Ransomware Detectors,"* **Computers & Security**, 2024, 104202, ISSN 0167-4048. [DOI: 10.1016/j.cose.2024.104202](https://doi.org/10.1016/j.cose.2024.104202).
- **Secondary Citation**:
  > M. Hirano and R. Kobayashi, *"Machine Learning-based Ransomware Detection Using Low-level Memory Access Patterns Obtained From Live-forensic Hypervisor,"* **2022 IEEE International Conference on Cyber Security and Resilience (CSR)**, 2022, pp. 323-330. [DOI: 10.1109/CSR54599.2022.9850340](https://doi.org/10.1109/CSR54599.2022.9850340).

The initial study analyzed **1,425,904 memory operations** across 12 software categories (6 benign applications and 6 ransomware families).

---

## The Python Application & Preprocessing Pipeline

The application ([app.py](file:///home/nomad/Documents/Work/Projects/checkPS/app.py) and [classifier_pipeline.py](file:///home/nomad/Documents/Work/Projects/checkPS/classifier_pipeline.py)) accepts realistic RanSMAP workload submissions comprising **6 simultaneous CSV files**:

| File Channel | Channel Role | Description |
| :--- | :--- | :--- |
| `mem_read.csv` | Memory | Process memory page read operations |
| `mem_write.csv` | Memory | Process memory page write/modification operations |
| `mem_exec.csv` | Memory | Instruction fetch and code execution operations |
| `mem_readwrite.csv` | Memory | Combined read/write atomic operations |
| `ata_read.csv` | Storage | ATA disk sector read telemetry (auxiliary) |
| `ata_write.csv` | Storage | ATA disk sector write telemetry (auxiliary) |

### Pipeline Architecture:
1. **Multi-Source Ingestion**: Automatically detects headers or headless formats, distinguishes ATA storage channels from memory channels, and filters corrupt rows.
2. **Robust GPA Normalization**: Parses Guest Physical Addresses (GPA) across decimal integers, floating-point numbers, and hexadecimal representations (e.g., `0x80000000`).
3. **Memory Architecture Breakdown**: Aggregates operations across page types:
   - `4 KiB page` (Type 4: standard OS memory page)
   - `2 MiB page` (Type 2: large page allocation)
   - `1 GiB page` (Type 1: huge page allocation)
   - `MMIO access` (Type 0: memory-mapped I/O)
4. **Multi-Model Inference Engine**: Supports dynamic execution with either the classical baseline or the temporal transformer.
5. **Interactive Web Dashboard**: Beautiful dark-mode UI with risk dials, per-file verdict matrices, memory breakdown statistics, and full predictions CSV export.

---

## Evolution of Experimental Models

The core scientific challenge of memory-based ransomware detection is **distinguishing malicious encryption from legitimate high-entropy operations** (such as file compression, shredding, and encryption).

Below is the documented engineering progression in [RanSMAP_Ransomware_Binary_Classification.ipynb](file:///home/nomad/Documents/Work/Projects/checkPS/RanSMAP_Ransomware_Binary_Classification.ipynb) that broke through the initial performance barrier:

```
[Baseline Classical Ensembles]
          │  Plateaued at ~63.9% Accuracy / 51.1% Weighted F1
          ▼
[Sequential 2-Layer LSTM Baseline]
          │  High temporal complexity, sluggish convergence
          ▼
[Pure 2-Layer Transformer]  (Cells 181-182)
          │  Replaced recurrent hidden states with Multi-Head Self-Attention
          ▼
[Hybrid Architectures]       (Cells 183-186)
          ├── Transformer -> LSTM (Hybrid A)
          └── LSTM -> Transformer (Hybrid B)
          │
          ▼
[Architectural Parameter Nudges] (Cell 187)
          │  d_model: 64 -> 128 | nhead: 4 -> 8 | dim_feedforward: 128 -> 256
          │  Pre-LN (norm_first=True) | GELU | 2-Stage MLP feature projection
          ▼
[Standalone Single Transformer] (Cell 189)
          │  Removed all ensemble/hybrid mechanisms; replaced both LSTMs
          │  with a single optimized Transformer (num_layers=1)
          ▼
[Temporal-Aware Harmonic Transformer] (Cell 190 - FINAL BREAKTHROUGH)
             Multi-scale Fourier harmonic encodings on continuous hypervisor time
             Result: 96.50% Weighted F1 / 97.24% Binary F1
```

### 1. The Classical Baseline Plateau (~63.9% Accuracy, ~51.1% F1)
In the original repository, models such as **KNN (k=3, k=5)**, **Decision Trees**, **Bagging**, **Random Forest**, **AdaBoost**, and **Stacking** were trained exclusively on four static features:
$$\mathbf{X} = [\text{GPA}, \text{size}, \text{entropy}, \text{type}]$$
Execution timestamps (`UNIX_time_sec`, `UNIX_time_ns`) were discarded. On a single-operation basis, isolated 4 KiB writes with high entropy look identical between benign tools (e.g., `SDelete`, `AESCrypt`, `Zip`) and ransomware (`Conti`, `LockBit`, `WannaCry`).

This caused severe **asymmetric failure**:
- Ransomware Recall: **99.41%**
- Benign Recall: **only 2.93%** (a 97.07% False Positive Rate on benign data!)
- Weighted F1 Score: **0.511**

### 2. Deep Learning Progression (Cells 181–186)
- **Baseline LSTM**: 2-layer stacked LSTM (`num_layers=2`, `hidden_dim=64`).
- **Pure Transformer**: Replaced both recurrent LSTM layers with a 2-layer Transformer Encoder (`d_model=64`, `nhead=4`).
- **Hybrid Networks**: Explored Layer 1 Transformer $\to$ Layer 2 LSTM, and Layer 1 LSTM $\to$ Layer 2 Transformer.

### 3. Nudged Architectural Optimization (Cell 187)
Tuning hyperparameters across latent capacity, attention subspaces, and gradient flow:
- `d_model`: increased from 64 $\to$ 128
- `nhead`: increased from 4 $\to$ 8
- `dim_feedforward`: increased from 128 $\to$ 256
- `activation`: upgraded from `relu` $\to$ `gelu`
- `norm_first`: enabled `True` (Pre-LayerNorm residual flow)
- `dropout`: tuned to 0.05
- Feature Projection: upgraded to a 2-stage MLP (`Linear` $\to$ `LayerNorm` $\to$ `GELU` $\to$ `Linear`)

### 4. Standalone Single Optimized Transformer (Cell 189)
Removed all ensemble features, stacking components, and hybrid recurrence. Replaced both stacked LSTM layers with **1 single** (`num_layers=1`) optimized Transformer Encoder block on the core features, achieving 0.631 accuracy.

### 5. The Final Breakthrough: Harmonic Temporal Encodings (Cell 190)
Recognizing that ransomware behavior is inherently **bursty and rate-driven**, hypervisor execution time was reintroduced using **Multi-Scale Harmonic Fourier Encodings**:
$$\phi(t) = \left[ \sin(2\pi t \mathbf{B}),\; \cos(2\pi t \mathbf{B}),\; t_{\text{ns}} \right]$$
where continuous normalized execution time $t$ is projected across multi-frequency basis $\mathbf{B} \sim \mathcal{N}(0, 10^2)$. 

Concatenating the 33-dimensional harmonic temporal projection with the 4 standardized memory features yields a 37-dimensional representation. Self-attention now observes both physical memory topology and execution cadence, resolving previous feature overlaps and allowing the model to soar past 90%+ F1.

---

## Comprehensive Performance Comparison

Evaluated on the full 25% test set (**356,476 operations**):

| Model | Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1 Score (Weighted) | Binary F1 (Ransomware) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **KNN (k=3)** | Classical Instance | 0.535 | 0.569 | 0.535 | 0.550 | 0.697 |
| **KNN (k=5)** | Classical Instance | 0.591 | 0.592 | 0.591 | 0.591 | 0.743 |
| **Voting Ensemble** | Majority Hard Vote | 0.592 | 0.589 | 0.592 | 0.590 | 0.744 |
| **Baseline Bagging** | 10 Decision Trees | 0.639 | 0.677 | 0.639 | 0.511 | 0.776 |
| **Single Transformer** | 1-Layer Pre-LN (Static) | 0.631 | 0.399 | 0.631 | 0.489 | 0.776 |
| **Temporal Transformer** | **1-Layer + Harmonic Time** | **0.965** | **0.965** | **0.965** | **0.965** | **0.972** |

#### Final Model Classification Report:
```
                precision    recall  f1-score   support

    Benign (0)     0.9546    0.9503    0.9525    131,388
Ransomware (1)     0.9711    0.9736    0.9724    225,088

      accuracy                         0.9650    356,476
  weighted avg     0.9650    0.9650    0.9650    356,476
```

---

## Repository Layout

```
checkPS/
├── RanSMAP_Ransomware_Binary_Classification.ipynb # Master analytical & training notebook (190 cells)
├── app.py                                         # Flask web application and CLI runner
├── classifier_pipeline.py                         # Telemetry ingestion, preprocessing, & inference pipeline
├── test_app.py                                    # Comprehensive unit test suite (15 tests passing)
├── generate_test_samples.py                       # Generator for multi-file benign and ransomware test sets
│
├── temporal_transformer_model.pkl                 # Final Production Model (96.5% F1 Temporal Transformer)
├── bagging_ransomware_model.pkl                   # Baseline BaggingClassifier model
├── feature_scaler.pkl                             # Baseline StandardScaler
│
├── experimental/                                  # Archived intermediate deep learning checkpoints
│   ├── optimized_transformer_model.pt             # 2-layer optimized Transformer
│   ├── optimized_transformer_lstm_model.pt        # Hybrid A (Transformer -> LSTM)
│   ├── optimized_lstm_transformer_model.pt        # Hybrid B (LSTM -> Transformer)
│   ├── single_optimized_transformer_model.pt      # 1-layer standalone Transformer
│   └── temporal_aware_transformer_model.pt        # Temporal-aware PyTorch state dict
│
├── templates/
│   └── index.html                                 # Interactive web dashboard with model switcher
├── samples/                                       # Generated 6-CSV verification workloads
│   ├── benign/                                    # Firefox, Zip, Office, Idle memory & storage logs
│   └── ransomware/                                # WannaCry, Conti, LockBit memory & storage logs
├── requirements.txt                               # Project dependencies
└── README.md                                      # Project documentation
```

---

## Installation & Setup

### 1. Prerequisites
Ensure Python 3.10+ is installed on your system.

```bash
# Clone the repository
git clone https://github.com/gmh5225/RanSMAP-Ransomware-Binary-Classification.git
cd checkPS

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage Guide

### 1. Web Application Mode
Launch the local web server:
```bash
python3 app.py --port 5000
```
Open your web browser and navigate to **`http://localhost:5000`**.

- **Select Model**: Choose between the **Temporal-Aware Transformer (96.5% F1)** and the **Baseline Bagging Classifier (63.9% Accuracy)**.
- **Upload**: Provide 6 workload CSV files across the memory and storage channels.
- **One-Click Quick Demos**: Click **"Load Benign Sample"** or **"Load Ransomware Sample"** to instantly run pre-packaged RanSMAP workloads.
- **Export Results**: Download full row-by-row prediction tables as a CSV report.

---

### 2. Command-Line Interface (CLI) Mode

Run inference directly on directory workloads or specific files:

```bash
# Run on the pre-generated benign sample using the Temporal Transformer
python3 app.py --sample benign --model temporal_transformer

# Run on the pre-generated ransomware sample using the Baseline model
python3 app.py --sample ransomware --model baseline

# Run on a custom directory containing 6 CSV files
python3 app.py --dir samples/ransomware/ --model temporal_transformer

# Pass 6 individual CSV paths directly
python3 app.py mem_read.csv mem_write.csv mem_exec.csv mem_readwrite.csv ata_read.csv ata_write.csv
```

#### Example CLI Output:
```
======================================================================
 RanSMAP Ransomware Classifier - CLI Mode
======================================================================
Loading 6 input files with model 'temporal_transformer'...

======================================================================
 OVERALL VERDICT:  >>> BENIGN <<<  (LOW RISK)
======================================================================
  Model Used:                  Temporal-Aware Transformer Model (Accuracy: 96.5%, F1: 96.5%)
  Total Files Analyzed:        6 (4 memory, 2 storage)
  Total Memory Operations:     22,336
  Predicted Ransomware Ops:    10 (0.04%)
  Predicted Benign Ops:        22,326 (99.96%)
  Average Model Confidence:    99.5%
  Average Ransomware Prob:     0.005

----------------------------------------------------------------------
Filename               Role            Total Ops  Malicious  Ratio    Verdict
----------------------------------------------------------------------
mem_read.csv           Memory Log      5,584      0          0.0%     BENIGN
mem_write.csv          Memory Log      5,584      0          0.0%     BENIGN
mem_exec.csv           Memory Log      5,584      7          0.1%     BENIGN
mem_readwrite.csv      Memory Log      5,584      3          0.1%     BENIGN
ata_read.csv           ATA Storage Log 100        N/A        N/A      STORAGE_TELEMETRY
ata_write.csv          ATA Storage Log 100        N/A        N/A      STORAGE_TELEMETRY
```

---

### 3. Running Unit Tests
Verify pipeline integrity, data parsers, model switches, and Flask endpoints:
```bash
python3 -m unittest test_app.py
```
```
...............
----------------------------------------------------------------------
Ran 15 tests in 1.380s

OK
```

---

## License

This project is licensed under the [Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0)](https://creativecommons.org/licenses/by-nd/4.0/).
