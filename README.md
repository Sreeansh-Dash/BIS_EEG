# DMK-I — Sub-5 % MAE Depth-of-Anaesthesia Prediction

> **Differential Mamba-KAN Integrator** for predicting the continuous
> Bispectral Index (BIS) from raw EEG signals using the open
> [VitalDB](https://vitaldb.net/) dataset.

---

## Architecture

```
EEG (1 ch, 128 Hz, 5 s)
        │
  ┌─────▼─────┐
  │    TFF     │  Temporal Feature Fusion
  │ diff → dw  │  • differential feature (X_t − X_{t-1})
  │ sep-conv   │  • depthwise-separable 1-D conv
  │ + gate     │  • sigmoid adaptive weighting
  └─────┬──────┘
        │  (B, T, D)
  ┌─────▼──────────┐
  │  4 × BiMamba   │  Bidirectional Selective-SSM blocks
  │  (mamba-ssm)   │  forward + reversed + residual + LayerNorm
  └─────┬──────────┘
        │  avg-pool → (B, D)
  ┌─────▼─────┐
  │  KAN Head │  Kolmogorov-Arnold Network (efficient-kan)
  │  → σ      │  outputs BIS ∈ [0, 1]
  └───────────┘
```

| Component | Details |
|-----------|---------|
| **TFF** | 1-D depthwise-separable conv + sigmoid gating |
| **Sequence core** | 4 stacked BiMamba blocks (`mamba-ssm`) |
| **Head** | KAN layer → Sigmoid → scalar BIS |
| **Loss** | L1 (MAE) — robust to BIS outlier noise |
| **Precision** | Mixed-precision (AMP) with `GradScaler` |

---

## Repository Layout

```
.
├── data_downloader.py   # Downloads EEG + BIS from VitalDB (128 Hz)
├── dataset.py           # BISDataset, Harmonic Mix augmentation, DataLoaders
├── model.py             # DMK-I model definition
├── train.py             # Training loop with validation & checkpointing
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/dmk-i.git
cd dmk-i
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\activate
```

### 3. Install Dependencies

> **Prerequisites**: NVIDIA GPU with CUDA ≥ 11.8 and `nvcc`.
> `mamba-ssm` compiles custom CUDA kernels during installation.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download the VitalDB Data

```bash
python data_downloader.py
```

This downloads up to 6 400 surgical cases from VitalDB and saves each as
a `.npy` file inside `data/`.  Already-downloaded cases are skipped
automatically, so the script is safe to re-run.

### 5. Train

```bash
python train.py
```

Training runs for **50 epochs** with:

| Hyperparameter | Value |
|---------------|-------|
| Batch size | 128 |
| Learning rate | 1 × 10⁻³ (Adam) |
| Loss | L1 (MAE) |
| Mixed precision | Enabled (CUDA) / disabled (CPU) |

The best model (lowest validation MAE on the 0-100 BIS scale) is saved to
`best_dmk_i_model.pth`.

---

## Dataset Details

| Track | Source | Role |
|-------|--------|------|
| `BIS/EEG1_WAV` | VitalDB | Input (frontal EEG) |
| `BIS/BIS` | VitalDB | Target (Bispectral Index) |

- **Sampling rate**: 128 Hz (via `vitaldb.VitalFile.to_numpy`)
- **Window**: 5 seconds = 640 time-steps
- **Target normalisation**: BIS / 100 → [0, 1]
- **Split**: chronological at subject level — 70 % train / 10 % val / 20 % test
- **Augmentation**: *Harmonic Mix* — frequency-domain interpolation with a
  45 Hz harmonic low-pass filter

---

## License

This project is provided for **research and educational purposes only**.
The VitalDB dataset is governed by its own
[licence](https://vitaldb.net/dataset/).
