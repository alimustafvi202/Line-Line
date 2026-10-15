# AI-Based Heart Anomaly Detection Using ECG Signals

An academic research framework for automated arrhythmia classification and heart anomaly detection using ambulatory electrocardiogram (ECG) signals from the official PhysioNet **MIT-BIH Arrhythmia Database**.

---

## 1. Project Directory Structure

```text
ECG-Heart-Anomaly-Research/
├── package.json                 # Next.js 15 project configuration & scripts
├── next.config.mjs              # Standalone Vercel deployment configuration
├── jsconfig.json                # Path alias mapping (@/*)
├── requirements.txt             # Python machine learning requirements (wfdb, torch, etc.)
├── README.md                    # Academic documentation and reproduction guide
├── .gitignore                   # Excludes raw data, node_modules, .next, and venv from Git
│
├── app/                         # Next.js App Router (100% Vercel Ready)
│   ├── layout.js                # Global layout, Google Fonts & SEO tags
│   ├── page.js                  # Interactive Oscilloscope & AI Dashboard
│   ├── globals.css              # Glassmorphic responsive dark theme
│   └── api/                     # Vercel Serverless Route Handlers
│       ├── status/route.js      # /api/status endpoint
│       ├── records/route.js     # /api/records endpoint
│       ├── models/route.js      # /api/models endpoint
│       ├── benchmarks/route.js  # /api/benchmarks endpoint
│       ├── record-signal/route.js # /api/record-signal endpoint
│       └── predict/route.js     # /api/predict endpoint
│
├── public/                      # Static assets & icons
│
├── data/
│   ├── ecgData.json             # Bundled MIT-BIH signals & AI inference data for Vercel
│   ├── raw/                     # Downloaded official PhysioNet records (.dat, .hea, .atr)
│   └── processed/               # Extracted heartbeat segments and splits
│
├── src/                         # Python Signal Processing & AI Training Pipeline
│   ├── __init__.py              # Package marker
│   ├── data_loader.py           # PhysioNet WFDB loading & record verification
│   ├── preprocessing.py         # Zero-phase bandpass filter, Z-score & segmentation
│   ├── model.py                 # PyTorch 1D-CNN Architecture
│   ├── train.py                 # Inter-patient training & validation loop
│   ├── predict.py               # Local CLI inference & visualization script
│   ├── visualization.py         # Morphological inspection and distribution plots
│   └── evaluation.py            # Academic evaluation metrics & confusion matrices
│
├── experiments/
│   └── baseline_config.json     # Hardware, software versions & hyperparameters
│
└── outputs/
    ├── figures/                 # Generated publication-quality diagnostic plots
    ├── tables/                  # Benchmark CSV and JSON metric reports
    └── models/                  # Saved PyTorch checkpoint weights (.pt)
```

---

## 2. Dataset Documentation & Standard Label Mapping

### Dataset Source
- **Database:** MIT-BIH Arrhythmia Database (`mitdb` v1.0.0)
- **Source:** [PhysioNet MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- **Specifications:** 48 two-channel 30-minute recordings from 47 ambulatory patients, sampled at **360 Hz** with 11-bit resolution over a ±10 mV range.
- **Channels:** Modified Lead II (MLII) is available in almost all recordings, accompanied by modified lead V1, V2, V4, or V5.

### AAMI EC57 5-Class Heartbeat Mapping
In accordance with the Association for the Advancement of Medical Instrumentation (AAMI EC57) standard, individual heartbeat annotations are grouped into five super-classes:

| AAMI Super-Class | Category Name | MIT-BIH PhysioNet Annotation Symbols | Clinical Description |
|:---:|:---|:---|:---|
| **N** | Non-Ectopic | `N`, `L`, `R`, `e`, `j` | Normal beats, Left/Right bundle branch block, Atrial/Nodal escape |
| **S** | Supraventricular Ectopic (SVEB) | `A`, `a`, `J`, `S` | Atrial premature, Aberrated atrial premature, Nodal premature |
| **V** | Ventricular Ectopic (VEB) | `V`, `E` | Premature ventricular contractions (PVC), Ventricular escape |
| **F** | Fusion | `F` | Fusion of ventricular and normal beats |
| **Q** | Unknown / Paced | `/`, `f`, `Q`, `?` | Paced beats, Fusion of paced beat, Unclassifiable |

> **Non-Beat Annotations:** Non-beat markers (e.g. rhythm changes `+`, signal quality changes `~`, isolated noise `|`, measurement comments) are explicitly filtered out before heartbeat segmentation to prevent false samples.

---

## 3. Data Leakage Prevention: Inter-Patient Separation Protocol

A critical flaw in naive ECG classification is **intra-patient random splitting** (randomly shuffling heartbeats across train and test sets). Because heartbeats from the same patient share identical morphological signatures, random splitting produces artificially inflated test accuracy (>98%) that fails catastrophically when deployed on unseen patients.

To prevent patient data leakage, this project implements the standard **De Chazal et al. (2004)** inter-patient division:

- **DS1 (Training Set, 22 recordings):**  
  `101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230`
- **DS2 (Testing Set, 22 recordings):**  
  `100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234`
- **Excluded Paced Records (4 recordings):**  
  `102, 104, 107, 217` (recordings with predominantly paced beats per AAMI standard recommendation).

---

## 4. Signal Processing Pipeline

1. **Baseline Wander & High-Frequency Noise Filtering:**  
   Zero-phase 3rd-order Butterworth bandpass filter (`0.5 Hz` to `40.0 Hz`) using `scipy.signal.filtfilt`. This eliminates low-frequency respiration baseline wander and 50/60 Hz powerline interference without introducing phase distortion.
2. **Lead Normalization:**  
   Z-score standardization $(x - \mu) / \sigma$ per recording preserves relative intra-beat morphology while normalizing amplitude scales across subjects.
3. **Heartbeat Segmentation:**  
   Windows of 200 samples (`[-90, +110]` relative to the annotated R-peak, approximately 555 ms at 360 Hz) capturing the full P-QRS-T complex.

---

## 5. Execution Commands on Windows (PowerShell)

### Step 0: Activate Virtual Environment
```powershell
.venv\Scripts\activate
```

### Step 1: Download & Verify Sample Data
Download sample records (`100`, `101`) to test the pipeline without waiting for the full dataset:
```powershell
python -m src.data_loader --download-sample
```
*(To download all 48 records later: `python -m src.data_loader --download-all`)*

Inspect a record's properties:
```powershell
python -m src.data_loader --inspect 100
```

### Step 2: Run Signal Preprocessing & Segmentation
Preprocess Record 100 and display heartbeat counts and AAMI label distribution:
```powershell
python -m src.preprocessing --record 100
```

### Step 3: Generate Figures (Waveforms, Morphologies, Distributions)
Generate high-resolution plots saved directly into `outputs/figures/`:
```powershell
python -m src.visualization --record 100
```

### Step 4: Run Exploratory Data Analysis (EDA) Pipeline
Execute the full automated EDA pipeline script:
```powershell
python notebooks/run_eda.py
```

### Step 5: Test the Academic Evaluation Pipeline
Run the evaluation test suite to verify metric computation (Accuracy, Precision, Recall/Sensitivity, Specificity, Macro F1) and confusion matrix visualization:
```powershell
python -m src.evaluation --demo
```

---

## 6. Next.js Web Dashboard & 1-Click Vercel Deployment

This project includes a **100% self-contained Next.js 15 App Router web application** directly at the repository root, ready for immediate, zero-configuration deployment to [Vercel](https://vercel.com).

### Running Locally
```powershell
# Run the Next.js development server
npm run dev
```
Open **[http://localhost:3000](http://localhost:3000)** in your browser.

### Deploying to Vercel (1-Click)
1. Push this repository to **GitHub**.
2. Go to [Vercel Dashboard](https://vercel.com/dashboard) and click **"Add New Project"**.
3. Import this repository.
4. Vercel automatically detects Next.js at the root (`package.json`), sets the build command to `next build`, and deploys with **zero extra configuration**.
5. All endpoints (`/api/records`, `/api/models`, `/api/predict`, etc.) run as **Vercel Serverless Functions** without needing any external Python server.

---

## 6. Current Status & Proposed Next Steps

### Completed Pipeline Components
- Modular project architecture adhering to academic reproducibility guidelines.
- Official PhysioNet MIT-BIH dataset loader via `wfdb`.
- Biomedical preprocessing (zero-phase bandpass filtering and Z-score normalization).
- AAMI EC57 label mapping and R-peak heartbeat segmentation.
- Visualization module producing waveform comparisons, class distributions, and mean morphologies.
- Evaluation module calculating sensitivity, specificity, macro F1, and normalized confusion matrices.
- Baseline configuration file (`experiments/baseline_config.json`).

### Proposed Next Steps (Awaiting Approval)
1. **Full Dataset Ingestion:** Download all 44 non-paced recordings and construct partitioned tensors for DS1 (train) and DS2 (test).
2. **Transparent Baseline Model:** Train an interpretable baseline classifier (e.g., Random Forest or 1D-CNN) strictly on DS1 and evaluate on DS2.
3. **Class Imbalance Mitigation:** Address severe clinical class imbalance (high `N`, rare `S` and `V`) using focal loss or class-weighted cross-entropy.
