# Transaction-Recommender-RandomForest

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Two--Tower-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industrial-grade, leakage-safe recommendation engine that converts historical customer transaction logs (invoices, orders, timestamps, quantities, and basket values) into personalized purchase-propensity scores and Top-K product rankings using **Random Forest** and hybrid collaborative filtering models.

---

## 🌟 Key Highlights

* **Two-Stage Recommendation Architecture:** High-throughput candidate retrieval bounded to top frequent items combined with high-precision Random Forest candidate re-ranking.
* **Leakage-Safe Feature Engineering:** 26+ cutoff-valid features spanning Customer RFM, Item Velocity, Customer-Item Pair Affinity, and Temporal Context.
* **Strict Chronological Evaluation:** Hard temporal train/validation/locked-test cutoffs preventing future transaction leakage.
* **Retrieval-Stage Coverage Audit:** Explicit candidate-recall auditing before ranking ($63.99\%$ test candidate recall) to establish theoretical upper bounds.
* **Comprehensive Model Benchmarks:** Head-to-head comparison across Popularity Baseline, Supervised Random Forest, Item-Item Collaborative Filtering, Implicit Matrix Factorization (ALS), Truncated SVD, and Deep Neural Two-Tower models.
* **Multi-Dataset Generalizability:** Replicated and evaluated across 4 major e-commerce transaction datasets (UCI Online Retail UK, Retailrocket, Instacart Grocery, and UCI Online Retail II).
* **Interactive Streamlit GUI:** Real-time customer profiling, catalog exploration, personalized Top-K recommendation generation, and forensic diagnostic auditing.

---

## 🏛️ System Architecture

```
Transaction-Recommender-RandomForest/
├── config.yaml                       # Central system configuration
├── requirements.txt                  # Python dependencies
├── retrain.py                        # CLI: retrain & tune Random Forest model
├── evaluate.py                       # CLI: locked-test ranking evaluation
├── inference.py                      # CLI: real-time single-customer Top-K recommendations
├── run_all.bat / run_all.ps1         # One-click master build scripts
├── data/                             # Dataset storage and cryptographic manifests
│   └── DATASET_MANIFEST.json         # SHA-256 byte hashes and dataset metadata
├── src/                              # Core modular pipeline source code
│   ├── data_ingest.py                # Ingestion, validation, and cancellation cleaning
│   ├── candidate_generation.py       # Candidate universe, negative sampling & candidate recall audit
│   ├── features.py                   # Cutoff-valid RFM, velocity, and pair interaction features
│   ├── models_core.py                # Popularity baseline & Random Forest ranker
│   ├── models_advanced.py            # Item-Item CF, Implicit ALS, SVD, Two-Tower Neural Rec
│   ├── evaluate.py                   # Precision@K, Recall@K, NDCG@K, HitRate@K, and PR-AUC
│   ├── error_analysis.py             # Five-case audit & cold-start diagnostics
│   └── plotting.py                   # Publication diagnostic figures
├── notebooks/
│   └── 23MID0037_Lab07_Recommender_RF.ipynb   # Executable end-to-end master notebook
├── results/                          # Benchmark tables and experimental evaluation CSVs
├── figures/                          # 10 diagnostic figures (300 DPI PNG)
├── models/
│   └── random_forest.joblib          # Serialized production Random Forest model
├── artifacts/
│   ├── feature_schema.json           # Registered feature schema and dictionary
│   ├── split_manifest.json           # Frozen chronological cutoffs
│   └── candidate_policy.json         # Candidate universe policy parameters
├── gui/
│   └── app.py                        # Interactive Streamlit Web Application
└── scripts/
    ├── run_all.py                    # Master orchestrator script
    ├── make_notebook.py              # Automated notebook builder
    └── validate_submission.py        # System verification and validation suite
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/lokant712/Transaction-Recommender-RandomForest.git
cd Transaction-Recommender-RandomForest
pip install -r requirements.txt
```

### 2. Run the Full End-to-End Pipeline
Execute the full training, hyperparameter tuning, multi-model benchmarking, and figure generation workflow in a single command:
```bash
python scripts/run_all.py
```
*Or using the platform launcher:*
```bash
run_all.bat        # Windows CMD
./run_all.ps1      # Windows PowerShell / Linux / macOS
```

---

## 💻 CLI Usage

### Model Retraining & Tuning
Tune hyperparameters on the validation split and train the production Random Forest model:
```bash
# Retrain with tuned parameters
python retrain.py --tune

# Train with custom hyperparameters
python retrain.py --n-estimators 300 --max-depth 20 --min-samples-leaf 2
```

### Locked-Test Evaluation
Evaluate all models on the locked chronological test set:
```bash
python evaluate.py --k 10 --dataset d1
```

### Real-Time Customer Inference
Generate personalized Top-K product recommendations for any customer ID:
```bash
python inference.py --customer-id 17841 --top-k 10
```

---

## 🖥️ Interactive Streamlit Dashboard

Launch the built-in web dashboard for interactive exploration and model serving:
```bash
streamlit run gui/app.py
```

### Dashboard Features:
1. **Customer Profiling:** Select any customer ID to view historical transaction frequency, unique items purchased, total monetary spend, and recency.
2. **Personalized Top-K Recommendations:** Real-time generation of top recommendations with predicted purchase propensity scores.
3. **Model Comparison:** Side-by-side comparison of Random Forest predictions against Popularity, Collaborative Filtering, and Neural models.
4. **Diagnostic Metrics:** Interactive exploration of Precision@K, Recall@K, NDCG@K curves, feature importance rankings, and catalog coverage Lorenz curves.

---

## 📊 Benchmark Results (Primary D1 Dataset)

| Model Architecture | Precision@5 | Recall@5 | HitRate@5 | Precision@10 | Recall@10 | HitRate@10 | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0274 | 0.0069 | 0.1094 | 0.0284 | 0.0146 | 0.1770 | N/A (Heuristic) |
| **Item-Item CF** | 0.0376 | 0.0084 | 0.1139 | 0.0350 | 0.0167 | 0.1538 | N/A (Sim) |
| **Implicit ALS / MF** | 0.0440 | 0.0171 | 0.1980 | 0.0378 | 0.0260 | 0.1879 | 0.4820 |
| **Two-Tower Neural Rec** | 0.0512 | 0.0151 | 0.1776 | 0.0398 | 0.0211 | 0.2111 | 0.5120 |
| **Random Forest Ranker** | **0.2780** | **0.0989** | **0.3179** | **0.2443** | **0.1453** | **0.3179** | **0.5507** |

> **Key Finding:** Supervised Random Forest achieves a **+895.2% relative gain in Recall@10** and **+760.2% in Precision@10** over the Popularity baseline while distributing recommendations across $68.4\%$ of the candidate catalog.

---

## 🛡️ Responsible AI & Ethical Guidelines

* **Non-Discriminatory Personalization:** Purchase predictions reflect behavioral affinity and must never be utilized for dynamic predatory pricing, credit decisions, or protected-class profiling.
* **Catalog Diversity & Fairness:** Combines personalized pair history with popularity exploration to ensure niche products and long-tail merchants remain discoverable.
* **Privacy by Design:** Customer identifiers are treated as non-ordinal categorization keys; no personally identifiable information (PII) is stored or exposed.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
