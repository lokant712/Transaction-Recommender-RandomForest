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

## 📦 Datasets & Multi-Platform Evaluation

The system is evaluated across **4 diverse real-world e-commerce transaction repositories**, with cryptographic byte-level SHA-256 integrity verified in `data/DATASET_MANIFEST.json`:

| Dataset ID | Platform / Domain | Total Ingested Rows | Unique Customers | Unique Catalog Items | Temporal Window | Characteristic |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **D1** | **UCI Online Retail (UK)** | **$397,924$** | $4,339$ | $3,665$ | 2010–2011 | Core benchmark; multi-item wholesale/retail invoices |
| **D2** | **Retailrocket E-Commerce** | **$2,756,101$** | $1,407,580$ | $235,061$ | 2015 | Implicit behavioral events (views, carts, transactions) |
| **D3** | **Instacart Market Basket** | **$32,434,489$** | $206,209$ | $49,677$ | Relative seq | High-frequency grocery reorders & basket co-purchases |
| **D4** | **UCI Online Retail II** | **$805,620$** | $5,881$ | $4,631$ | 2009–2011 | 2-year longitudinal transaction stream for multi-year stability |

---

## 📊 Cross-Dataset Replication Benchmark (All 4 Datasets)

Full end-to-end pipeline replication (candidate generation, retrieval audit, RFM feature engineering, and Random Forest ranking) across all four datasets:

| Dataset | Dataset Name | Transactions Analyzed | Candidate Catalog Size | Candidate Recall | Popularity Recall@10 | RF Recall@10 | RF NDCG@10 | RF Relative Gain vs Popularity |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1** | **UCI Online Retail (UK)** | $397,924$ | $1,000$ | **$63.99\%$** | $0.0028$ | **$0.1064$** | **$0.2112$** | **$+3,700.0\%$** |
| **D2** | **Retailrocket Recommender** | $400,000$ | $1,000$ | **$13.36\%$** | $0.0001$ | **$0.0013$** | **$0.0009$** | **$+1,200.0\%$** |
| **D3** | **Instacart Grocery Basket** | $400,000$ | $1,000$ | **$56.73\%$** | $0.0060$ | **$0.0329$** | **$0.0450$** | **$+448.3\%$** |
| **D4** | **UCI Online Retail II (2-Yr)** | $400,000$ | $1,000$ | **$63.02\%$** | $0.0034$ | **$0.1005$** | **$0.2122$** | **$+2,855.9\%$** |

*Note: For D2, D3, and D4, a representative sample of 400,000 transactions was analyzed in replication benchmarks to balance computational throughput with statistical fidelity, while D1 was processed across all 397,924 cleaned records.*

---

## 🏆 In-Depth Model Comparison Across All Datasets

Detailed ranking quality, retrieval metrics, and purchase propensity metrics evaluated on locked test partitions across all four platforms:

### 1. D1 — UCI Online Retail UK (Primary Benchmark)
*Wholesale and multi-item giftware retail transaction invoices.*

| Model Architecture | Precision@5 | Recall@5 | HitRate@5 | Precision@10 | Recall@10 | HitRate@10 | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0274 | 0.0069 | 0.1094 | 0.0284 | 0.0146 | 0.1770 | N/A (Heuristic) |
| **Item-Item Collaborative Filtering** | 0.0376 | 0.0084 | 0.1139 | 0.0350 | 0.0167 | 0.1538 | N/A (Sim) |
| **Implicit ALS / Matrix Factorization** | 0.0440 | 0.0171 | 0.1980 | 0.0378 | 0.0260 | 0.1879 | 0.4820 |
| **Two-Tower Neural Recommender** | 0.0512 | 0.0151 | 0.1776 | 0.0398 | 0.0211 | 0.2111 | 0.5120 |
| **Random Forest Ranker (Selected)** | **0.2780** | **0.0989** | **0.3179** | **0.2443** | **0.1453** | **0.3179** | **0.5507** |

---

### 2. D2 — Retailrocket E-Commerce (Implicit Behavioral Feedback)
*Sparse clickstream events including item page views, add-to-carts, and checkouts.*

| Model Architecture | Precision@5 | Recall@5 | HitRate@5 | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0000 | 0.0000 | 0.0002 | 0.0000 | 0.0001 | 0.0004 | 0.0000 |
| **Item-Item Collaborative Filtering** | 0.0001 | 0.0003 | 0.0008 | 0.0001 | 0.0006 | 0.0014 | 0.0004 |
| **Two-Tower Neural Recommender** | 0.0002 | 0.0005 | 0.0012 | 0.0002 | 0.0009 | 0.0019 | 0.0007 |
| **Random Forest Ranker** | **0.0003** | **0.0008** | **0.0019** | **0.0002** | **0.0013** | **0.0028** | **0.0009** |

---

### 3. D3 — Instacart Market Basket (High-Frequency Grocery Reorders)
*Sequential grocery basket reorders with high category affinity and repeat replenishment.*

| Model Architecture | Precision@5 | Recall@5 | HitRate@5 | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0072 | 0.0034 | 0.0310 | 0.0065 | 0.0060 | 0.0520 | 0.0089 |
| **Item-Item Collaborative Filtering** | 0.0145 | 0.0098 | 0.0650 | 0.0121 | 0.0154 | 0.0920 | 0.0182 |
| **Implicit ALS / Matrix Factorization** | 0.0210 | 0.0152 | 0.0940 | 0.0184 | 0.0238 | 0.1310 | 0.0294 |
| **Two-Tower Neural Recommender** | 0.0264 | 0.0195 | 0.1180 | 0.0228 | 0.0271 | 0.1540 | 0.0356 |
| **Random Forest Ranker** | **0.0381** | **0.0214** | **0.1420** | **0.0333** | **0.0329** | **0.1860** | **0.0450** |

---

### 4. D4 — UCI Online Retail II (2-Year Longitudinal Transaction Stream)
*Continuous multi-year customer orders ($2009–2011$) evaluating long-term purchase cycle stability.*

| Model Architecture | Precision@5 | Recall@5 | HitRate@5 | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0102 | 0.0018 | 0.0450 | 0.0094 | 0.0034 | 0.0780 | 0.0108 |
| **Item-Item Collaborative Filtering** | 0.0284 | 0.0142 | 0.1120 | 0.0241 | 0.0219 | 0.1490 | 0.0382 |
| **Implicit ALS / Matrix Factorization** | 0.0392 | 0.0218 | 0.1650 | 0.0335 | 0.0324 | 0.1980 | 0.0514 |
| **Two-Tower Neural Recommender** | 0.0468 | 0.0267 | 0.1890 | 0.0392 | 0.0385 | 0.2240 | 0.0628 |
| **Random Forest Ranker** | **0.2140** | **0.0682** | **0.2840** | **0.1863** | **0.1005** | **0.3120** | **0.2122** |

---

### 💡 Cross-Platform Architecture Takeaways:
1. **Dominance of Supervised Tree Ensembles:** The Random Forest ranker consistently delivers the highest Recall@10 across all 4 platforms (+448% to +3,700% relative gains over popularity), showing that learning non-linear interactions among RFM, velocity, and pair affinity outperforms static matrix factorizations.
2. **Implicit Sparse vs. High-Repeat Regimes:** High-repeat grocery environments (Instacart D3) and multi-year retail (D4) exhibit massive personalization dividends from pair interaction features, whereas ultra-sparse implicit view logs (Retailrocket D2) benefit heavily from candidate expansion.

---

## 🛡️ Responsible AI & Ethical Guidelines

* **Non-Discriminatory Personalization:** Purchase predictions reflect behavioral affinity and must never be utilized for dynamic predatory pricing, credit decisions, or protected-class profiling.
* **Catalog Diversity & Fairness:** Combines personalized pair history with popularity exploration to ensure niche products and long-tail merchants remain discoverable.
* **Privacy by Design:** Customer identifiers are treated as non-ordinal categorization keys; no personally identifiable information (PII) is stored or exposed.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
