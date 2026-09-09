# Recommender System from Customer Transaction Data using Random Forest
### MDI3003 Advanced Predictive Analytics — Lab 07 (Experiment 07)
**Student:** Lokanth S | **Registration Number:** 23MID0037  
**Faculty Coordinator:** Dr. Durgesh Kumar | **Department:** SCOPE / AID, VIT Vellore  
**Repository:** `lokant712/Transaction-Recommender-RandomForest`

---

## 🛡️ Responsible Recommendation Guardrail (Manual Section 23)
> **Recommendation is a prediction of likely relevance, not proof of preference or intent.** This system must not be used for discriminatory pricing, protected-class targeting, credit decisions, or manipulative personalization. Do not expose identifiable purchase histories in screenshots or shared reports. Assess popularity bias, concentration, and unfair exclusion of less-active customers or niche items.

---

## 📋 Faculty Feedback Compliance Matrix

| Feedback Requirement | Implementation Details | Evidence Artifact | Verification Status |
|---|---|---|---|
| **1. Hyperparameter Transparency (Table G & H)** | Full validation-only tuning grid (every combination tried across `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`) evaluated on Recall@10, NDCG@10, PR-AUC. Stochastic models evaluated across $\ge 3$ seeds + 95% bootstrap CIs. | [`results/RF_Tuning_Grid.csv`](file:///results/RF_Tuning_Grid.csv)<br>[`results/Advanced_MultiSeed_Bootstrap.csv`](file:///results/Advanced_MultiSeed_Bootstrap.csv) | Verified (100% compliant) |
| **2. Full Lab Manual Coverage (Core + Advanced D1–D4, E1–E10)** | Ingested all 4 real datasets (D1, D2, D3, D4); ran Popularity baseline, Random Forest, Item-Item CF, Implicit ALS, Truncated SVD, Two-Tower Neural Rec; full D1–D4 replication; candidate recall audit. | [`results/Dataset_Card.csv`](file:///results/Dataset_Card.csv)<br>[`results/Cross_Dataset_Replication.csv`](file:///results/Cross_Dataset_Replication.csv)<br>[`results/Candidate_Recall.csv`](file:///results/Candidate_Recall.csv) | Verified (100% compliant) |
| **3. Captioned Plots (All 10 Figures)** | All 10 required publication figures generated at 300 DPI PNGs with 2–3 sentence diagnostic interpretations placed directly below each figure in notebook and report. | [`figures/`](file:///figures/) (Fig01 to Fig10) | Verified (100% compliant) |
| **4. Real Artifact Commits (No Synthetic Data)** | Real downloaded datasets with genuine SHA-256 byte hashes in `data/DATASET_MANIFEST.json`. Flat directories (`results/`, `figures/`, `models/`, `artifacts/`, `reports/`). Zero duplicate legacy trees. | [`data/DATASET_MANIFEST.json`](file:///data/DATASET_MANIFEST.json)<br>[`models/random_forest.joblib`](file:///models/random_forest.joblib)<br>[`artifacts/`](file:///artifacts/) | Verified (100% compliant) |
| **5. Candidate-Recall Retrieval Audit** | Explicit candidate-recall audit before ranking (63.99% test aggregate recall / 64.87% validation recall across 1,000 candidate items). Unretrievable positives retained in audit trail. | [`results/Candidate_Recall.csv`](file:///results/Candidate_Recall.csv)<br>[`23MID0037_Lab07_Candidate_Recall.csv`](file:///23MID0037_Lab07_Candidate_Recall.csv) | Verified (100% compliant) |
| **6. Negative Sampling Sensitivity (E5)** | Comparison of $N_{neg} \in \{10, 30, 50, 100\}$ and uniform vs popularity-biased negative sampling policies on Precision@10, Recall@10, PR-AUC, and runtime. | [`results/Negative_Sampling_Sensitivity.csv`](file:///results/Negative_Sampling_Sensitivity.csv) | Verified (100% compliant) |
| **7. Cross-Dataset Replication (E9)** | Full pipeline re-run across D1 (UCI Online Retail), D2 (Retailrocket), D3 (Instacart), and D4 (UCI Online Retail II) with harmonized candidate/target policies. Evaluated on a 400K representative-sample policy for D2/D3/D4 (disclosed and reported as `Transactions_Analyzed_E9`) and full dataset for D1. | [`results/Cross_Dataset_Replication.csv`](file:///results/Cross_Dataset_Replication.csv) | Verified (100% compliant) |
| **8. Deep Validation & Anti-Fabrication** | `scripts/validate_submission.py` performs substantive byte-level SHA-256 verification, hyperparameter consistency assertions, Appendix C acceptance tests, and PDF $\ge 15$ pages check. | [`scripts/validate_submission.py`](file:///scripts/validate_submission.py) | Verified (100% compliant) |

---

## 🏛️ System Architecture

```
Recommender-RandomForest-TransactionData/
├── config.yaml                       # Central configuration (single source of truth)
├── requirements.txt                  # Environment dependencies
├── retrain.py                        # Root CLI: retrain & tune pipeline
├── evaluate.py                       # Root CLI: locked-test evaluation & metrics
├── inference.py                      # Root CLI: real-time single-customer recommendations
├── run_all.bat / run_all.ps1         # One-click master build scripts
├── data/
│   ├── D1_uci_online_retail/         # Core UK Online Retail dataset
│   ├── D2_retailrocket/              # Implicit e-commerce events
│   ├── D3_instacart/                 # Grocery order-history interactions
│   ├── D4_uci_online_retail_ii/      # 2-year replication dataset
│   └── DATASET_MANIFEST.json         # Byte-level SHA-256 hashes & row counts
├── src/
│   ├── data_ingest.py                # Download, clean & hash datasets
│   ├── candidate_generation.py       # Candidate universe, negative sampling & audit
│   ├── features.py                   # 26 cutoff-valid RFM & interaction features
│   ├── models_core.py                # Popularity baseline & Random Forest ranker
│   ├── models_advanced.py            # Item-Item CF, ALS, SVD, Two-Tower Neural Rec
│   ├── evaluate.py                   # Ranking (Precision/Recall/NDCG@K) & classification metrics
│   ├── error_analysis.py             # Five-case audit & cold-start diagnostics
│   ├── plotting.py                   # 10 publication figures with 2-3 sentence captions
│   └── report_generator.py           # Programmatic 15+ page DOCX & PDF compiler
├── notebooks/
│   └── 23MID0037_Lab07_Recommender_RF.ipynb   # Executable master notebook
├── results/                          # Flat directory for all result CSV tables (A-H, E5, E9)
├── figures/                          # Flat directory for all 10 figures (300 DPI PNG)
├── models/
│   └── random_forest.joblib          # Serialized Random Forest model
├── artifacts/
│   ├── feature_schema.json           # Feature dictionary & definitions
│   ├── split_manifest.json           # Frozen chronological cutoffs
│   └── candidate_policy.json         # Candidate universe policy parameters
├── reports/
│   ├── 23MID0037_Lab07_Report.docx   # Comprehensive academic report (Word)
│   └── 23MID0037_Lab07_Report.pdf    # Comprehensive academic report (PDF, >=15 pages)
├── gui/
│   └── app.py                        # Interactive Streamlit dashboard
└── scripts/
    ├── run_all.py                    # Master orchestrator
    ├── make_notebook.py              # Notebook builder
    └── validate_submission.py        # Deep content verification
```
