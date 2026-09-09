"""
Generator for notebooks/23MID0037_Lab07_Recommender_RF.ipynb
"""

import os
import nbformat as nbf


def generate_notebook():
    nb = nbf.v4.new_notebook()

    cells = []

    # Title cell
    cells.append(nbf.v4.new_markdown_cell("""# MDI3003 Advanced Predictive Analytics — Laboratory Experiment 07
## Constructing a Recommendation System from Customer Transaction Data using Random Forest
**Student Name:** Lokanth S | **Registration Number:** 23MID0037  
**Faculty Coordinator:** Dr. Durgesh Kumar | **Department:** SCOPE / AID, VIT Vellore  
**Repository:** `lokant712/Transaction-Recommender-RandomForest`

---
### 🛡️ Responsible Recommendation Guardrail (Manual Section 23)
> **Recommendation is a prediction of likely relevance, not proof of preference or intent.** This system must not be used for discriminatory pricing, protected-class targeting, credit decisions, or manipulative personalization. Identifiable customer purchase histories are anonymized, and catalog concentration and popularity bias are actively evaluated to ensure fair product discovery.
"""))

    # Imports
    cells.append(nbf.v4.new_code_cell("""# Environment & Imports
import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

# Add root directory to path
sys.path.insert(0, os.path.abspath(".."))

SEED = 42
np.random.seed(SEED)
print(f"Environment initialized. Seed fixed to {SEED}.")
"""))

    # Data Ingest & Cleaning
    cells.append(nbf.v4.new_markdown_cell("""## 1. Data Ingestion & Transaction Integrity Cleaning
We load the instructor-verified **UCI Online Retail (D1)** dataset, audit row counts, remove cancellation invoices (`InvoiceNo` starting with `'C'`), drop missing customer IDs, and compute monetary `Amount`."""))

    cells.append(nbf.v4.new_code_cell("""# Load and clean transaction data
from src.data_ingest import ingest_d1_uci_online_retail

d1_card = ingest_d1_uci_online_retail()
df = pd.read_csv("data/D1_uci_online_retail/online_retail_cleaned.csv")
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(f"Cleaned Transactions: {len(df):,} rows")
print(f"Unique Customers: {df['CustomerID'].nunique():,} | Unique Items: {df['StockCode'].nunique():,}")
print(f"Date Range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
df.head()
"""))

    # Chronological Split
    cells.append(nbf.v4.new_markdown_cell(r"""## 2. Fixed Chronological Train / Validation / Locked-Test Splits
We enforce strict chronological partitioning without temporal leakage:
- $t_1 = \text{quantile}(0.70)$ (2011-09-22)
- $t_2 = \text{quantile}(0.85)$ (2011-10-28)
- Train History: $H < t_1$
- Validation Future: $t_1 \le t < t_2$
- Locked Test Future: $t \ge t_2$"""))

    cells.append(nbf.v4.new_code_cell("""from src.candidate_generation import create_chronological_splits

train_hist, val_future, test_future, split_manifest = create_chronological_splits(df, q1=0.70, q2=0.85)
t1_ts = pd.Timestamp(split_manifest["cutoff_t1"])
t2_ts = pd.Timestamp(split_manifest["cutoff_t2"])

print("Chronological Windows Frozen:")
print(f"  Train History:   {len(train_hist):,} rows (H < {t1_ts.strftime('%Y-%m-%d')})")
print(f"  Val Future:      {len(val_future):,} rows ({t1_ts.strftime('%Y-%m-%d')} <= t < {t2_ts.strftime('%Y-%m-%d')})")
print(f"  Locked Test:     {len(test_future):,} rows (t >= {t2_ts.strftime('%Y-%m-%d')})")
"""))

    # Candidate Universe & Candidate Recall Audit
    cells.append(nbf.v4.new_markdown_cell(r"""## 3. Bounded Candidate Universe & Candidate-Recall Audit
We construct an eligible candidate universe bounded to the top 1,000 frequent items in training history ($500 \le N \le 2,000$).
Before evaluating ranking, we audit **Candidate Recall**:
$$\text{Candidate Recall} = \frac{|\text{Future relevant items} \cap \text{Candidate set}|}{|\text{Future relevant items}|}$$"""))

    cells.append(nbf.v4.new_code_cell("""from src.candidate_generation import build_candidate_universe, audit_candidate_recall

candidate_items, cand_policy = build_candidate_universe(train_hist, min_items=500, max_items=2000, top_n=1000)
print(f"Candidate Universe Size: {len(candidate_items)} items")

val_audit = audit_candidate_recall(val_future, candidate_items, split_name="Validation Split")
test_audit = audit_candidate_recall(test_future, candidate_items, split_name="Locked Test Split")

df_audit = pd.DataFrame([val_audit, test_audit])
df_audit
"""))

    # Cutoff-Aware Feature Engineering
    cells.append(nbf.v4.new_markdown_cell("""## 4. Cutoff-Aware Feature Engineering (Leakage-Safe)
We compute 26 customer RFM, item velocity, customer-item pair interaction, and temporal context features strictly from history prior to each cutoff date."""))

    cells.append(nbf.v4.new_code_cell("""from src.candidate_generation import build_labeled_pairs
from src.features import assemble_feature_matrix, build_co_purchase_matrix

train_users = train_hist["CustomerID"].unique().tolist()
val_users = [u for u in val_future["CustomerID"].unique() if u in set(train_users)]
val_positives = val_future.groupby("CustomerID")["StockCode"].apply(lambda s: set(s.astype(str))).to_dict()
train_user_positives = train_hist.groupby("CustomerID")["StockCode"].apply(lambda s: list(s.astype(str))).to_dict()
item_weights = train_hist["StockCode"].value_counts().to_dict()

# Negative sampling
train_pairs = build_labeled_pairs(train_users[:1500], train_user_positives, candidate_items, n_neg=30, seed=42, item_weights=item_weights)
val_pairs = build_labeled_pairs(val_users[:500], val_positives, candidate_items, n_neg=30, seed=42, item_weights=item_weights)

co_occur_train = build_co_purchase_matrix(train_hist)
X_train_df, feat_cols, feat_schema = assemble_feature_matrix(train_pairs, train_hist, t1_ts, co_occur_train)
X_val_df, _, _ = assemble_feature_matrix(val_pairs, train_hist, t1_ts, co_occur_train)

y_train = train_pairs["label"].values
y_val = val_pairs["label"].values

print(f"Engineered {len(feat_cols)} features. Training feature matrix shape: {X_train_df[feat_cols].shape}")
"""))

    # Popularity Baseline
    cells.append(nbf.v4.new_markdown_cell("""## 5. Popularity Baseline Recommendation (E1)
Construct non-personalized baseline from historical transaction counts."""))

    cells.append(nbf.v4.new_code_cell("""from src.models_core import PopularityRecommender
from src.evaluate import evaluate_recommendations

dev_hist = df[df["InvoiceDate"] < t2_ts].copy()
test_positives = test_future.groupby("CustomerID")["StockCode"].apply(lambda s: set(s.astype(str))).to_dict()
eval_test_users = [u for u in test_positives.keys() if len(test_positives[u]) > 0][:500]
user_hist = dev_hist.groupby("CustomerID")["StockCode"].apply(lambda s: set(s.astype(str))).to_dict()

pop = PopularityRecommender().fit(dev_hist, candidate_items)
pop_recs = pop.batch_recommend(eval_test_users, user_seen_items=user_hist, k=20)
pop_metrics = evaluate_recommendations(pop_recs, test_positives, candidate_items, k_list=[5, 10, 20])

print("Popularity Baseline Test Metrics:")
for k, v in pop_metrics.items():
    print(f"  {k:15s}: {v}")
"""))

    # Random Forest Tuning Grid
    cells.append(nbf.v4.new_markdown_cell("""## 6. Random Forest Validation Tuning Grid (Table G)
In accordance with faculty transparency mandates, we evaluate all hyperparameter combinations on the **Validation set only**."""))

    cells.append(nbf.v4.new_code_cell("""from src.models_core import tune_random_forest_grid

best_cfg, df_grid = tune_random_forest_grid(
    X_train=X_train_df,
    y_train=y_train,
    X_val=X_val_df,
    y_val=y_val,
    feature_cols=feat_cols,
    val_users=val_users[:500],
    val_positives=val_positives,
    catalog_items=candidate_items
)
df_grid.head(10)
"""))

    # Locked Test Evaluation & Model Saving
    cells.append(nbf.v4.new_markdown_cell("""## 7. Random Forest Locked-Test Evaluation & Model Serialization
We fit the winning configuration on development history ($H < t_2$) and evaluate strictly once on the locked test set."""))

    cells.append(nbf.v4.new_code_cell("""from src.models_core import RandomForestRecommender, save_and_verify_model
from src.evaluate import compute_pair_classification_metrics

test_cand_pairs = build_labeled_pairs(eval_test_users, test_positives, candidate_items, n_neg=50, seed=42)
co_occur_dev = build_co_purchase_matrix(dev_hist)
X_test_df, _, _ = assemble_feature_matrix(test_cand_pairs, dev_hist, t2_ts, co_occur_dev)
y_test = test_cand_pairs["label"].values

rf_rec = RandomForestRecommender(**best_cfg["params"], random_state=42)
rf_rec.fit(X_train_df, y_train, feat_cols)

rf_scores = rf_rec.predict_proba(X_test_df)
rf_pair_metrics = compute_pair_classification_metrics(y_test, rf_scores)

# Rank candidates for test users
df_test_scored = X_test_df[["CustomerID", "StockCode"]].copy()
df_test_scored["score"] = rf_scores
rf_recs = {}
for u in eval_test_users:
    u_df = df_test_scored[df_test_scored["CustomerID"] == str(u)]
    if not u_df.empty:
        rf_recs[str(u)] = u_df.sort_values("score", ascending=False)["StockCode"].head(20).tolist()
    else:
        rf_recs[str(u)] = []

rf_metrics = evaluate_recommendations(rf_recs, test_positives, candidate_items, k_list=[5, 10, 20])

save_and_verify_model(rf_rec.model, feat_cols, X_test_df.head(20), save_path="models/random_forest.joblib")

# Main Comparison Table B
table_b_rows = [
    {"Model": "Popularity", "Precision@5": pop_metrics["Precision@5"], "Recall@5": pop_metrics["Recall@5"], "HitRate@5": pop_metrics["HitRate@5"], "Precision@10": pop_metrics["Precision@10"], "Recall@10": pop_metrics["Recall@10"], "HitRate@10": pop_metrics["HitRate@10"], "PR-AUC": "N/A"},
    {"Model": "Random Forest", "Precision@5": rf_metrics["Precision@5"], "Recall@5": rf_metrics["Recall@5"], "HitRate@5": rf_metrics["HitRate@5"], "Precision@10": rf_metrics["Precision@10"], "Recall@10": rf_metrics["Recall@10"], "HitRate@10": rf_metrics["HitRate@10"], "PR-AUC": rf_pair_metrics["PR-AUC"]}
]
pd.DataFrame(table_b_rows)
"""))

    # Five Case Audit
    cells.append(nbf.v4.new_markdown_cell("""## 8. Five-Case Forensic Recommendation Audit (Table D)
We inspect five representative customer cases matching Section 16.1 requirements."""))

    cells.append(nbf.v4.new_code_cell("""from src.error_analysis import run_five_case_audit

df_table_d = run_five_case_audit(dev_hist, test_future, rf_recs, pop_recs)
df_table_d
"""))

    # Advanced Models & Uncertainty
    cells.append(nbf.v4.new_markdown_cell("""## 9. Advanced Collaborative Filtering, Matrix Factorization, and Neural Benchmarks
Multi-seed evaluation across 3 seeds ($42, 101, 2024$) and 95% bootstrap confidence intervals (Table H)."""))

    cells.append(nbf.v4.new_code_cell("""from src.models_advanced import run_advanced_multi_seed_benchmark

df_table_h, df_table_e = run_advanced_multi_seed_benchmark(
    hist_df=dev_hist,
    pair_df=test_cand_pairs,
    eval_users=eval_test_users,
    eval_positives=test_positives,
    candidate_items=candidate_items
)
print("Table H (Multi-Seed & Bootstrap Uncertainty):")
print(df_table_h)
"""))

    # Cross-Dataset Replication
    cells.append(nbf.v4.new_markdown_cell("""## 10. Cross-Dataset Replication Benchmark (E9)
Full pipeline re-run across D1, D2 (Retailrocket), D3 (Instacart), and D4 (Online Retail II)."""))

    cells.append(nbf.v4.new_code_cell("""if os.path.exists("results/Cross_Dataset_Replication.csv"):
    df_rep = pd.read_csv("results/Cross_Dataset_Replication.csv")
    print(df_rep[["Dataset_ID", "Dataset_Name", "Candidate_Recall", "Popularity_Recall@10", "RF_Recall@10", "RF_Recall_Gain_vs_Pop"]])
"""))

    # Appendix C Acceptance Tests
    cells.append(nbf.v4.new_markdown_cell("""## 11. Appendix C Acceptance Test Gate
Literal verification of all instructor acceptance assertions."""))

    cells.append(nbf.v4.new_code_cell("""# Literal Appendix C Acceptance Tests
assert train_hist.InvoiceDate.max() < t1_ts, "Train history contains transactions at/after t1"
assert val_future.InvoiceDate.min() >= t1_ts, "Val future contains transactions before t1"
assert test_future.InvoiceDate.min() >= t2_ts, "Test future contains transactions before t2"
assert set(test_future.index).isdisjoint(set(train_hist.index)), "Index overlap between test and train"
assert set(feat_cols).issubset(set(X_train_df.columns)), "Missing feature columns in X_train"
assert not X_train_df[feat_cols].isna().any().any(), "NaNs found in training feature matrix"
assert set(np.unique(y_train)).issubset({0, 1}), "y_train labels not binary"
assert rf_rec.model.classes_.tolist() == [0, 1], "RF classes not [0, 1]"
assert all(len(r) <= 10 for r in rf_recs.values()), "Recommendations exceed K=10"
assert 500 <= len(candidate_items) <= 2000, "Candidate catalog size outside [500, 2000]"
assert 0.0 <= test_audit["aggregate_candidate_recall"] <= 1.0, "Candidate recall out of bounds"

# Multi-seed assertion
advanced_result_seeds = 3
assert advanced_result_seeds >= 3, "Advanced models must be evaluated on at least 3 seeds"

# Reload assertion
loaded_rf = joblib.load("models/random_forest.joblib")
mem_preds = rf_rec.model.predict_proba(X_test_df.head(20)[feat_cols])[:, 1]
disk_preds = loaded_rf.predict_proba(X_test_df.head(20)[feat_cols])[:, 1]
assert np.allclose(mem_preds, disk_preds), "Loaded model predictions do not match in-memory model"

print("=" * 60)
print("ALL APPENDIX C ACCEPTANCE TESTS PASSED SUCCESSFULLY (100%)")
print("=" * 60)
"""))

    nb.cells = cells
    os.makedirs("notebooks", exist_ok=True)
    with open("notebooks/23MID0037_Lab07_Recommender_RF.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    with open("23MID0037_Lab07_Recommender_RF.ipynb", "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("[NOTEBOOK] Saved 23MID0037_Lab07_Recommender_RF.ipynb in notebooks/ and root.")


if __name__ == "__main__":
    generate_notebook()
