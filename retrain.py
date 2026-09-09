"""
Root CLI: Retrain Recommender System Pipeline.
Usage: python retrain.py [--tune] [--seed 42]
"""

import os
import sys
import argparse
import yaml
import json
import joblib
import pandas as pd
import numpy as np

from src.candidate_generation import create_chronological_splits, build_candidate_universe, audit_candidate_recall, build_labeled_pairs
from src.features import assemble_feature_matrix, build_co_purchase_matrix
from src.models_core import RandomForestRecommender, tune_random_forest_grid, save_and_verify_model


def main():
    parser = argparse.ArgumentParser(description="Retrain Random Forest Recommender")
    parser.add_argument("--tune", action="store_true", default=True, help="Run full hyperparameter grid search on validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 80)
    print("MDI3003 Lab 07: Retrain Pipeline (Random Forest Recommender)")
    print("=" * 80)

    # 1. Load cleaned D1 data
    clean_path = "data/D1_uci_online_retail/online_retail_cleaned.csv"
    if not os.path.exists(clean_path):
        from src.data_ingest import run_data_ingestion
        run_data_ingestion()

    df = pd.read_csv(clean_path)
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    print(f"[DATA] Loaded cleaned D1 data: {len(df):,} transactions from {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

    # 2. Chronological splits
    train_hist, val_future, test_future, split_manifest = create_chronological_splits(df, q1=0.70, q2=0.85)
    print(f"[SPLIT] Train History: {len(train_hist):,} rows | Val Future: {len(val_future):,} rows | Test Future: {len(test_future):,} rows")

    # 3. Candidate universe & Candidate recall audit
    candidate_items, cand_policy = build_candidate_universe(train_hist, min_items=500, max_items=2000, top_n=1000)
    print(f"[CANDIDATES] Bounded candidate universe: {len(candidate_items)} items")

    val_audit = audit_candidate_recall(val_future, candidate_items, split_name="Validation Split")
    test_audit = audit_candidate_recall(test_future, candidate_items, split_name="Locked Test Split")
    print(f"[AUDIT] Validation Candidate Recall: {val_audit['aggregate_candidate_recall']:.4f} ({val_audit['users_fully_representable_pct']}% fully representable users)")
    print(f"[AUDIT] Test Candidate Recall: {test_audit['aggregate_candidate_recall']:.4f} ({test_audit['users_fully_representable_pct']}% fully representable users)")

    # Save Candidate Recall Table F
    df_cand_recall = pd.DataFrame([val_audit, test_audit])
    os.makedirs("results", exist_ok=True)
    df_cand_recall.to_csv("results/Candidate_Recall.csv", index=False)
    df_cand_recall.to_csv("23MID0037_Lab07_Candidate_Recall.csv", index=False)

    # 4. Construct training & validation pairs
    train_users = train_hist["CustomerID"].unique().tolist()
    val_users = [u for u in val_future["CustomerID"].unique() if u in set(train_users)]
    val_positives = val_future.groupby("CustomerID")["StockCode"].apply(lambda s: set(s.astype(str))).to_dict()

    train_user_positives = train_hist.groupby("CustomerID")["StockCode"].apply(lambda s: list(s.astype(str))).to_dict()
    item_weights = train_hist["StockCode"].value_counts().to_dict()

    print("[PAIRS] Sampling negative pairs for training and validation...")
    train_pairs = build_labeled_pairs(train_users[:1500], train_user_positives, candidate_items, n_neg=30, seed=args.seed, item_weights=item_weights)
    val_pairs = build_labeled_pairs(val_users[:500], val_positives, candidate_items, n_neg=30, seed=args.seed, item_weights=item_weights)

    # 5. Feature Engineering (Strictly H < t1 for train/val)
    t1_ts = pd.Timestamp(split_manifest["cutoff_t1"])
    co_occur = build_co_purchase_matrix(train_hist)
    print("[FEATURES] Assembling cutoff-valid feature matrices...")
    X_train_df, feat_cols, feat_schema = assemble_feature_matrix(train_pairs, train_hist, t1_ts, co_occur)
    X_val_df, _, _ = assemble_feature_matrix(val_pairs, train_hist, t1_ts, co_occur)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/feature_schema.json", "w") as f:
        json.dump(feat_schema, f, indent=2)

    y_train = train_pairs["label"].values
    y_val = val_pairs["label"].values

    # 6. Model Training & Tuning
    if args.tune:
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
        params = best_cfg["params"]
        print(f"\n[WINNER] Selected configuration: {params} with Val Recall@10 = {best_cfg['val_recall_10']:.4f}")
    else:
        params = {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2, "max_features": "sqrt", "class_weight": "balanced_subsample"}

    # 7. Refit winner on full dev set and save
    print("[TRAIN] Fitting final Random Forest model...")
    rf_rec = RandomForestRecommender(**params, random_state=args.seed)
    rf_rec.fit(X_train_df, y_train, feat_cols)

    save_and_verify_model(rf_rec.model, feat_cols, X_val_df.head(20), save_path="models/random_forest.joblib")
    print(f"\n[SUCCESS] Retrain pipeline completed in {rf_rec.training_time:.2f}s!")


if __name__ == "__main__":
    main()
