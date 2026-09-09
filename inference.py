"""
Root CLI: Single-Customer Real-Time Inference Utility.
Usage: python inference.py --customer-id 17841 --top-k 10
"""

import os
import json
import argparse
import joblib
import pandas as pd
import numpy as np

from src.candidate_generation import create_chronological_splits, build_candidate_universe
from src.features import assemble_feature_matrix, build_co_purchase_matrix


def run_inference(customer_id: str, top_k: int = 10):
    clean_path = "data/D1_uci_online_retail/online_retail_cleaned.csv"
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"Cleaned dataset not found at {clean_path}. Run retrain.py first.")

    df = pd.read_csv(clean_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Load splits & artifacts
    train_hist, val_future, test_future, split_manifest = create_chronological_splits(df, q1=0.70, q2=0.85)
    t2_ts = pd.Timestamp(split_manifest["cutoff_t2"])
    dev_hist = df[df["InvoiceDate"] < t2_ts].copy()

    candidate_items, _ = build_candidate_universe(train_hist, min_items=500, max_items=2000, top_n=1000)

    model_path = "models/random_forest.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run retrain.py first.")
    model = joblib.load(model_path)

    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)
    feat_cols = schema["feature_names"]

    item_desc = df.drop_duplicates("StockCode").set_index("StockCode")["Description"].to_dict()

    # User history
    cust_id_str = str(customer_id)
    u_hist = dev_hist[dev_hist["CustomerID"] == cust_id_str]
    u_hist_items = u_hist["StockCode"].unique().tolist()

    print("\n" + "=" * 70)
    print(f"RECOMMENDATION INFERENCE FOR CUSTOMER: {cust_id_str}")
    print("=" * 70)
    print(f"Historical Transactions: {u_hist['InvoiceNo'].nunique()} orders, {len(u_hist_items)} unique items")
    print(f"Total Historical Spend: £{u_hist['Amount'].sum():.2f}")
    if u_hist_items:
        print("Sample Historical Purchases:")
        for it in u_hist_items[:5]:
            print(f"  - [{it}] {item_desc.get(it, 'N/A')}")

    # Build candidate pairs for this user
    cand_pairs = pd.DataFrame([{"CustomerID": cust_id_str, "StockCode": str(it)} for it in candidate_items])
    co_occur = build_co_purchase_matrix(dev_hist)
    feat_df, _, _ = assemble_feature_matrix(cand_pairs, dev_hist, t2_ts, co_occur)

    scores = model.predict_proba(feat_df[feat_cols])[:, 1]
    feat_df["score"] = scores
    top_df = feat_df.sort_values("score", ascending=False).head(top_k)

    print(f"\n--- TOP-{top_k} PERSONALIZED RECOMMENDATIONS ---")
    recs = []
    for rank, (_, row) in enumerate(top_df.iterrows(), start=1):
        stk = str(row["StockCode"])
        sc = float(row["score"])
        desc = item_desc.get(stk, "No description available")
        prior = "YES" if row["pair_has_prior_purchase"] == 1 else "NO (Discovery)"
        print(f"{rank:2d}. [{stk}] (Score: {sc:.4f} | Repeat: {prior}) - {desc}")
        recs.append({"rank": rank, "stock_code": stk, "score": sc, "description": desc, "is_repeat": prior})

    return recs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Customer Recommendation Inference")
    parser.add_argument("--customer-id", type=str, default="17841", help="Target CustomerID")
    parser.add_argument("--top-k", type=int, default=10, help="Number of items to recommend")
    args = parser.parse_args()

    run_inference(args.customer_id, args.top_k)
