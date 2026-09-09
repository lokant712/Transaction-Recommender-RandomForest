"""
Feature Engineering Module for MDI3003 Lab 07.
Author: Lokanth S (23MID0037)
Faculty: Dr. Durgesh Kumar

Extracts leakage-safe customer, item, customer-item pair, and context features
strictly from the transaction history available prior to the given cutoff date.
"""

import json
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any


def compute_customer_features(hist: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Compute customer-level behavioral RFM features strictly from historical data H < cutoff.
    """
    hist = hist.copy()
    hist["CustomerID"] = hist["CustomerID"].astype(str)
    hist["StockCode"] = hist["StockCode"].astype(str)
    g = hist.groupby("CustomerID")
    agg_df = g.agg(
        cust_txns=("InvoiceNo", "nunique"),
        cust_items=("StockCode", "nunique"),
        cust_qty=("Quantity", "sum"),
        cust_spend=("Amount", "sum"),
        cust_last=("InvoiceDate", "max"),
        cust_first=("InvoiceDate", "min")
    ).reset_index()

    agg_df["cust_recency_days"] = (cutoff - agg_df["cust_last"]).dt.total_seconds() / (24 * 3600.0)
    agg_df["cust_active_span_days"] = (agg_df["cust_last"] - agg_df["cust_first"]).dt.total_seconds() / (24 * 3600.0)
    agg_df["cust_avg_basket_size"] = agg_df["cust_qty"] / np.maximum(agg_df["cust_txns"], 1)
    agg_df["cust_avg_item_price"] = agg_df["cust_spend"] / np.maximum(agg_df["cust_qty"], 1e-5)
    agg_df["CustomerID"] = agg_df["CustomerID"].astype(str)

    return agg_df.drop(columns=["cust_last", "cust_first"])


def compute_item_features(hist: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Compute item-level statistics strictly from historical data H < cutoff.
    """
    hist = hist.copy()
    hist["CustomerID"] = hist["CustomerID"].astype(str)
    hist["StockCode"] = hist["StockCode"].astype(str)
    g = hist.groupby("StockCode")
    agg_df = g.agg(
        item_txns=("InvoiceNo", "nunique"),
        item_buyers=("CustomerID", "nunique"),
        item_qty=("Quantity", "sum"),
        item_spend=("Amount", "sum"),
        item_avg_price=("UnitPrice", "mean"),
        item_last=("InvoiceDate", "max"),
        item_first=("InvoiceDate", "min")
    ).reset_index()

    agg_df["item_recency_days"] = (cutoff - agg_df["item_last"]).dt.total_seconds() / (24 * 3600.0)
    agg_df["item_active_span_days"] = (agg_df["item_last"] - agg_df["item_first"]).dt.total_seconds() / (24 * 3600.0)

    # Compute repeat buyer rate
    user_item_counts = hist.groupby(["StockCode", "CustomerID"])["InvoiceNo"].nunique().reset_index()
    repeat_buyers = user_item_counts[user_item_counts["InvoiceNo"] > 1].groupby("StockCode")["CustomerID"].nunique().reset_index(name="repeat_buyers")
    repeat_buyers["StockCode"] = repeat_buyers["StockCode"].astype(str)
    agg_df["StockCode"] = agg_df["StockCode"].astype(str)
    agg_df = agg_df.merge(repeat_buyers, on="StockCode", how="left")
    agg_df["repeat_buyers"] = agg_df["repeat_buyers"].fillna(0)
    agg_df["item_repeat_rate"] = agg_df["repeat_buyers"] / np.maximum(agg_df["item_buyers"], 1)
    agg_df["item_popularity_rank"] = agg_df["item_txns"].rank(ascending=False, method="dense")

    return agg_df.drop(columns=["cust_last", "cust_first", "repeat_buyers"], errors="ignore").drop(columns=["item_last", "item_first"], errors="ignore")


def compute_pair_features(hist: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Compute customer-item interaction features strictly from historical data H < cutoff.
    """
    hist = hist.copy()
    hist["CustomerID"] = hist["CustomerID"].astype(str)
    hist["StockCode"] = hist["StockCode"].astype(str)
    pair_agg = hist.groupby(["CustomerID", "StockCode"]).agg(
        pair_purchases=("InvoiceNo", "nunique"),
        pair_qty=("Quantity", "sum"),
        pair_spend=("Amount", "sum"),
        pair_last=("InvoiceDate", "max")
    ).reset_index()

    pair_agg["CustomerID"] = pair_agg["CustomerID"].astype(str)
    pair_agg["StockCode"] = pair_agg["StockCode"].astype(str)
    pair_agg["pair_days_since_last"] = (cutoff - pair_agg["pair_last"]).dt.total_seconds() / (24 * 3600.0)
    pair_agg["pair_repeat_indicator"] = (pair_agg["pair_purchases"] > 1).astype(int)
    pair_agg["pair_has_prior_purchase"] = 1

    return pair_agg.drop(columns=["pair_last"])


def build_co_purchase_matrix(hist: pd.DataFrame, top_k_items: int = 500) -> Dict[str, Dict[str, float]]:
    """
    Compute item-item co-occurrence matrix from historical baskets for co-purchase affinity feature.
    """
    hist = hist.copy()
    hist["StockCode"] = hist["StockCode"].astype(str)
    baskets = hist.groupby("InvoiceNo")["StockCode"].apply(set).tolist()
    item_freq = hist["StockCode"].value_counts().to_dict()
    co_occur = {}

    for b in baskets:
        if len(b) < 2 or len(b) > 50:
            continue
        items = list(b)
        for i in range(len(items)):
            it1 = items[i]
            if it1 not in co_occur:
                co_occur[it1] = {}
            for j in range(i + 1, len(items)):
                it2 = items[j]
                if it2 not in co_occur:
                    co_occur[it2] = {}
                co_occur[it1][it2] = co_occur[it1].get(it2, 0) + 1
                co_occur[it2][it1] = co_occur[it2].get(it1, 0) + 1

    # Normalize to Jaccard / cosine-like score
    affinity = {}
    for it1, neighbors in co_occur.items():
        affinity[it1] = {}
        for it2, count in neighbors.items():
            f1 = item_freq.get(it1, 1)
            f2 = item_freq.get(it2, 1)
            affinity[it1][it2] = count / np.sqrt(f1 * f2)
    return affinity


def assemble_feature_matrix(
    pair_df: pd.DataFrame,
    hist: pd.DataFrame,
    cutoff: pd.Timestamp,
    co_purchase_affinity: Dict[str, Dict[str, float]] = None
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Join customer, item, and pair features onto candidate pairs for scoring.
    """
    pair_df = pair_df.copy()
    pair_df["CustomerID"] = pair_df["CustomerID"].astype(str)
    pair_df["StockCode"] = pair_df["StockCode"].astype(str)

    cust_df = compute_customer_features(hist, cutoff)
    item_df = compute_item_features(hist, cutoff)
    pair_feat_df = compute_pair_features(hist, cutoff)

    # Merge customer features
    merged = pair_df.merge(cust_df, on="CustomerID", how="left")
    # Merge item features
    merged = merged.merge(item_df, on="StockCode", how="left")
    # Merge pair features
    merged = merged.merge(pair_feat_df, on=["CustomerID", "StockCode"], how="left")

    # Fill defaults for non-interacted pairs
    merged["pair_purchases"] = merged["pair_purchases"].fillna(0)
    merged["pair_qty"] = merged["pair_qty"].fillna(0)
    merged["pair_spend"] = merged["pair_spend"].fillna(0)
    merged["pair_days_since_last"] = merged["pair_days_since_last"].fillna(999.0)
    merged["pair_repeat_indicator"] = merged["pair_repeat_indicator"].fillna(0)
    merged["pair_has_prior_purchase"] = merged["pair_has_prior_purchase"].fillna(0)

    # Relative shares
    merged["pair_spend_share"] = merged["pair_spend"] / np.maximum(merged["cust_spend"].fillna(0), 1e-5)
    merged["pair_qty_share"] = merged["pair_qty"] / np.maximum(merged["cust_qty"].fillna(0), 1e-5)

    # Customer fillna (e.g. cold-start users)
    cust_cols = ["cust_txns", "cust_items", "cust_qty", "cust_spend", "cust_recency_days", "cust_active_span_days", "cust_avg_basket_size", "cust_avg_item_price"]
    for c in cust_cols:
        merged[c] = merged[c].fillna(0.0)

    # Item fillna (e.g. cold-start items)
    item_cols = ["item_txns", "item_buyers", "item_qty", "item_spend", "item_avg_price", "item_recency_days", "item_active_span_days", "item_repeat_rate", "item_popularity_rank"]
    for c in item_cols:
        merged[c] = merged[c].fillna(0.0)

    # Co-purchase affinity feature
    if co_purchase_affinity is not None:
        user_hist_items = hist.groupby("CustomerID")["StockCode"].apply(set).to_dict()
        def get_affinity(row):
            u = row["CustomerID"]
            it = row["StockCode"]
            h_items = user_hist_items.get(u, set())
            if not h_items:
                return 0.0
            scores = [co_purchase_affinity.get(it, {}).get(prev, 0.0) for prev in h_items]
            return max(scores) if scores else 0.0
        merged["pair_co_purchase_affinity"] = merged.apply(get_affinity, axis=1)
    else:
        merged["pair_co_purchase_affinity"] = 0.0

    # Context features from cutoff date
    merged["context_day_of_week"] = cutoff.dayofweek
    merged["context_month"] = cutoff.month
    merged["context_quarter"] = cutoff.quarter

    feature_cols = [
        # Customer RFM
        "cust_txns", "cust_items", "cust_qty", "cust_spend",
        "cust_recency_days", "cust_active_span_days", "cust_avg_basket_size", "cust_avg_item_price",
        # Item features
        "item_txns", "item_buyers", "item_qty", "item_spend",
        "item_avg_price", "item_recency_days", "item_active_span_days", "item_repeat_rate", "item_popularity_rank",
        # Pair interaction
        "pair_purchases", "pair_qty", "pair_spend", "pair_days_since_last",
        "pair_repeat_indicator", "pair_has_prior_purchase", "pair_spend_share", "pair_qty_share", "pair_co_purchase_affinity",
        # Context
        "context_day_of_week", "context_month", "context_quarter"
    ]

    schema = {
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "feature_groups": {
            "customer_rfm": ["cust_txns", "cust_items", "cust_qty", "cust_spend", "cust_recency_days", "cust_active_span_days", "cust_avg_basket_size", "cust_avg_item_price"],
            "item_popularity": ["item_txns", "item_buyers", "item_qty", "item_spend", "item_avg_price", "item_recency_days", "item_active_span_days", "item_repeat_rate", "item_popularity_rank"],
            "pair_history": ["pair_purchases", "pair_qty", "pair_spend", "pair_days_since_last", "pair_repeat_indicator", "pair_has_prior_purchase", "pair_spend_share", "pair_qty_share", "pair_co_purchase_affinity"],
            "context": ["context_day_of_week", "context_month", "context_quarter"]
        },
        "cutoff_date": cutoff.isoformat(),
        "leakage_proof": "All features aggregated strictly over transactions where InvoiceDate < cutoff."
    }

    return merged, feature_cols, schema
