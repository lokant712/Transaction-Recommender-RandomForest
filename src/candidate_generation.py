"""
Candidate Generation, Negative Sampling, and Candidate-Recall Audit Module.
Author: Lokanth S (23MID0037)
Faculty: Dr. Durgesh Kumar

Implements chronological windowing, bounded candidate universe construction (500-2,000 items),
candidate recall auditing (Section 11.3), and reproducible negative sampling policies (E5).
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

SEED = 42


def create_chronological_splits(
    df: pd.DataFrame,
    date_col: str = "InvoiceDate",
    q1: float = 0.70,
    q2: float = 0.85
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split transaction data into train history, validation future, and locked test future.
    Cutoffs are strictly chronological and frozen once.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    t1 = df[date_col].quantile(q1)
    t2 = df[date_col].quantile(q2)

    # For timestamp / string safety, record ISO strings
    t1_ts = pd.Timestamp(t1)
    t2_ts = pd.Timestamp(t2)

    train_hist = df[df[date_col] < t1_ts].copy()
    val_future = df[(df[date_col] >= t1_ts) & (df[date_col] < t2_ts)].copy()
    test_future = df[df[date_col] >= t2_ts].copy()

    split_manifest = {
        "quantile_t1": q1,
        "quantile_t2": q2,
        "cutoff_t1": t1_ts.isoformat(),
        "cutoff_t2": t2_ts.isoformat(),
        "train_hist_rows": int(len(train_hist)),
        "train_hist_date_range": [train_hist[date_col].min().isoformat(), train_hist[date_col].max().isoformat()],
        "train_hist_users": int(train_hist["CustomerID"].nunique()),
        "train_hist_items": int(train_hist["StockCode"].nunique()),
        "val_future_rows": int(len(val_future)),
        "val_future_date_range": [val_future[date_col].min().isoformat(), val_future[date_col].max().isoformat()],
        "val_future_users": int(val_future["CustomerID"].nunique()),
        "test_future_rows": int(len(test_future)),
        "test_future_date_range": [test_future[date_col].min().isoformat(), test_future[date_col].max().isoformat()],
        "test_future_users": int(test_future["CustomerID"].nunique()),
        "leakage_rule": "Every feature for validation/test must be computed strictly before cutoff_t1 and cutoff_t2 respectively."
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/split_manifest.json", "w") as f:
        json.dump(split_manifest, f, indent=2)

    return train_hist, val_future, test_future, split_manifest


def build_candidate_universe(
    train_hist: pd.DataFrame,
    item_col: str = "StockCode",
    min_items: int = 500,
    max_items: int = 2000,
    top_n: int = 1000
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Build bounded candidate universe from training history only (H < t1).
    Ensures candidate pool is strictly bounded between 500 and 2,000 items (Appendix C).
    """
    item_counts = train_hist[item_col].value_counts()
    
    # Select top_n items
    selected_items = item_counts.head(top_n).index.tolist()
    if len(selected_items) < min_items:
        selected_items = item_counts.head(min_items).index.tolist()
    elif len(selected_items) > max_items:
        selected_items = item_counts.head(max_items).index.tolist()

    assert min_items <= len(selected_items) <= max_items, (
        f"Candidate universe size {len(selected_items)} outside [{min_items}, {max_items}]"
    )

    policy = {
        "version": "1.0.0",
        "item_column": item_col,
        "selection_strategy": "top_frequent_in_training_history",
        "num_candidate_items": len(selected_items),
        "min_items_bound": min_items,
        "max_items_bound": max_items,
        "most_popular_item": str(selected_items[0]),
        "least_popular_candidate_item": str(selected_items[-1]),
        "allow_repeat_purchases": True,
        "filter_seen_items_for_baseline": False
    }

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/candidate_policy.json", "w") as f:
        json.dump(policy, f, indent=2)

    return selected_items, policy


def audit_candidate_recall(
    eval_future_df: pd.DataFrame,
    candidate_items: List[str],
    user_col: str = "CustomerID",
    item_col: str = "StockCode",
    split_name: str = "Locked Test"
) -> Dict[str, Any]:
    """
    Candidate-recall audit (Section 11.3 / 19.5).
    Candidate Recall = |Future relevant items ∩ Candidate set| / |Future relevant items|
    Audits aggregate recall, user-level recall, and percentage of users with 100% representable future positives.
    """
    candidate_set = set(candidate_items)
    user_positives = eval_future_df.groupby(user_col)[item_col].apply(lambda s: set(s.astype(str))).to_dict()

    total_future_positives = 0
    total_retrievable_positives = 0
    user_recalls = []
    fully_representable_users = 0

    for user, pos_set in user_positives.items():
        if not pos_set:
            continue
        n_pos = len(pos_set)
        n_hit = len(pos_set & candidate_set)
        total_future_positives += n_pos
        total_retrievable_positives += n_hit
        u_rec = n_hit / n_pos
        user_recalls.append(u_rec)
        if u_rec == 1.0:
            fully_representable_users += 1

    agg_candidate_recall = total_retrievable_positives / max(total_future_positives, 1)
    mean_user_candidate_recall = float(np.mean(user_recalls)) if user_recalls else 0.0
    pct_fully_representable = (fully_representable_users / max(len(user_positives), 1)) * 100.0

    audit_result = {
        "split": split_name,
        "num_eval_users": len(user_positives),
        "total_future_positives": total_future_positives,
        "retrievable_future_positives": total_retrievable_positives,
        "excluded_future_positives": total_future_positives - total_retrievable_positives,
        "aggregate_candidate_recall": round(agg_candidate_recall, 4),
        "mean_user_candidate_recall": round(mean_user_candidate_recall, 4),
        "users_fully_representable_count": fully_representable_users,
        "users_fully_representable_pct": round(pct_fully_representable, 2),
        "candidate_catalog_size": len(candidate_items),
        "policy_version": "1.0.0"
    }

    return audit_result


def sample_negatives(
    positives: List[str],
    candidate_items: List[str],
    n_neg: int = 50,
    rng: np.random.Generator = None,
    policy: str = "uniform",
    item_weights: Dict[str, float] = None
) -> List[str]:
    """
    Sample unpurchased candidate items for a user.
    Supports uniform random sampling or popularity-skewed sampling (E5).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    pool = [it for it in candidate_items if it not in set(positives)]
    if not pool:
        return []

    n = min(n_neg, len(pool))
    if policy == "uniform" or item_weights is None:
        return rng.choice(pool, size=n, replace=False).tolist()
    elif policy == "popularity_biased":
        # Weight by frequency in item_weights
        weights = np.array([item_weights.get(it, 1.0) for it in pool], dtype=float)
        weights = weights / weights.sum()
        return rng.choice(pool, size=n, replace=False, p=weights).tolist()
    else:
        return rng.choice(pool, size=n, replace=False).tolist()


def build_labeled_pairs(
    users: List[str],
    user_positives: Dict[str, List[str]],
    candidate_items: List[str],
    n_neg: int = 50,
    seed: int = SEED,
    policy: str = "uniform",
    item_weights: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Construct positive (y=1) and sampled negative (y=0) pairs for given users.
    """
    rng = np.random.default_rng(seed)
    records = []

    for u in users:
        pos_items = list(set(user_positives.get(u, [])))
        # Filter positives to candidate universe
        retrievable_pos = [it for it in pos_items if it in set(candidate_items)]
        for it in retrievable_pos:
            records.append({"CustomerID": str(u), "StockCode": str(it), "label": 1})
        
        # Negatives
        neg_items = sample_negatives(
            positives=pos_items,
            candidate_items=candidate_items,
            n_neg=n_neg,
            rng=rng,
            policy=policy,
            item_weights=item_weights
        )
        for it in neg_items:
            records.append({"CustomerID": str(u), "StockCode": str(it), "label": 0})

    return pd.DataFrame(records)
