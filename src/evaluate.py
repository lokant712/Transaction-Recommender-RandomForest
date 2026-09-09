"""
Evaluation and Ranking Metrics Module for MDI3003 Lab 07.
Author: Lokanth S (23MID0037)
Faculty: Dr. Durgesh Kumar

Implements Precision@K, Recall@K, HitRate@K, MAP@K, NDCG@K,
Catalog Coverage, Gini Index, ROC-AUC, and PR-AUC.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Union, Any
from sklearn.metrics import roc_auc_score, average_precision_score


def precision_at_k(recs: List[str], relevant: Set[str], k: int = 10) -> float:
    """Precision@K = |recs[:k] ∩ relevant| / k."""
    if k <= 0:
        return 0.0
    r = recs[:k]
    hits = len(set(r) & set(relevant))
    return hits / float(k)


def recall_at_k(recs: List[str], relevant: Set[str], k: int = 10) -> float:
    """Recall@K = |recs[:k] ∩ relevant| / |relevant|."""
    if not relevant:
        return np.nan
    r = recs[:k]
    hits = len(set(r) & set(relevant))
    return hits / float(len(relevant))


def hit_rate_at_k(recs: List[str], relevant: Set[str], k: int = 10) -> float:
    """HitRate@K = 1 if |recs[:k] ∩ relevant| > 0 else 0."""
    r = recs[:k]
    return 1.0 if len(set(r) & set(relevant)) > 0 else 0.0


def average_precision_at_k(recs: List[str], relevant: Set[str], k: int = 10) -> float:
    """Average Precision at K (AP@K)."""
    if not relevant:
        return np.nan
    r = recs[:k]
    hits = 0.0
    score = 0.0
    for rank, item in enumerate(r, start=1):
        if item in relevant:
            hits += 1.0
            score += hits / rank
    denom = min(len(relevant), k)
    return (score / denom) if denom > 0 else 0.0


def ndcg_at_k(recs: List[str], relevant: Set[str], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at K (NDCG@K)."""
    if not relevant:
        return np.nan
    r = recs[:k]
    dcg = 0.0
    for rank, item in enumerate(r, start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    
    # Ideal DCG
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def catalog_coverage(all_recs: List[List[str]], catalog_size: int, k: int = 10) -> float:
    """Catalog Coverage = |unique recommended items in Top-K| / catalog_size."""
    unique_rec_items = set()
    for rec_list in all_recs:
        unique_rec_items.update(rec_list[:k])
    return len(unique_rec_items) / float(max(catalog_size, 1))


def gini_coefficient(item_counts: np.ndarray) -> float:
    """Gini coefficient of item recommendation distribution (0 = perfectly equal, 1 = maximum inequality)."""
    if len(item_counts) == 0:
        return 0.0
    counts = np.sort(np.array(item_counts, dtype=float))
    n = len(counts)
    if counts.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * counts) - (n + 1) * np.sum(counts)) / (n * np.sum(counts)))


def evaluate_recommendations(
    user_recommendations: Dict[str, List[str]],
    user_positives: Dict[str, Set[str]],
    catalog_items: List[str],
    k_list: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Compute comprehensive ranking metrics across all evaluation users.
    """
    clean_positives = {str(k): {str(x) for x in v} for k, v in user_positives.items()}
    clean_recs = {str(k): [str(x) for x in v] for k, v in user_recommendations.items()}
    eval_users = [u for u, pos in clean_positives.items() if len(pos) > 0]
    metrics = {}

    all_recs_by_user = [clean_recs.get(u, []) for u in eval_users]

    for k in k_list:
        precisions = [precision_at_k(clean_recs.get(u, []), clean_positives[u], k=k) for u in eval_users]
        recalls = [recall_at_k(clean_recs.get(u, []), clean_positives[u], k=k) for u in eval_users]
        hitrates = [hit_rate_at_k(clean_recs.get(u, []), clean_positives[u], k=k) for u in eval_users]
        maps = [average_precision_at_k(clean_recs.get(u, []), clean_positives[u], k=k) for u in eval_users]
        ndcgs = [ndcg_at_k(clean_recs.get(u, []), clean_positives[u], k=k) for u in eval_users]

        metrics[f"Precision@{k}"] = round(float(np.nanmean(precisions)), 4)
        metrics[f"Recall@{k}"] = round(float(np.nanmean(recalls)), 4)
        metrics[f"HitRate@{k}"] = round(float(np.nanmean(hitrates)), 4)
        metrics[f"MAP@{k}"] = round(float(np.nanmean(maps)), 4)
        metrics[f"NDCG@{k}"] = round(float(np.nanmean(ndcgs)), 4)
        metrics[f"Coverage@{k}"] = round(catalog_coverage(all_recs_by_user, len(catalog_items), k=k), 4)

    # Item frequency distribution across Top-10 recs
    item_freq = {}
    for r in all_recs_by_user:
        for it in r[:10]:
            item_freq[it] = item_freq.get(it, 0) + 1
    all_counts = [item_freq.get(it, 0) for it in catalog_items]
    metrics["GiniIndex@10"] = round(gini_coefficient(np.array(all_counts)), 4)

    return metrics


def compute_pair_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """Compute pair-level diagnostic classification metrics (ROC-AUC, PR-AUC)."""
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_score))
    except Exception:
        pr_auc = float(np.mean(y_true))
    return {
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4)
    }


def compute_bootstrap_ci(
    user_recommendations: Dict[str, List[str]],
    user_positives: Dict[str, Set[str]],
    metric_func,
    k: int = 10,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Compute user-level bootstrap 95% confidence interval for ranking metric (Appendix D).
    """
    eval_users = [u for u in user_positives.keys() if len(user_positives[u]) > 0]
    n_users = len(eval_users)
    user_scores = np.array([metric_func(user_recommendations.get(u, []), user_positives[u], k=k) for u in eval_users])
    
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_users, size=n_users, replace=True)
        boot_means.append(np.nanmean(user_scores[idx]))
    
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    mean_val = float(np.mean(boot_means))
    std_val = float(np.std(boot_means))

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4)
    }
