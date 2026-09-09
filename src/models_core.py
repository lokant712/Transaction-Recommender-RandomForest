"""
Core Recommendation Models Module: Popularity Baseline and Random Forest Ranker.
Author: Lokanth S (23MID0037)
Faculty: Dr. Durgesh Kumar

Implements Popularity baseline, Random Forest candidate scoring ranker,
and exhaustive training-only hyperparameter tuning grid (Table G) with validation selection.
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

from src.evaluate import evaluate_recommendations, compute_pair_classification_metrics

SEED = 42


class PopularityRecommender:
    """
    Time-valid Popularity Baseline recommender.
    Ranks candidate items based strictly on purchase count in historical transactions.
    """
    def __init__(self, item_col: str = "StockCode"):
        self.item_col = item_col
        self.popular_items_ = []
        self.item_scores_ = {}

    def fit(self, hist_df: pd.DataFrame, candidate_items: List[str] = None):
        """Fit popularity ordering from historical transactions only."""
        counts = hist_df.groupby(self.item_col)["InvoiceNo"].nunique()
        if candidate_items is not None:
            # Reindex to candidate items
            cand_set = set(candidate_items)
            cand_counts = {it: counts.get(it, 0) for it in candidate_items}
            sorted_items = sorted(cand_counts.items(), key=lambda x: x[1], reverse=True)
            self.popular_items_ = [it for it, c in sorted_items]
            self.item_scores_ = dict(sorted_items)
        else:
            sorted_s = counts.sort_values(ascending=False)
            self.popular_items_ = sorted_s.index.astype(str).tolist()
            self.item_scores_ = sorted_s.to_dict()
        return self

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        seen_items: Set[str] = None,
        filter_seen: bool = False
    ) -> List[str]:
        """Generate Top-K popular recommendations for a user."""
        if filter_seen and seen_items:
            recs = [it for it in self.popular_items_ if it not in seen_items]
        else:
            recs = self.popular_items_
        return recs[:k]

    def batch_recommend(
        self,
        users: List[str],
        user_seen_items: Dict[str, Set[str]] = None,
        k: int = 10,
        filter_seen: bool = False
    ) -> Dict[str, List[str]]:
        """Batch recommendation for evaluation users."""
        recs_dict = {}
        for u in users:
            seen = user_seen_items.get(u, set()) if user_seen_items else set()
            recs_dict[str(u)] = self.recommend(str(u), k=k, seen_items=seen, filter_seen=filter_seen)
        return recs_dict


class RandomForestRecommender:
    """
    Supervised candidate-scoring Random Forest ranker.
    Estimates P(y=1 | x_{u,i,t}) and sorts candidate items descending by score.
    """
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = None,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        class_weight: str = "balanced_subsample",
        random_state: int = SEED,
        n_jobs: int = -1
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": n_jobs
        }
        self.model = RandomForestClassifier(**self.params)
        self.feature_cols = []
        self.training_time = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray, feature_cols: List[str]):
        """Fit Random Forest on feature matrix."""
        self.feature_cols = list(feature_cols)
        t0 = time.time()
        self.model.fit(X[self.feature_cols], y)
        self.training_time = time.time() - t0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive class probabilities P(y=1 | x)."""
        return self.model.predict_proba(X[self.feature_cols])[:, 1]

    def recommend_for_user(
        self,
        user_id: str,
        user_candidate_features_df: pd.DataFrame,
        k: int = 10
    ) -> List[Tuple[str, float]]:
        """Score candidate items for single user and return Top-K (item, score)."""
        u_df = user_candidate_features_df[user_candidate_features_df["CustomerID"] == str(user_id)].copy()
        if u_df.empty:
            return []
        scores = self.predict_proba(u_df)
        u_df["score"] = scores
        top_k_df = u_df.sort_values("score", ascending=False).head(k)
        return list(zip(top_k_df["StockCode"].astype(str), top_k_df["score"].astype(float)))

    def batch_recommend(
        self,
        candidate_features_df: pd.DataFrame,
        users: List[str],
        k: int = 10
    ) -> Dict[str, List[str]]:
        """Score all candidate pairs and return Top-K items per user."""
        scores = self.predict_proba(candidate_features_df)
        df_scored = candidate_features_df[["CustomerID", "StockCode"]].copy()
        df_scored["score"] = scores

        # Group and rank
        recs_dict = {}
        grouped = df_scored.groupby("CustomerID")
        for u in users:
            u_str = str(u)
            if u_str in grouped.groups:
                g = grouped.get_group(u_str).sort_values("score", ascending=False)
                recs_dict[u_str] = g["StockCode"].head(k).astype(str).tolist()
            else:
                recs_dict[u_str] = []
        return recs_dict

    def get_feature_importances(self) -> pd.DataFrame:
        """Extract MDI feature importances."""
        importances = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False).reset_index(drop=True)


def tune_random_forest_grid(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    feature_cols: List[str],
    val_users: List[str],
    val_positives: Dict[str, Set[str]],
    catalog_items: List[str],
    grid_params: Dict[str, List[Any]] = None
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Exhaustive Random Forest hyperparameter tuning grid on VALIDATION data only.
    Tuning transparency requirement (Manual Section 13.3 & Faculty Feedback Item 1).
    """
    if grid_params is None:
        grid_params = {
            "n_estimators": [200, 400],
            "max_depth": [None, 12, 20],
            "min_samples_leaf": [1, 2, 5],
            "max_features": ["sqrt", 0.5],
            "class_weight": ["balanced_subsample"]
        }

    param_list = list(ParameterGrid(grid_params))
    print(f"[TUNING] Evaluating {len(param_list)} Random Forest configurations on VALIDATION set only...")

    results = []
    best_config = None
    best_score = -1.0

    for idx, params in enumerate(param_list, start=1):
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params["class_weight"],
            random_state=SEED,
            n_jobs=-1
        )
        rf.fit(X_train[feature_cols], y_train)
        fit_time = time.time() - t0

        # Predict validation probabilities
        val_probs = rf.predict_proba(X_val[feature_cols])[:, 1]
        pair_metrics = compute_pair_classification_metrics(y_val, val_probs)

        # Batch recommendations on validation candidate pairs
        df_val_scored = X_val[["CustomerID", "StockCode"]].copy()
        df_val_scored["CustomerID"] = df_val_scored["CustomerID"].astype(str)
        df_val_scored["StockCode"] = df_val_scored["StockCode"].astype(str)
        df_val_scored["score"] = val_probs
        val_recs = {}
        grouped_val = df_val_scored.groupby("CustomerID")
        for u in val_users:
            u_str = str(u)
            if u_str in grouped_val.groups:
                val_recs[u_str] = grouped_val.get_group(u_str).sort_values("score", ascending=False)["StockCode"].head(10).tolist()
            else:
                val_recs[u_str] = []

        val_pos_clean = {str(k): {str(x) for x in v} for k, v in val_positives.items()}
        ranking_metrics = evaluate_recommendations(
            val_recs, val_pos_clean, catalog_items, k_list=[5, 10, 20]
        )

        rec_10 = ranking_metrics["Recall@10"]
        ndcg_10 = ranking_metrics["NDCG@10"]
        pr_auc = pair_metrics["PR-AUC"]
        roc_auc = pair_metrics["ROC-AUC"]

        is_best = False
        if rec_10 > best_score or (rec_10 == best_score and ndcg_10 > best_config.get("val_ndcg_10", -1.0)):
            best_score = rec_10
            best_config = {
                "params": params,
                "val_recall_10": rec_10,
                "val_ndcg_10": ndcg_10,
                "val_pr_auc": pr_auc,
                "val_roc_auc": roc_auc,
                "fit_time": fit_time
            }

        row = {
            "config_id": idx,
            "n_estimators": params["n_estimators"],
            "max_depth": str(params["max_depth"]),
            "min_samples_leaf": params["min_samples_leaf"],
            "max_features": str(params["max_features"]),
            "class_weight": str(params["class_weight"]),
            "val_precision_10": ranking_metrics["Precision@10"],
            "val_recall_10": rec_10,
            "val_hitrate_10": ranking_metrics["HitRate@10"],
            "val_ndcg_10": ndcg_10,
            "val_pr_auc": pr_auc,
            "val_roc_auc": roc_auc,
            "fit_time_sec": round(fit_time, 2),
            "selected": "NO"
        }
        results.append(row)
        print(f"  Config {idx:02d}/{len(param_list):02d} | depth={params['max_depth']}, leaf={params['min_samples_leaf']}, feat={params['max_features']} -> Val Recall@10={rec_10:.4f}, NDCG@10={ndcg_10:.4f}, PR-AUC={pr_auc:.4f}")

    df_grid = pd.DataFrame(results)
    # Mark winner
    best_idx = df_grid["val_recall_10"].idxmax()
    df_grid.loc[best_idx, "selected"] = "YES (WINNER)"

    os.makedirs("results", exist_ok=True)
    table_g_path = "results/RF_Tuning_Grid.csv"
    df_grid.to_csv(table_g_path, index=False)
    print(f"[TABLE G] Saved Random Forest tuning grid to {table_g_path}")

    return best_config, df_grid


def save_and_verify_model(
    model: RandomForestClassifier,
    feature_cols: List[str],
    verification_X: pd.DataFrame,
    save_path: str = "models/random_forest.joblib"
) -> bool:
    """Save model to disk and verify predictions match in-memory instance."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[MODEL] Saved Random Forest model to {save_path}")

    loaded_model = joblib.load(save_path)
    mem_preds = model.predict_proba(verification_X[feature_cols])[:, 1]
    disk_preds = loaded_model.predict_proba(verification_X[feature_cols])[:, 1]
    diff = np.max(np.abs(mem_preds - disk_preds))
    assert diff < 1e-6, f"Saved model prediction mismatch! Max diff: {diff}"
    print(f"[VERIFY] Saved model reloaded successfully (max diff = {diff:.2e})")
    return True
