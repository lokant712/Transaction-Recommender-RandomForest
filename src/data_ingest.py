"""
Data Ingestion and Integrity Module for MDI3003 Lab 07.
Author: Lokanth S (23MID0037)
Faculty: Dr. Durgesh Kumar

Handles download, extraction, validation, SHA-256 checksumming,
and leakage-safe transaction cleaning for D1, D2, D3, and D4.
"""

import os
import sys
import json
import shutil
import hashlib
import zipfile
import urllib.request
from datetime import datetime
import pandas as pd
import yaml

# Responsible Recommendation Guardrail (Manual Section 23)
GUARDRAIL_NOTICE = (
    "RESPONSIBLE RECOMMENDATION GUARDRAIL: Recommendation is a prediction of likely relevance, "
    "not proof of preference or intent. This system must not be used for discriminatory pricing, "
    "protected-class targeting, credit decisions, or manipulative personalization. Do not expose "
    "identifiable purchase histories in screenshots or shared reports. Assess popularity bias, "
    "concentration, and unfair exclusion of less-active customers or niche items."
)


def compute_sha256(filepath: str) -> str:
    """Compute exact byte-level SHA-256 hash of a file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot compute hash: {filepath} does not exist.")
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def download_file(url: str, dest_path: str) -> None:
    """Download file with progress and error handling."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f"[INGEST] File already exists: {dest_path}")
        return
    print(f"[INGEST] Downloading {url} -> {dest_path} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"[INGEST] Download complete: {dest_path} ({os.path.getsize(dest_path):,} bytes)")


def ingest_d1_uci_online_retail(base_dir: str = "data/D1_uci_online_retail") -> dict:
    """Download and process D1: UCI Online Retail."""
    os.makedirs(base_dir, exist_ok=True)
    zip_path = os.path.join(base_dir, "online_retail.zip")
    raw_csv_path = os.path.join(base_dir, "online_retail_raw.csv")
    cleaned_csv_path = os.path.join(base_dir, "online_retail_cleaned.csv")
    url = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"

    if not os.path.exists(raw_csv_path):
        download_file(url, zip_path)
        print(f"[INGEST] Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(base_dir)
        
        # Locate xlsx
        xlsx_candidates = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".xlsx")]
        if not xlsx_candidates:
            raise FileNotFoundError("Could not find Online Retail .xlsx in extracted zip.")
        xlsx_file = xlsx_candidates[0]
        print(f"[INGEST] Converting {xlsx_file} to CSV ...")
        df_raw = pd.read_excel(xlsx_file, engine="calamine")
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"[INGEST] Saved raw CSV: {raw_csv_path} ({len(df_raw):,} rows)")
    else:
        df_raw = pd.read_csv(raw_csv_path)

    raw_hash = compute_sha256(raw_csv_path)
    raw_count = len(df_raw)

    # Clean according to manual section 10.3 & 6.1
    # 1. Missing CustomerID
    missing_cust_count = df_raw["CustomerID"].isna().sum()
    df_clean = df_raw.dropna(subset=["CustomerID"]).copy()
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int).astype(str)

    # 2. Cancellations / returns (InvoiceNo starts with 'C')
    df_clean["is_cancel"] = df_clean["InvoiceNo"].astype(str).str.startswith("C")
    cancel_count = df_clean["is_cancel"].sum()
    df_clean = df_clean[~df_clean["is_cancel"]].copy()

    # 3. Positive quantity and non-negative UnitPrice
    invalid_qty_price = len(df_clean[(df_clean["Quantity"] <= 0) | (df_clean["UnitPrice"] < 0)])
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] >= 0)].copy()

    # 4. Dates & Amount
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["StockCode"] = df_clean["StockCode"].astype(str).str.strip()
    df_clean["Amount"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    df_clean = df_clean.sort_values("InvoiceDate").reset_index(drop=True)

    df_clean.to_csv(cleaned_csv_path, index=False)
    cleaned_hash = compute_sha256(cleaned_csv_path)

    card = {
        "dataset_id": "D1",
        "dataset_name": "UCI Online Retail",
        "role": "Core / primary benchmark",
        "source_url": url,
        "access_date": "2026-09-09",
        "raw_file": raw_csv_path,
        "raw_sha256": raw_hash,
        "cleaned_file": cleaned_csv_path,
        "cleaned_sha256": cleaned_hash,
        "raw_rows": int(raw_count),
        "filtered_rows": int(len(df_clean)),
        "missing_customer_ids_removed": int(missing_cust_count),
        "cancellations_removed": int(cancel_count),
        "invalid_qty_price_removed": int(invalid_qty_price),
        "unique_customers": int(df_clean["CustomerID"].nunique()),
        "unique_items": int(df_clean["StockCode"].nunique()),
        "date_range": f"{df_clean['InvoiceDate'].min().strftime('%Y-%m-%d')} to {df_clean['InvoiceDate'].max().strftime('%Y-%m-%d')}",
        "return_cancellation_policy": "Explicitly removed InvoiceNo starting with 'C'; never treated as positive relevance",
        "missing_id_policy": "Explicitly dropped unauthenticated sessions (missing CustomerID) with count audited",
        "privacy_usage_note": "Public research dataset; no direct PII; customer IDs treated as non-ordinal join keys"
    }
    return card


def ingest_d2_retailrocket(base_dir: str = "data/D2_retailrocket") -> dict:
    """Copy and process D2: Retailrocket Recommender System Dataset."""
    os.makedirs(base_dir, exist_ok=True)
    cache_dir = os.path.expanduser(r"~/.cache/kagglehub/datasets/retailrocket/ecommerce-dataset/versions/2")
    if not os.path.exists(cache_dir):
        import kagglehub
        cache_dir = kagglehub.dataset_download("retailrocket/ecommerce-dataset")

    dest_events = os.path.join(base_dir, "events.csv")
    if not os.path.exists(dest_events):
        shutil.copyfile(os.path.join(cache_dir, "events.csv"), dest_events)
        shutil.copyfile(os.path.join(cache_dir, "category_tree.csv"), os.path.join(base_dir, "category_tree.csv"))
        shutil.copyfile(os.path.join(cache_dir, "item_properties_part1.csv"), os.path.join(base_dir, "item_properties_part1.csv"))
        shutil.copyfile(os.path.join(cache_dir, "item_properties_part2.csv"), os.path.join(base_dir, "item_properties_part2.csv"))

    raw_hash = compute_sha256(dest_events)
    df_events = pd.read_csv(dest_events)
    raw_count = len(df_events)

    # Clean & standardize
    df_clean = df_events.dropna(subset=["visitorid", "itemid"]).copy()
    df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], unit="ms")
    df_clean = df_clean.sort_values("timestamp").reset_index(drop=True)
    cleaned_csv = os.path.join(base_dir, "retailrocket_cleaned.csv")
    df_clean.to_csv(cleaned_csv, index=False)
    cleaned_hash = compute_sha256(cleaned_csv)

    card = {
        "dataset_id": "D2",
        "dataset_name": "Retailrocket Recommender Dataset",
        "role": "Advanced implicit feedback benchmark",
        "source_url": "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
        "access_date": "2026-09-09",
        "raw_file": dest_events,
        "raw_sha256": raw_hash,
        "cleaned_file": cleaned_csv,
        "cleaned_sha256": cleaned_hash,
        "raw_rows": int(raw_count),
        "filtered_rows": int(len(df_clean)),
        "missing_customer_ids_removed": int(raw_count - len(df_clean)),
        "cancellations_removed": 0,
        "invalid_qty_price_removed": 0,
        "unique_customers": int(df_clean["visitorid"].nunique()),
        "unique_items": int(df_clean["itemid"].nunique()),
        "date_range": f"{df_clean['timestamp'].min().strftime('%Y-%m-%d')} to {df_clean['timestamp'].max().strftime('%Y-%m-%d')}",
        "return_cancellation_policy": "Implicit events (views, add-to-cart, transactions); views serve as implicit interaction signal",
        "missing_id_policy": "All rows contain visitorid; dropped any nulls if present",
        "privacy_usage_note": "Anonymized visitor IDs and item IDs; CC0 Public Domain"
    }
    return card


def ingest_d3_instacart(base_dir: str = "data/D3_instacart") -> dict:
    """Copy and process D3: Instacart Market Basket Analysis."""
    os.makedirs(base_dir, exist_ok=True)
    cache_dir = os.path.expanduser(r"~/.cache/kagglehub/datasets/psparks/instacart-market-basket-analysis/versions/1")
    if not os.path.exists(cache_dir):
        import kagglehub
        cache_dir = kagglehub.dataset_download("psparks/instacart-market-basket-analysis")

    # Copy key files
    files = ["orders.csv", "order_products__prior.csv", "order_products__train.csv", "products.csv", "aisles.csv", "departments.csv"]
    for f in files:
        src = os.path.join(cache_dir, f)
        dst = os.path.join(base_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)

    prior_orders_path = os.path.join(base_dir, "order_products__prior.csv")
    raw_hash = compute_sha256(prior_orders_path)
    df_prior = pd.read_csv(prior_orders_path)
    df_orders = pd.read_csv(os.path.join(base_dir, "orders.csv"))
    raw_count = len(df_prior)

    # Merge user_id from orders into prior interactions
    df_merged = df_prior.merge(df_orders[["order_id", "user_id", "order_number", "order_dow", "order_hour_of_day", "days_since_prior_order"]], on="order_id", how="inner")
    cleaned_csv = os.path.join(base_dir, "instacart_cleaned.csv")
    df_merged.to_csv(cleaned_csv, index=False)
    cleaned_hash = compute_sha256(cleaned_csv)

    card = {
        "dataset_id": "D3",
        "dataset_name": "Instacart Market Basket Analysis",
        "role": "Advanced repeat-purchase / next-basket extension",
        "source_url": "https://www.kaggle.com/competitions/instacart-market-basket-analysis/data",
        "access_date": "2026-09-09",
        "raw_file": prior_orders_path,
        "raw_sha256": raw_hash,
        "cleaned_file": cleaned_csv,
        "cleaned_sha256": cleaned_hash,
        "raw_rows": int(raw_count),
        "filtered_rows": int(len(df_merged)),
        "missing_customer_ids_removed": 0,
        "cancellations_removed": 0,
        "invalid_qty_price_removed": 0,
        "unique_customers": int(df_merged["user_id"].nunique()),
        "unique_items": int(df_merged["product_id"].nunique()),
        "date_range": "Relative sequence (orders 1 to 100 per user)",
        "return_cancellation_policy": "Completed grocery orders; reorder binary flag indicates repeat purchase",
        "missing_id_policy": "User IDs fully mapped through orders metadata table",
        "privacy_usage_note": "Anonymized grocery orders published by Instacart for research competitions"
    }
    return card


def ingest_d4_uci_online_retail_ii(base_dir: str = "data/D4_uci_online_retail_ii") -> dict:
    """Download and process D4: UCI Online Retail II."""
    os.makedirs(base_dir, exist_ok=True)
    zip_path = os.path.join(base_dir, "online_retail_ii.zip")
    raw_csv_path = os.path.join(base_dir, "online_retail_ii_raw.csv")
    cleaned_csv_path = os.path.join(base_dir, "online_retail_ii_cleaned.csv")
    url = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"

    if not os.path.exists(raw_csv_path):
        download_file(url, zip_path)
        print(f"[INGEST] Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(base_dir)
        
        # Locate xlsx
        xlsx_candidates = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(".xlsx")]
        if not xlsx_candidates:
            raise FileNotFoundError("Could not find Online Retail II .xlsx in extracted zip.")
        xlsx_file = xlsx_candidates[0]
        print(f"[INGEST] Converting {xlsx_file} (all sheets) to CSV ...")
        excel_obj = pd.ExcelFile(xlsx_file, engine="calamine")
        sheets = [excel_obj.parse(sheet_name) for sheet_name in excel_obj.sheet_names]
        df_raw = pd.concat(sheets, ignore_index=True)
        # Rename standard columns if Customer ID has space
        df_raw = df_raw.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice", "Invoice": "InvoiceNo"})
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"[INGEST] Saved raw CSV: {raw_csv_path} ({len(df_raw):,} rows)")
    else:
        df_raw = pd.read_csv(raw_csv_path)

    raw_hash = compute_sha256(raw_csv_path)
    raw_count = len(df_raw)

    missing_cust_count = df_raw["CustomerID"].isna().sum()
    df_clean = df_raw.dropna(subset=["CustomerID"]).copy()
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int).astype(str)

    df_clean["is_cancel"] = df_clean["InvoiceNo"].astype(str).str.startswith("C")
    cancel_count = df_clean["is_cancel"].sum()
    df_clean = df_clean[~df_clean["is_cancel"]].copy()

    invalid_qty_price = len(df_clean[(df_clean["Quantity"] <= 0) | (df_clean["UnitPrice"] < 0)])
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] >= 0)].copy()

    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["StockCode"] = df_clean["StockCode"].astype(str).str.strip()
    df_clean["Amount"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    df_clean = df_clean.sort_values("InvoiceDate").reset_index(drop=True)

    df_clean.to_csv(cleaned_csv_path, index=False)
    cleaned_hash = compute_sha256(cleaned_csv_path)

    card = {
        "dataset_id": "D4",
        "dataset_name": "UCI Online Retail II",
        "role": "Advanced 2-year cross-dataset replication / robustness",
        "source_url": url,
        "access_date": "2026-09-09",
        "raw_file": raw_csv_path,
        "raw_sha256": raw_hash,
        "cleaned_file": cleaned_csv_path,
        "cleaned_sha256": cleaned_hash,
        "raw_rows": int(raw_count),
        "filtered_rows": int(len(df_clean)),
        "missing_customer_ids_removed": int(missing_cust_count),
        "cancellations_removed": int(cancel_count),
        "invalid_qty_price_removed": int(invalid_qty_price),
        "unique_customers": int(df_clean["CustomerID"].nunique()),
        "unique_items": int(df_clean["StockCode"].nunique()),
        "date_range": f"{df_clean['InvoiceDate'].min().strftime('%Y-%m-%d')} to {df_clean['InvoiceDate'].max().strftime('%Y-%m-%d')}",
        "return_cancellation_policy": "Explicitly removed InvoiceNo starting with 'C'; never treated as positive relevance",
        "missing_id_policy": "Explicitly dropped missing CustomerID records with count logged",
        "privacy_usage_note": "Public 2-year transaction benchmark; no direct PII; identifiers treated as non-ordinal keys"
    }
    return card


def run_data_ingestion():
    """Main data ingestion orchestrator for all 4 datasets."""
    print("=" * 80)
    print("MDI3003 Lab 07: Data Ingestion and Integrity Verification")
    print(GUARDRAIL_NOTICE)
    print("=" * 80)

    os.makedirs("results", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    cards = []
    # Ingest D1
    print("\n--- Ingesting D1 (UCI Online Retail) ---")
    d1_card = ingest_d1_uci_online_retail()
    cards.append(d1_card)

    # Ingest D2
    print("\n--- Ingesting D2 (Retailrocket) ---")
    d2_card = ingest_d2_retailrocket()
    cards.append(d2_card)

    # Ingest D3
    print("\n--- Ingesting D3 (Instacart) ---")
    d3_card = ingest_d3_instacart()
    cards.append(d3_card)

    # Ingest D4
    print("\n--- Ingesting D4 (UCI Online Retail II) ---")
    d4_card = ingest_d4_uci_online_retail_ii()
    cards.append(d4_card)

    # Save DATASET_MANIFEST.json
    manifest = {
        "metadata": {
            "title": "MDI3003 Lab 07 Dataset Manifest",
            "student": "Lokanth S (23MID0037)",
            "faculty": "Dr. Durgesh Kumar",
            "timestamp": datetime.now().isoformat(),
            "guardrail": GUARDRAIL_NOTICE
        },
        "datasets": {c["dataset_id"]: c for c in cards}
    }
    manifest_path = "data/DATASET_MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[MANIFEST] Saved dataset manifest to {manifest_path}")

    # Save Table A: Dataset_Card.csv in results/
    df_cards = pd.DataFrame(cards)
    table_a_path = "results/Dataset_Card.csv"
    df_cards.to_csv(table_a_path, index=False)
    print(f"[TABLE A] Saved dataset card table to {table_a_path}")
    print("\nSummary Table A:")
    print(df_cards[["dataset_id", "dataset_name", "raw_rows", "filtered_rows", "unique_customers", "unique_items", "date_range"]].to_string())

    return manifest


if __name__ == "__main__":
    run_data_ingestion()
