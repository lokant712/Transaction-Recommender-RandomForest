"""
Interactive Streamlit Dashboard for MDI3003 Lab 07 Recommender System.
Author: Lokanth S (23MID0037)
Faculty: Dr. Durgesh Kumar
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Ensure workspace root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(
    page_title="MDI3003 Lab 07: Recommender System Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# Custom CSS for modern premium UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f6f8fb 0%, #edf2f7 100%);
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #2B6CB0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .guardrail-box {
        background-color: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        padding: 1rem;
        color: #9b2c2c;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data_and_models():
    clean_path = "data/D1_uci_online_retail/online_retail_cleaned.csv"
    if not os.path.exists(clean_path):
        return None, None, None, None, None
    df = pd.read_csv(clean_path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Load artifacts
    with open("artifacts/split_manifest.json", "r") as f:
        split_manifest = json.load(f)
    with open("artifacts/feature_schema.json", "r") as f:
        schema = json.load(f)
    with open("artifacts/candidate_policy.json", "r") as f:
        cand_policy = json.load(f)

    model = joblib.load("models/random_forest.joblib")
    return df, split_manifest, schema, cand_policy, model


def main():
    st.markdown('<div class="main-title">🛍️ Customer Recommendation System Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">MDI3003 Advanced Predictive Analytics — Experiment 07 | <b>Lokanth S (23MID0037)</b> | Faculty: <b>Dr. Durgesh Kumar</b></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="guardrail-box">
        <b>🛡️ Responsible Recommendation Guardrail:</b> Recommendations represent estimated purchase relevance, not proof of intent. 
        This system prohibits predatory personalization, discriminatory pricing, and exposure of private individual purchase histories.
    </div>
    """, unsafe_allow_html=True)

    df, split_manifest, schema, cand_policy, model = load_data_and_models()
    if df is None:
        st.error("Data or models not found. Please run `python scripts/run_all.py` first.")
        return

    t2_ts = pd.Timestamp(split_manifest["cutoff_t2"])
    dev_hist = df[df["InvoiceDate"] < t2_ts]
    test_future = df[df["InvoiceDate"] >= t2_ts]

    item_desc = df.drop_duplicates("StockCode").set_index("StockCode")["Description"].to_dict()

    # Sidebar
    st.sidebar.title("Navigation & Parameters")
    tab_selection = st.sidebar.radio(
        "Select Dashboard View:",
        ["👤 Customer Recommender", "📊 Main Ranking Comparison", "🔍 Five-Case Audit", "🧪 Sensitivity & Ablations", "📈 Catalog Coverage & Bias"]
    )

    top_k = st.sidebar.slider("Top-K Recommendation Cutoff", min_value=3, max_value=20, value=10)

    # 1. Customer Recommender View
    if tab_selection == "👤 Customer Recommender":
        st.subheader("Interactive Customer Recommendation Inference")
        
        all_test_users = test_future["CustomerID"].unique().tolist()
        sample_users = [u for u in all_test_users if u in set(dev_hist["CustomerID"].unique())][:50]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_user = st.selectbox("Select Customer ID:", sample_users, index=0)
            st.info(f"Target Evaluation Window: **{t2_ts.strftime('%Y-%m-%d')}** onwards")

        # History summary
        u_hist = dev_hist[dev_hist["CustomerID"] == str(selected_user)]
        u_test = test_future[test_future["CustomerID"] == str(selected_user)]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Historical Invoices", u_hist["InvoiceNo"].nunique())
        c2.metric("Historical Items", u_hist["StockCode"].nunique())
        c3.metric("Historical Spend", f"£{u_hist['Amount'].sum():.2f}")
        c4.metric("Future Test Items", u_test["StockCode"].nunique())

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 📜 Historical Purchases (H < t2)")
            if not u_hist.empty:
                hist_summary = u_hist.groupby("StockCode").agg(
                    Description=("Description", "first"),
                    Orders=("InvoiceNo", "nunique"),
                    TotalQty=("Quantity", "sum"),
                    TotalSpend=("Amount", "sum")
                ).reset_index().sort_values("Orders", ascending=False).head(8)
                st.dataframe(hist_summary, use_container_width=True)
            else:
                st.write("No prior purchase history (Cold-Start Customer).")

        with col_right:
            st.markdown("#### 🎯 True Future Purchases (Locked Test)")
            if not u_test.empty:
                test_summary = u_test.groupby("StockCode").agg(
                    Description=("Description", "first"),
                    TotalQty=("Quantity", "sum")
                ).reset_index()
                st.dataframe(test_summary, use_container_width=True)
            else:
                st.write("No future purchases in test window.")

        # Real-time inference
        from inference import run_inference
        st.markdown(f"#### 🌟 Live Top-{top_k} Recommendations (Random Forest)")
        recs = run_inference(selected_user, top_k=top_k)
        
        rec_df = pd.DataFrame(recs)
        rec_df["Actual Future Hit?"] = rec_df["stock_code"].apply(lambda s: "✅ HIT" if s in set(u_test["StockCode"]) else "❌")
        st.dataframe(rec_df[["rank", "stock_code", "score", "is_repeat", "Actual Future Hit?", "description"]], use_container_width=True)

    # 2. Main Comparison
    elif tab_selection == "📊 Main Ranking Comparison":
        st.subheader("Benchmark Ranking Performance on Locked Test Set (Table B)")
        if os.path.exists("results/Main_Ranking_Comparison.csv"):
            df_b = pd.read_csv("results/Main_Ranking_Comparison.csv")
            st.dataframe(df_b.style.highlight_max(axis=0, color="#d4edda"), use_container_width=True)

        st.markdown("### Visual Benchmark Comparison")
        if os.path.exists("figures/Fig08_popularity_vs_rf_ranking.png"):
            st.image("figures/Fig08_popularity_vs_rf_ranking.png", caption="Fig 8: Popularity vs Random Forest Ranking Metrics")
        if os.path.exists("figures/Fig07_precision_recall_vs_k.png"):
            st.image("figures/Fig07_precision_recall_vs_k.png", caption="Fig 7: Precision@K & Recall@K vs K")

    # 3. Five Case Audit
    elif tab_selection == "🔍 Five-Case Audit":
        st.subheader("Five-Case Recommendation Forensic Audit (Table D)")
        if os.path.exists("results/Five_Case_Audit.csv"):
            df_d = pd.read_csv("results/Five_Case_Audit.csv")
            st.dataframe(df_d, use_container_width=True)

    # 4. Sensitivity & Ablations
    elif tab_selection == "🧪 Sensitivity & Ablations":
        st.subheader("Feature Ablation (Table C) & Negative Sampling Sensitivity (E5)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Feature Ablation Study (E4)")
            if os.path.exists("results/Feature_Ablation.csv"):
                st.dataframe(pd.read_csv("results/Feature_Ablation.csv"), use_container_width=True)
        with col2:
            st.markdown("#### Negative Sampling Sensitivity (E5)")
            if os.path.exists("results/Negative_Sampling_Sensitivity.csv"):
                st.dataframe(pd.read_csv("results/Negative_Sampling_Sensitivity.csv"), use_container_width=True)

        st.markdown("#### Cross-Dataset Replication Benchmark (E9)")
        if os.path.exists("results/Cross_Dataset_Replication.csv"):
            st.dataframe(pd.read_csv("results/Cross_Dataset_Replication.csv"), use_container_width=True)

    # 5. Catalog Coverage & Bias
    elif tab_selection == "📈 Catalog Coverage & Bias":
        st.subheader("Catalog Coverage and Popularity Concentration (Figure 10)")
        if os.path.exists("figures/Fig10_catalog_coverage_popularity_bias.png"):
            st.image("figures/Fig10_catalog_coverage_popularity_bias.png", caption="Fig 10: Recommendation Lorenz Curves")
        if os.path.exists("results/Efficiency_Complexity_Comparison.csv"):
            st.markdown("#### System Efficiency & Model Complexity (Table E)")
            st.dataframe(pd.read_csv("results/Efficiency_Complexity_Comparison.csv"), use_container_width=True)


if __name__ == "__main__":
    main()
