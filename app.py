"""
Customer Segmentation Analytics Dashboard
Streamlit Web Application
"""

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from data_preprocessing import load_data, inspect_data, clean_data, scale_features
from clustering import (
    calculate_elbow,
    calculate_silhouette,
    find_optimal_k,
    train_kmeans,
    compute_pca,
    compute_cluster_profiles,
    interpret_clusters
)
from visualization import (
    plot_feature_distributions,
    plot_correlation_heatmap,
    plot_elbow_curve,
    plot_silhouette_scores,
    plot_clusters_scatter,
    plot_pca_scatter,
    plot_cluster_sizes
)

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Analytics",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .segment-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
</style>
""", unsafe_allow_html=True)


# --- SIDEBAR CONTROLS ---
st.sidebar.image("https://img.icons8.com/fluency/96/group.png", width=64)
st.sidebar.title("Configuration")

# 1. Dataset Upload
st.sidebar.subheader("1. Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Customer CSV", type=["csv"])

# Default dataset path
default_data_path = os.path.join(CURRENT_DIR, "data", "customer_data.csv")

@st.cache_data
def get_dataset(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    elif os.path.exists(default_data_path):
        return pd.read_csv(default_data_path)
    else:
        st.error("Default dataset not found. Please upload a CSV file.")
        st.stop()

df_raw = get_dataset(uploaded_file)
cleaned_df = clean_data(df_raw)

# 2. Feature Selection
st.sidebar.subheader("2. Features for Clustering")
numeric_columns = list(cleaned_df.select_dtypes(include=[np.number]).columns)
# Exclude identifier columns if present
candidate_features = [col for col in numeric_columns if col.lower() not in ["customerid", "id", "cluster"]]

default_selected = [
    f for f in ["AnnualIncome", "SpendingScore", "PurchaseFrequency", "AveragePurchaseValue", "Age"]
    if f in candidate_features
]
if not default_selected:
    default_selected = candidate_features[:4]

selected_features = st.sidebar.multiselect(
    "Select Clustering Features:",
    options=candidate_features,
    default=default_selected
)

if len(selected_features) < 2:
    st.sidebar.warning("Please select at least 2 features for clustering.")
    st.stop()

# 3. Cluster Count Selection
st.sidebar.subheader("3. Model Parameters")
cluster_count = st.sidebar.slider(
    "Number of Clusters (K):",
    min_value=2,
    max_value=8,
    value=5,
    help="Select the number of customer segments to identify"
)

run_clustering_btn = st.sidebar.button("Run Segmentation", type="primary")

# Sidebar Info
st.sidebar.markdown("---")
st.sidebar.caption("🚀 **Antigravity Machine Learning Suite**\nK-Means Clustering & Customer Intelligence")


# --- MAIN DASHBOARD HEADER ---
st.markdown('<div class="main-header">Customer Segmentation Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Unsupervised Machine Learning & Behavioral Customer Segmentation using K-Means</div>', 
    unsafe_allow_html=True
)


# --- SECTION 1: DATASET OVERVIEW ---
st.subheader("1. Dataset Overview")

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Customers", f"{len(cleaned_df):,}")
with col2:
    avg_age = cleaned_df["Age"].mean() if "Age" in cleaned_df.columns else 0
    st.metric("Average Age", f"{avg_age:.1f} yrs")
with col3:
    avg_income = cleaned_df["AnnualIncome"].mean() if "AnnualIncome" in cleaned_df.columns else 0
    st.metric("Avg Annual Income", f"${avg_income:,.0f}")
with col4:
    avg_spending = cleaned_df["SpendingScore"].mean() if "SpendingScore" in cleaned_df.columns else 0
    st.metric("Avg Spending Score", f"{avg_spending:.1f} / 100")
with col5:
    avg_freq = cleaned_df["PurchaseFrequency"].mean() if "PurchaseFrequency" in cleaned_df.columns else 0
    st.metric("Avg Purchase Freq", f"{avg_freq:.1f} / yr")

with st.expander("🔍 View Raw Customer Data Preview & Health Summary", expanded=False):
    tab_data, tab_summary = st.tabs(["Dataset Sample", "Descriptive Statistics"])
    with tab_data:
        st.dataframe(cleaned_df.head(10), use_container_width=True)
    with tab_summary:
        st.dataframe(cleaned_df.describe().round(2), use_container_width=True)


# --- PERFORM CLUSTERING ---
scaled_array, scaler, scaled_df = scale_features(cleaned_df, selected_features)
kmeans_model, cluster_labels = train_kmeans(scaled_array, n_clusters=cluster_count, random_state=42)

# Attach cluster labels to dataset
clustered_df = cleaned_df.copy()
clustered_df["Cluster"] = cluster_labels

# Profiles & Interpretation
profiles_df = compute_cluster_profiles(clustered_df, cluster_col="Cluster")
interpretations = interpret_clusters(profiles_df)
cluster_name_map = {cid: info["name"] for cid, info in interpretations.items()}
clustered_df["SegmentName"] = clustered_df["Cluster"].map(cluster_name_map)


# --- SECTION 2: CUSTOMER SEGMENTS ---
st.markdown("---")
st.subheader("2. Customer Segments")

col_pca, col_dist = st.columns([3, 2])

with col_pca:
    # 2D PCA or 2-Feature Scatter
    pca_coords, variance_ratio, _ = compute_pca(scaled_array, n_components=2)
    pca_df = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
    pca_df["Cluster"] = cluster_labels
    
    fig_pca = plot_pca_scatter(
        pca_df, 
        cluster_col="Cluster", 
        explained_variance=variance_ratio,
        cluster_names=cluster_name_map,
        figsize=(8, 5)
    )
    st.pyplot(fig_pca)
    plt.close(fig_pca)

with col_dist:
    fig_sizes = plot_cluster_sizes(
        clustered_df, 
        cluster_col="Cluster", 
        cluster_names=cluster_name_map,
        figsize=(7, 5)
    )
    st.pyplot(fig_sizes)
    plt.close(fig_sizes)


# --- SECTION 3: CLUSTER ANALYSIS ---
st.markdown("---")
st.subheader("3. Cluster Profile Analysis")

# Cluster Summary Table
summary_display = profiles_df.copy()
summary_display.index = [f"Cluster {idx}: {cluster_name_map.get(idx, '')}" for idx in summary_display.index]
st.dataframe(summary_display, use_container_width=True)

# Comparison Bar Charts
chart_col1, chart_col2, chart_col3 = st.columns(3)

with chart_col1:
    fig_inc, ax_inc = plt.subplots(figsize=(5, 3.8))
    sns.barplot(
        data=clustered_df, 
        x="Cluster", 
        y="AnnualIncome", 
        palette="Blues_d", 
        ci=None, 
        ax=ax_inc
    )
    ax_inc.set_title("Average Income by Cluster", fontweight="bold", fontsize=11)
    ax_inc.set_xlabel("Cluster")
    ax_inc.set_ylabel("Annual Income ($)")
    plt.tight_layout()
    st.pyplot(fig_inc)
    plt.close(fig_inc)

with chart_col2:
    fig_spd, ax_spd = plt.subplots(figsize=(5, 3.8))
    sns.barplot(
        data=clustered_df, 
        x="Cluster", 
        y="SpendingScore", 
        palette="Greens_d", 
        ci=None, 
        ax=ax_spd
    )
    ax_spd.set_title("Average Spending Score by Cluster", fontweight="bold", fontsize=11)
    ax_spd.set_xlabel("Cluster")
    ax_spd.set_ylabel("Spending Score (1-100)")
    plt.tight_layout()
    st.pyplot(fig_spd)
    plt.close(fig_spd)

with chart_col3:
    fig_frq, ax_frq = plt.subplots(figsize=(5, 3.8))
    sns.barplot(
        data=clustered_df, 
        x="Cluster", 
        y="PurchaseFrequency", 
        palette="Purples_d", 
        ci=None, 
        ax=ax_frq
    )
    ax_frq.set_title("Purchase Frequency by Cluster", fontweight="bold", fontsize=11)
    ax_frq.set_xlabel("Cluster")
    ax_frq.set_ylabel("Orders / Year")
    plt.tight_layout()
    st.pyplot(fig_frq)
    plt.close(fig_frq)


# --- SECTION 4: CUSTOMER BEHAVIOR ---
st.markdown("---")
st.subheader("4. Customer Behavioral Relationships")

tab_bev1, tab_bev2, tab_bev3, tab_diagnostics = st.tabs([
    "Income vs Spending Score", 
    "Purchase Frequency vs Spending Score", 
    "Age vs Spending Score",
    "Model Diagnostics (Elbow & Silhouette)"
])

with tab_bev1:
    fig_scatter1 = plot_clusters_scatter(
        clustered_df, 
        x_col="AnnualIncome", 
        y_col="SpendingScore", 
        cluster_col="Cluster", 
        cluster_names=cluster_name_map,
        figsize=(10, 5)
    )
    st.pyplot(fig_scatter1)
    plt.close(fig_scatter1)

with tab_bev2:
    fig_scatter2 = plot_clusters_scatter(
        clustered_df, 
        x_col="PurchaseFrequency", 
        y_col="SpendingScore", 
        cluster_col="Cluster", 
        cluster_names=cluster_name_map,
        figsize=(10, 5)
    )
    st.pyplot(fig_scatter2)
    plt.close(fig_scatter2)

with tab_bev3:
    fig_scatter3 = plot_clusters_scatter(
        clustered_df, 
        x_col="Age", 
        y_col="SpendingScore", 
        cluster_col="Cluster", 
        cluster_names=cluster_name_map,
        figsize=(10, 5)
    )
    st.pyplot(fig_scatter3)
    plt.close(fig_scatter3)

with tab_diagnostics:
    col_elb, col_sil = st.columns(2)
    with col_elb:
        inertias = calculate_elbow(scaled_array, k_range=range(2, 9))
        fig_elb = plot_elbow_curve(inertias, optimal_k=cluster_count, figsize=(7, 4.5))
        st.pyplot(fig_elb)
        plt.close(fig_elb)
    with col_sil:
        sil_scores = calculate_silhouette(scaled_array, k_range=range(2, 9))
        fig_sil = plot_silhouette_scores(sil_scores, optimal_k=cluster_count, figsize=(7, 4.5))
        st.pyplot(fig_sil)
        plt.close(fig_sil)


# --- SECTION 5: BUSINESS INSIGHTS ---
st.markdown("---")
st.subheader("5. Automatically Generated Segment Insights")

for cid, info in interpretations.items():
    with st.expander(f"📍 **Cluster {cid}: {info['name']}** ({info['percentage']}% of customers)", expanded=True):
        c_left, c_right = st.columns([2, 1])
        with c_left:
            st.markdown(f"**Customer Profile:** {info['summary']}")
            st.markdown(f"**Classification:** `{info['tier']}`")
        with c_right:
            st.write(f"- **Typical Age:** {info['avg_age']:.0f} years")
            st.write(f"- **Income Level:** ${info['avg_income']:,.0f}")
            st.write(f"- **Spending Score:** {info['avg_spending']:.1f}/100")
            st.write(f"- **Purchase Frequency:** {info['avg_frequency']:.1f} orders/yr")
            st.write(f"- **Avg Ticket Size:** ${info['avg_purchase_val']:.2f}")
            st.write(f"- **Web Visits:** {info['avg_visits']:.1f}/month")
            st.write(f"- **Membership Duration:** {info['avg_membership']:.1f} years")


# --- SECTION 6: MARKETING RECOMMENDATIONS ---
st.markdown("---")
st.subheader("6. Targeted Marketing Strategies & Business Recommendations")

rec_cols = st.columns(min(3, len(interpretations)))
for i, (cid, info) in enumerate(interpretations.items()):
    col_target = rec_cols[i % len(rec_cols)]
    with col_target:
        st.markdown(f"""
        <div class="segment-card">
            <h4 style="color:#1E3A8A; margin-top:0;">Cluster {cid}: {info['name']}</h4>
            <p><strong>Strategic Objective:</strong> {info['tier']}</p>
            <p style="font-size:0.95rem; color:#374151;"><strong>Recommended Action:</strong> {info['marketing_strategy']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
### 📌 Overall Business Action Plan
1. **Focus on High-Value Retention:** High-Value Loyalists generate over 50% of operating profits. Protect this segment with dedicated account managers and premium loyalty tier benefits.
2. **Nurture High-Income Savers into Active Buyers:** Conservative Affluent customers have deep wallets but low purchase frequency. Build trust through quality-centric campaigns and curated collections.
3. **Engage Young Trendsetters via Digital Omnichannel:** Young Trendsetters have high digital velocity. Leverage social commerce, TikTok/Instagram marketing, and seamless mobile checkout.
4. **Automate Personalized Remarketing:** Integrate this segmentation engine with CRM tools (Salesforce, HubSpot, Klaviyo) to trigger tailored email and SMS workflows based on cluster assignment.
""")
