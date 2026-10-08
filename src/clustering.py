"""
Clustering and Modeling Module for Customer Segmentation Project.
Implements K-Means clustering, Elbow Method, Silhouette Analysis,
PCA dimensionality reduction, and automated business interpretation.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Any


def calculate_elbow(
    scaled_data: np.ndarray, 
    k_range: range = range(2, 11),
    random_state: int = 42
) -> Dict[int, float]:
    """
    Computes Within-Cluster Sum of Squares (WCSS / Inertia) across a range of K.
    Used for the Elbow Method to find where adding more clusters yields diminishing returns.
    
    Args:
        scaled_data: Standardized feature matrix.
        k_range: Range of K values to evaluate.
        random_state: Reproducibility seed.
        
    Returns:
        Dict[int, float]: Mapping of K to WCSS (inertia).
    """
    inertias = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        kmeans.fit(scaled_data)
        inertias[k] = float(kmeans.inertia_)
    return inertias


def calculate_silhouette(
    scaled_data: np.ndarray, 
    k_range: range = range(2, 11),
    random_state: int = 42
) -> Dict[int, float]:
    """
    Computes Silhouette Scores across a range of K.
    Silhouette score s(i) = (b(i) - a(i)) / max(a(i), b(i))
    ranges from -1 (poor grouping) to +1 (dense, well-separated clusters).
    
    Args:
        scaled_data: Standardized feature matrix.
        k_range: Range of K values to evaluate.
        random_state: Reproducibility seed.
        
    Returns:
        Dict[int, float]: Mapping of K to average silhouette score.
    """
    scores = {}
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        cluster_labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, cluster_labels)
        scores[k] = float(score)
    return scores


def find_optimal_k(silhouette_scores: Dict[int, float]) -> int:
    """
    Identifies the optimal K that maximizes the silhouette score.
    
    Args:
        silhouette_scores: Dictionary of K to silhouette scores.
        
    Returns:
        int: Optimal K value.
    """
    return max(silhouette_scores, key=silhouette_scores.get)


def train_kmeans(
    scaled_data: np.ndarray, 
    n_clusters: int, 
    random_state: int = 42
) -> Tuple[KMeans, np.ndarray]:
    """
    Trains a K-Means model with k-means++ initialization on scaled data.
    
    Args:
        scaled_data: Standardized feature matrix.
        n_clusters: Number of clusters (K).
        random_state: Reproducibility seed.
        
    Returns:
        Tuple:
            - kmeans (KMeans): Fitted scikit-learn model
            - labels (np.ndarray): Cluster assignments (0 to K-1)
    """
    kmeans = KMeans(
        n_clusters=n_clusters, 
        init="k-means++", 
        n_init=10, 
        max_iter=300, 
        random_state=random_state
    )
    labels = kmeans.fit_predict(scaled_data)
    return kmeans, labels


def compute_pca(
    scaled_data: np.ndarray, 
    n_components: int = 2, 
    random_state: int = 42
) -> Tuple[np.ndarray, List[float], PCA]:
    """
    Applies Principal Component Analysis (PCA) to project high-dimensional data
    into 2D space for intuitive visualization.
    
    Args:
        scaled_data: Standardized feature matrix.
        n_components: Number of principal components (default: 2).
        random_state: Reproducibility seed.
        
    Returns:
        Tuple:
            - pca_transformed (np.ndarray): 2D coordinates
            - explained_variance (List[float]): Percentage of variance explained by each component
            - pca_model (PCA): Fitted PCA model
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_transformed = pca.fit_transform(scaled_data)
    explained_variance = [float(var) for var in pca.explained_variance_ratio_]
    return pca_transformed, explained_variance, pca


def compute_cluster_profiles(
    df: pd.DataFrame, 
    cluster_col: str = "Cluster"
) -> pd.DataFrame:
    """
    Computes statistical profiles (mean, median, count, percentage) for each cluster.
    
    Args:
        df: DataFrame containing original customer data and assigned clusters.
        cluster_col: Column name representing cluster labels.
        
    Returns:
        pd.DataFrame: Aggregated profile metrics grouped by cluster.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(cluster_col, errors="ignore")
    
    profiles = df.groupby(cluster_col)[numeric_cols].mean().round(2)
    profiles["CustomerCount"] = df.groupby(cluster_col).size()
    profiles["Percentage"] = (profiles["CustomerCount"] / len(df) * 100).round(1)
    
    return profiles


def interpret_clusters(profiles_df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Automatically analyzes cluster statistics and assigns meaningful business names,
    behavioral summaries, and tailored marketing strategies.
    
    Dynamically identifies attributes relative to overall dataset benchmarks.
    
    Args:
        profiles_df: Summary profile table produced by compute_cluster_profiles.
        
    Returns:
        Dict[int, Dict[str, Any]]: Comprehensive interpretation and strategy per cluster.
    """
    interpretations = {}
    
    overall_mean_income = profiles_df["AnnualIncome"].mean() if "AnnualIncome" in profiles_df.columns else 60000
    overall_mean_spending = profiles_df["SpendingScore"].mean() if "SpendingScore" in profiles_df.columns else 50
    overall_mean_freq = profiles_df["PurchaseFrequency"].mean() if "PurchaseFrequency" in profiles_df.columns else 20
    
    for cluster_id, row in profiles_df.iterrows():
        income = row.get("AnnualIncome", 0)
        spending = row.get("SpendingScore", 0)
        freq = row.get("PurchaseFrequency", 0)
        age = row.get("Age", 0)
        avg_val = row.get("AveragePurchaseValue", 0)
        visits = row.get("WebsiteVisits", 0)
        membership = row.get("MembershipYears", 0)
        count = int(row.get("CustomerCount", 0))
        pct = row.get("Percentage", 0)
        
        # Determine archetype dynamically from multi-dimensional characteristics
        if income >= overall_mean_income and spending >= overall_mean_spending:
            name = "High-Value Loyalists (Champions)"
            tier = "Tier 1: Premium"
            summary = (
                f"Affluent customers (Avg Income: ${income:,.0f}) with outstanding spending engagement "
                f"(Score: {spending:.1f}/100) and high purchase frequency ({freq:.1f} orders/year). "
                f"Long-term members (Avg: {membership:.1f} years) who generate the highest revenue per transaction."
            )
            strategy = (
                "VIP loyalty privileges, early access to new product drops, personalized concierge service, "
                "and premium product bundling to maximize lifetime value."
            )
        elif income < overall_mean_income and spending >= overall_mean_spending:
            name = "Young Trendsetters (High Spenders)"
            tier = "Tier 2: Growth Potential"
            summary = (
                f"Younger demographics (Avg Age: {age:.0f}) with moderate income (${income:,.0f}) "
                f"yet exceptionally high spending score ({spending:.1f}/100) and high digital activity "
                f"({visits:.1f} web visits/month). Highly trend-driven."
            )
            strategy = (
                "Social media campaigns, influencer collaborations, flash sales, gamified rewards, "
                "and Buy-Now-Pay-Later (BNPL) flexible payment options."
            )
        elif income >= overall_mean_income and spending < overall_mean_spending:
            name = "Conservative Affluent (High-Income Savers)"
            tier = "Tier 3: High Opportunity"
            summary = (
                f"Mature customers (Avg Age: {age:.0f}) possessing high purchasing power (${income:,.0f}) "
                f"but low spending frequency (Score: {spending:.1f}/100, {freq:.1f} orders/year). "
                f"However, average ticket size is high (${avg_val:.2f}) when they do transact."
            )
            strategy = (
                "Value-focused prestige marketing, warranties, premium quality assurance campaigns, "
                "exclusive seasonal invitations, and personalized email outreach highlighting durability."
            )
        elif income < overall_mean_income and spending < overall_mean_spending:
            name = "Budget / Value Seekers"
            tier = "Tier 4: Price-Sensitive"
            summary = (
                f"Cost-conscious customers with lower annual income (${income:,.0f}) and conservative "
                f"spending patterns (Score: {spending:.1f}/100). Low purchase frequency ({freq:.1f} orders/year) "
                f"and low basket value (${avg_val:.2f})."
            )
            strategy = (
                "Clearance promotions, discount coupons, budget bundles, free-shipping thresholds, "
                "and cost-effective essential product recommendations."
            )
        else:
            name = "Moderate Mainstream (Core Customers)"
            tier = "Tier 3: Steady Baseline"
            summary = (
                f"Balanced mid-tier customers (Avg Income: ${income:,.0f}, Avg Spending Score: {spending:.1f}/100) "
                f"representing reliable recurring revenue across everyday categories."
            )
            strategy = (
                "Cross-selling, loyalty point multipliers, seasonal newsletters, and referral incentives "
                "to encourage upsell into higher spending brackets."
            )
            
        interpretations[int(cluster_id)] = {
            "name": name,
            "tier": tier,
            "count": count,
            "percentage": pct,
            "avg_age": age,
            "avg_income": income,
            "avg_spending": spending,
            "avg_frequency": freq,
            "avg_purchase_val": avg_val,
            "avg_visits": visits,
            "avg_membership": membership,
            "summary": summary,
            "marketing_strategy": strategy
        }
        
    return interpretations
