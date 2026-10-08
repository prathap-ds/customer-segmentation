"""
Visualization Module for Customer Segmentation Project.
Generates publication-quality charts for EDA, clustering diagnostics,
PCA projections, and cluster comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


# Set aesthetic defaults
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10


def plot_feature_distributions(
    df: pd.DataFrame, 
    columns: List[str], 
    figsize: tuple = (16, 10)
) -> plt.Figure:
    """
    Plots histograms with KDE curves for key customer features.
    """
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    
    colors = sns.color_palette("deep", len(columns))
    
    for i, col in enumerate(columns):
        if col in df.columns:
            sns.histplot(
                df[col], 
                kde=True, 
                ax=axes[i], 
                color=colors[i % len(colors)],
                edgecolor="white",
                alpha=0.65
            )
            axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Count")
            
    # Hide any unused subplots
    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)
        
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None, 
    figsize: tuple = (10, 8)
) -> plt.Figure:
    """
    Plots correlation matrix heatmap for numeric features.
    """
    if columns is None:
        numeric_df = df.select_dtypes(include=[np.number])
    else:
        numeric_df = df[columns]
        
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
        ax=ax
    )
    ax.set_title("Customer Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_elbow_curve(
    inertias: Dict[int, float], 
    optimal_k: Optional[int] = None, 
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plots the Elbow Curve (WCSS vs Number of Clusters K).
    """
    k_vals = list(inertias.keys())
    wcss = list(inertias.values())
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(k_vals, wcss, marker="o", color="#2b5c8f", linewidth=2.5, markersize=8, label="WCSS / Inertia")
    
    if optimal_k and optimal_k in inertias:
        ax.scatter([optimal_k], [inertias[optimal_k]], color="#e63946", s=180, zorder=5, label=f"Selected K={optimal_k}")
        ax.axvline(x=optimal_k, color="#e63946", linestyle="--", alpha=0.7)
        
    ax.set_title("Elbow Method for Optimal K", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=11)
    ax.set_xticks(k_vals)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(frameon=True)
    plt.tight_layout()
    return fig


def plot_silhouette_scores(
    scores: Dict[int, float], 
    optimal_k: Optional[int] = None, 
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plots Silhouette Score vs K.
    """
    k_vals = list(scores.keys())
    silhouette_vals = list(scores.values())
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(k_vals, silhouette_vals, color="#457b9d", alpha=0.85, width=0.55, edgecolor="none")
    
    if optimal_k and optimal_k in scores:
        best_idx = k_vals.index(optimal_k)
        bars[best_idx].set_color("#2a9d8f")
        ax.axhline(y=scores[optimal_k], color="#2a9d8f", linestyle="--", alpha=0.7, label=f"Best Silhouette ({scores[optimal_k]:.3f})")
        
    for k, val in zip(k_vals, silhouette_vals):
        ax.text(k, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_title("Silhouette Score by Cluster Count (K)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of Clusters (K)", fontsize=11)
    ax.set_ylabel("Average Silhouette Score", fontsize=11)
    ax.set_xticks(k_vals)
    ax.set_ylim(0, max(silhouette_vals) + 0.1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    if optimal_k:
        ax.legend(frameon=True)
    plt.tight_layout()
    return fig


def plot_clusters_scatter(
    df: pd.DataFrame, 
    x_col: str, 
    y_col: str, 
    cluster_col: str = "Cluster",
    cluster_names: Optional[Dict[int, str]] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plots 2D scatter plot colored by customer segment cluster.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_df = df.copy()
    if cluster_names:
        plot_df["Segment"] = plot_df[cluster_col].map(lambda c: f"Cluster {c}: {cluster_names.get(c, '')}")
        hue_col = "Segment"
    else:
        plot_df["Segment"] = plot_df[cluster_col].astype(str)
        hue_col = "Segment"
        
    palette = sns.color_palette("tab10", plot_df[cluster_col].nunique())
    
    sns.scatterplot(
        data=plot_df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette=palette,
        s=80,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        ax=ax
    )
    
    ax.set_title(f"Customer Segments: {x_col} vs {y_col}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.legend(title="Customer Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_pca_scatter(
    pca_df: pd.DataFrame, 
    cluster_col: str = "Cluster",
    explained_variance: Optional[List[float]] = None,
    cluster_names: Optional[Dict[int, str]] = None,
    figsize: tuple = (10, 6)
) -> plt.Figure:
    """
    Plots PCA 2D projection of customer segments.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    plot_df = pca_df.copy()
    if cluster_names:
        plot_df["Segment"] = plot_df[cluster_col].map(lambda c: f"Cluster {c}: {cluster_names.get(c, '')}")
        hue_col = "Segment"
    else:
        plot_df["Segment"] = plot_df[cluster_col].astype(str)
        hue_col = "Segment"
        
    palette = sns.color_palette("Set2", plot_df[cluster_col].nunique())
    
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue=hue_col,
        palette=palette,
        s=85,
        alpha=0.85,
        edgecolor="w",
        ax=ax
    )
    
    var_label = ""
    if explained_variance and len(explained_variance) >= 2:
        var_label = f" (PC1: {explained_variance[0]*100:.1f}%, PC2: {explained_variance[1]*100:.1f}%)"
        
    ax.set_title(f"2D PCA Customer Cluster Map{var_label}", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Principal Component 1 (PC1)", fontsize=11)
    ax.set_ylabel("Principal Component 2 (PC2)", fontsize=11)
    ax.legend(title="Customer Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_cluster_sizes(
    df: pd.DataFrame, 
    cluster_col: str = "Cluster",
    cluster_names: Optional[Dict[int, str]] = None,
    figsize: tuple = (8, 5)
) -> plt.Figure:
    """
    Plots cluster size distribution bar chart.
    """
    counts = df[cluster_col].value_counts().sort_index()
    labels = [f"Cluster {c}" if not cluster_names else f"C{c}: {cluster_names.get(c, '')}" for c in counts.index]
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(labels, counts.values, color=sns.color_palette("pastel", len(counts)), edgecolor="gray")
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{int(h)} ({h/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    ax.set_title("Customer Count by Segment", fontsize=14, fontweight="bold")
    ax.set_xlabel("Cluster Segment", fontsize=11)
    ax.set_ylabel("Number of Customers", fontsize=11)
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    return fig
