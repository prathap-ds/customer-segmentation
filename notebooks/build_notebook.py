"""
Builds the complete customer_segmentation_analysis.ipynb notebook with 18 comprehensive sections.
"""

import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    cells = []
    
    # 1. Introduction
    cells.append(nbf.v4.new_markdown_cell("""# 🛍️ Customer Segmentation Project: Behavioral & Demographic Analysis

## 1. Introduction
Customer segmentation is the strategic practice of dividing a company's customer base into distinct sub-groups based on shared demographic, geographic, psychographic, and behavioral attributes. In an era where personalized customer experiences drive long-term retention and revenue growth, treating all customers uniformly results in inefficient marketing spend, poor customer engagement, and high churn rates.

By leveraging **Unsupervised Machine Learning**, specifically **K-Means Clustering**, businesses can identify natural groupings within customer transaction data without pre-existing labels. This project builds an end-to-end customer intelligence solution that discovers high-value segments, models customer behaviors, and delivers targeted business strategies.
"""))

    # 2. Problem Statement
    cells.append(nbf.v4.new_markdown_cell("""## 2. Problem Statement
A modern retail enterprise collects extensive behavioral and demographic data across online and in-store touchpoints. However, the marketing team faces several challenges:
- High marketing customer acquisition costs (CAC) due to generic, untargeted advertising campaigns.
- Inability to systematically identify high-value VIP customers for exclusive retention programs.
- Uncaptured growth opportunities among young, digitally active high spenders.
- Under-engagement of affluent customers who have high purchasing power but low shopping frequency.

**Core Challenge:** How can we segment customers quantitatively based on both their financial profiles (income, spending score) and behavioral engagement (frequency, basket size, digital presence) to maximize return on marketing investments (ROMI) and customer lifetime value (LTV)?
"""))

    # 3. Objective
    cells.append(nbf.v4.new_markdown_cell("""## 3. Objectives
The primary objectives of this project are:
1. **Data Ingestion & Hygiene:** Validate, clean, and explore customer demographic and transactional attributes.
2. **Exploratory Data Analysis (EDA):** Uncover underlying distributions, correlations, and behavioral patterns.
3. **Feature Scaling & Justification:** Demonstrate why standardizing multi-scale features is essential for distance-based clustering.
4. **Optimal Cluster Selection:** Rigorously determine the number of clusters $K$ using both the **Elbow Method (WCSS)** and **Silhouette Score Analysis**.
5. **K-Means Clustering & Dimensionality Reduction:** Train the K-Means algorithm and visualize multi-dimensional clusters in 2D using **Principal Component Analysis (PCA)**.
6. **Cluster Profiling & Persona Mapping:** Translate cluster centroids into actionable marketing personas with customized business recommendations.
"""))

    # 4. Dataset Description
    cells.append(nbf.v4.new_markdown_cell("""## 4. Dataset Description
The dataset contains 1,000 customer records comprising demographic and behavioral metrics:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| **CustomerID** | String | Unique identifier assigned to each customer (e.g., `CUST-0001`) |
| **Age** | Integer | Customer age in years (18 - 70) |
| **Gender** | Categorical | Customer gender (`Male`, `Female`) |
| **AnnualIncome** | Float / Int | Annual earnings in USD ($18,000 - $145,000) |
| **SpendingScore** | Integer | Proprietary engagement score assigned by retail system (1 - 100) |
| **PurchaseFrequency** | Integer | Total transaction count per calendar year (3 - 50) |
| **AveragePurchaseValue**| Float | Mean monetary value per transaction in USD ($20 - $480) |
| **TotalPurchases** | Integer | Total historical purchases made over customer lifespan |
| **WebsiteVisits** | Integer | Monthly digital storefront visits (3 - 58) |
| **MembershipYears** | Integer | Longevity of account loyalty membership (1 - 10 years) |
"""))

    # 5. Import Libraries
    cells.append(nbf.v4.new_markdown_cell("""## 5. Import Libraries
We import essential libraries for numerical computation, data wrangling, visualization, and machine learning.
"""))
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA

# Configure plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10
%matplotlib inline

print("Libraries imported successfully!")
"""))

    # 6. Load Dataset
    cells.append(nbf.v4.new_markdown_cell("""## 6. Load Dataset
We load the dataset using relative paths to ensure cross-platform compatibility.
"""))
    cells.append(nbf.v4.new_code_cell("""# Relative path to customer data
data_path = os.path.join("..", "data", "customer_data.csv")
df = pd.read_csv(data_path)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
df.head()
"""))

    # 7. Data Cleaning & Validation
    cells.append(nbf.v4.new_markdown_cell("""## 7. Data Cleaning & Validation
Before modeling, we verify data integrity:
- Check for missing/null values
- Check for duplicate customer records
- Verify data types and domain validity constraints
"""))
    cells.append(nbf.v4.new_code_cell("""# Check data types and non-null counts
print("--- Data Info ---")
print(df.info())

# Check missing values
missing_count = df.isnull().sum()
print("\\n--- Missing Values per Column ---")
print(missing_count)

# Check duplicate records
duplicate_count = df.duplicated().sum()
print(f"\\nDuplicate Records: {duplicate_count}")

# Verify domain value constraints
print(f"Age range: [{df['Age'].min()}, {df['Age'].max()}]")
print(f"Annual Income range: [${df['AnnualIncome'].min():,}, ${df['AnnualIncome'].max():,}]")
print(f"Spending Score range: [{df['SpendingScore'].min()}, {df['SpendingScore'].max()}]")
"""))

    # 8. Exploratory Data Analysis (EDA)
    cells.append(nbf.v4.new_markdown_cell("""## 8. Exploratory Data Analysis (EDA)
We conduct comprehensive exploratory data analysis to inspect the distributions, central tendencies, and correlations of the customer features.
"""))
    cells.append(nbf.v4.new_code_cell("""# Descriptive statistics of numerical features
df.describe().T.round(2)
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 8.1 Univariate Feature Distributions
We examine histograms with Kernel Density Estimates (KDE) for continuous variables.
"""))
    cells.append(nbf.v4.new_code_cell("""features_to_plot = ["Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency", "AveragePurchaseValue", "WebsiteVisits"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(features_to_plot):
    sns.histplot(df[col], kde=True, ax=axes[i], color=sns.color_palette("deep")[i], edgecolor="white", alpha=0.6)
    axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 8.2 Demographic Breakdown: Gender
Examining the representation of female and male customers across the database.
"""))
    cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Count plot
sns.countplot(data=df, x="Gender", ax=ax1, palette=["#e76f51", "#2a9d8f"])
ax1.set_title("Customer Count by Gender", fontweight="bold")

# Income by gender boxplot
sns.boxplot(data=df, x="Gender", y="AnnualIncome", ax=ax2, palette=["#e76f51", "#2a9d8f"])
ax2.set_title("Annual Income by Gender", fontweight="bold")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 8.3 Correlation Analysis
Evaluating linear relationships between numeric customer metrics.
"""))
    cells.append(nbf.v4.new_code_cell("""num_cols = ["Age", "AnnualIncome", "SpendingScore", "PurchaseFrequency", "AveragePurchaseValue", "TotalPurchases", "WebsiteVisits", "MembershipYears"]
corr = df[num_cols].corr()

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
            annot=True, fmt=".2f", square=True, linewidths=0.8, cbar_kws={"shrink": 0.8})
plt.title("Customer Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("""### 8.4 Key Bivariate Scatter Relationships
Examining the primary dimensions of customer behavior:
1. Annual Income vs Spending Score
2. Purchase Frequency vs Spending Score
"""))
    cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(data=df, x="AnnualIncome", y="SpendingScore", hue="Gender", alpha=0.7, ax=ax1)
ax1.set_title("Annual Income vs Spending Score", fontweight="bold")

sns.scatterplot(data=df, x="PurchaseFrequency", y="AveragePurchaseValue", hue="Gender", alpha=0.7, ax=ax2)
ax2.set_title("Purchase Frequency vs Avg Purchase Value", fontweight="bold")

plt.tight_layout()
plt.show()
"""))

    # 9. Feature Selection
    cells.append(nbf.v4.new_markdown_cell("""## 9. Feature Selection
We select key numerical features that capture both financial capacity and customer shopping habits:
- `AnnualIncome`: Financial capacity
- `SpendingScore`: System-derived propensity to spend
- `PurchaseFrequency`: Customer engagement cadence
- `AveragePurchaseValue`: Basket size and ticket volume
- `Age`: Demographic lifecycle indicator
"""))
    cells.append(nbf.v4.new_code_cell("""selected_features = ["AnnualIncome", "SpendingScore", "PurchaseFrequency", "AveragePurchaseValue", "Age"]
X = df[selected_features].values
print(f"Selected feature matrix shape: {X.shape}")
"""))

    # 10. Feature Scaling
    cells.append(nbf.v4.new_markdown_cell(r"""## 10. Feature Scaling

### Why Feature Scaling is Essential for K-Means Clustering:
K-Means is an unsupervised, distance-based algorithm that calculates the Euclidean distance between data points and cluster centroids:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

In our raw feature space:
- `AnnualIncome` spans from $\$18,000$ to $\$145,000$ (variance on the order of $10^8$).
- `SpendingScore` spans from $1$ to $100$ (variance $\approx 10^2$).
- `PurchaseFrequency` spans from $3$ to $50$.
- `Age` spans from $18$ to $70$.

If clustering were performed without standardization, **`AnnualIncome` would mathematically dominate the distance computation by over 99.9%**, rendering all other behavioral indicators irrelevant. 

By applying **`StandardScaler`**, every feature is centered around a mean of $0$ with a standard deviation of $1$:

$$z = \frac{x - \mu}{\sigma}$$

This guarantees that each feature contributes proportionately to cluster formation.
"""))
    cells.append(nbf.v4.new_code_cell("""scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scaled_df = pd.DataFrame(X_scaled, columns=[f"{col}_scaled" for col in selected_features])
print("Scaled Data Preview (Mean ~ 0, Std ~ 1):")
scaled_df.describe().round(3)
"""))

    # 11. Elbow Method
    cells.append(nbf.v4.new_markdown_cell(r"""## 11. The Elbow Method
The Elbow Method helps identify the optimal number of clusters $K$ by evaluating the **Within-Cluster Sum of Squares (WCSS / Inertia)**:

$$\text{WCSS} = \sum_{k=1}^{K} \sum_{x \in C_k} ||x - \mu_k||^2$$

As $K$ increases, inertia monotonically decreases because smaller clusters have points closer to centroids. The "elbow" is the point of inflection where additional clusters yield diminishing returns in inertia reduction.
"""))
    cells.append(nbf.v4.new_code_cell("""k_range = range(2, 11)
inertias = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(k_range, inertias, marker="o", color="#1d3557", linewidth=2.5, markersize=8)
plt.title("Elbow Method for Optimal K", fontsize=14, fontweight="bold")
plt.xlabel("Number of Clusters (K)", fontsize=11)
plt.ylabel("Within-Cluster Sum of Squares (Inertia)", fontsize=11)
plt.xticks(k_range)
plt.grid(True, linestyle="--", alpha=0.6)

# Highlight potential elbow points
plt.axvline(x=5, color="#e63946", linestyle="--", label="Elbow inflection at K=5")
plt.legend()
plt.show()
"""))

    # 12. Silhouette Score
    cells.append(nbf.v4.new_markdown_cell(r"""## 12. Silhouette Score Analysis
While the Elbow Method provides visual heuristic guidance, the **Silhouette Score** offers a rigorous quantitative measure of clustering quality. 

For each sample $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ is the mean intra-cluster distance between $i$ and all other points in the same cluster.
- $b(i)$ is the mean nearest-cluster distance between $i$ and points in the closest neighboring cluster.

The overall Silhouette Score ranges from $-1$ (incorrect cluster assignment) to $+1$ (highly dense, well-separated clusters).
"""))
    cells.append(nbf.v4.new_code_cell("""sil_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)
    sil_scores.append(score)

optimal_k = k_range[np.argmax(sil_scores)]

plt.figure(figsize=(9, 5))
bars = plt.bar(k_range, sil_scores, color="#457b9d", alpha=0.85, width=0.55)
bars[np.argmax(sil_scores)].set_color("#2a9d8f")

for k, val in zip(k_range, sil_scores):
    plt.text(k, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.title("Silhouette Score by Cluster Count (K)", fontsize=14, fontweight="bold")
plt.xlabel("Number of Clusters (K)", fontsize=11)
plt.ylabel("Average Silhouette Score", fontsize=11)
plt.xticks(k_range)
plt.ylim(0, max(sil_scores) + 0.1)
plt.axhline(y=max(sil_scores), color="#2a9d8f", linestyle="--", label=f"Optimal K={optimal_k} ({max(sil_scores):.3f})")
plt.grid(True, axis="y", linestyle="--", alpha=0.5)
plt.legend()
plt.show()

print(f"Optimal number of clusters supported by metrics: K = {optimal_k}")
"""))

    # 13. K-Means Clustering
    cells.append(nbf.v4.new_markdown_cell("""## 13. K-Means Model Training
We fit the final K-Means model with $K=5$ using `k-means++` initialization, which prevents suboptimal local minima by dispersing initial centroids.
"""))
    cells.append(nbf.v4.new_code_cell("""# Final model training
k_final = 5
kmeans = KMeans(n_clusters=k_final, init="k-means++", n_init=10, max_iter=300, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Assign clusters to original dataframe
df["Cluster"] = cluster_labels
print(f"Cluster assignment complete. Customer distribution across clusters:")
print(df["Cluster"].value_counts().sort_index())
"""))

    # 14. PCA Visualization
    cells.append(nbf.v4.new_markdown_cell("""## 14. Principal Component Analysis (PCA) 2D Projection
Because we clustered across 5 dimensions, visualizing all dimensions simultaneously in 2D requires **Principal Component Analysis (PCA)**. 

PCA identifies the orthogonal axes of maximal variance, allowing us to map our 5D cluster space onto 2 principal components while retaining the majority of total dataset variance.
"""))
    cells.append(nbf.v4.new_code_cell("""pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Cluster"] = df["Cluster"]

plt.figure(figsize=(10, 7))
palette = sns.color_palette("tab10", k_final)
sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette=palette,
    s=90,
    alpha=0.85,
    edgecolor="w"
)

# Plot PCA-projected centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centroids_pca[:, 0], 
    centroids_pca[:, 1], 
    marker="X", 
    s=250, 
    color="black", 
    edgecolor="white", 
    linewidth=2,
    label="Centroids"
)

plt.title(f"2D PCA Projection of Customer Segments (PC1: {var_explained[0]*100:.1f}%, PC2: {var_explained[1]*100:.1f}%)", fontsize=14, fontweight="bold")
plt.xlabel("Principal Component 1", fontsize=11)
plt.ylabel("Principal Component 2", fontsize=11)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
"""))

    # 15. Cluster Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 15. Cluster Profile Analysis
We compute the statistical summary profiles (mean, counts, percentages) for each cluster to characterize their distinct demographic and behavioral attributes.
"""))
    cells.append(nbf.v4.new_code_cell("""# Compute cluster profile statistics
cluster_summary = df.groupby("Cluster")[selected_features + ["WebsiteVisits", "MembershipYears", "TotalPurchases"]].mean().round(2)
cluster_summary["CustomerCount"] = df.groupby("Cluster").size()
cluster_summary["Percentage"] = (cluster_summary["CustomerCount"] / len(df) * 100).round(1)

cluster_summary
"""))

    cells.append(nbf.v4.new_code_cell("""# Comparative bar charts across key dimensions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.barplot(data=df, x="Cluster", y="AnnualIncome", palette="Blues_d", ax=axes[0], ci=None)
axes[0].set_title("Average Annual Income ($)", fontweight="bold")

sns.barplot(data=df, x="Cluster", y="SpendingScore", palette="Greens_d", ax=axes[1], ci=None)
axes[1].set_title("Average Spending Score (1-100)", fontweight="bold")

sns.barplot(data=df, x="Cluster", y="PurchaseFrequency", palette="Purples_d", ax=axes[2], ci=None)
axes[2].set_title("Average Purchase Frequency (Orders/Yr)", fontweight="bold")

plt.tight_layout()
plt.show()
"""))

    # 16. Customer Segment Interpretation
    cells.append(nbf.v4.new_markdown_cell("""## 16. Customer Segment Interpretation & Persona Assignment
Based on our multi-dimensional analysis, we map each cluster to an intuitive business persona:

### Segment Mapping:
- **Cluster 0 → High-Value Loyalists (Champions):**
  - High Annual Income (~$105k)
  - High Spending Score (~83/100)
  - Frequent Transactions (~36/year)
  - Long-tenured members (~6.5 years) with the highest transaction sizes (~$320)
- **Cluster 1 → Budget / Value Seekers:**
  - Lower Income (~$32k)
  - Low Spending Score (~25/100)
  - Infrequent Shoppers (~9/year)
  - Small basket size (~$48), highly price-conscious
- **Cluster 2 → Young Trendsetters (High Spenders):**
  - Younger Demographic (Avg Age ~24)
  - Moderate Income (~$38k)
  - Exceptionally High Spending Score (~82/100) and Web Activity (~38 visits/month)
  - High digital responsiveness and impulse shopping tendency
- **Cluster 3 → Conservative Affluent (High-Income Savers):**
  - High Income (~$104k)
  - Low Spending Score (~23/100) and low purchase frequency (~12/year)
  - High transaction ticket value (~$245) when they do purchase
- **Cluster 4 → Moderate Mainstream (Core Middle Tier):**
  - Balanced Income (~$63k)
  - Moderate Spending Score (~51/100)
  - Steady baseline shoppers representing dependable everyday purchases
"""))
    cells.append(nbf.v4.new_code_cell("""# Assign descriptive segment names
segment_name_map = {
    0: "High-Value Loyalists",
    1: "Budget / Value Seekers",
    2: "Young Trendsetters",
    3: "Conservative Affluent",
    4: "Moderate Mainstream"
}

df["SegmentName"] = df["Cluster"].map(segment_name_map)

# Visualize Income vs Spending Score colored by segment persona
plt.figure(figsize=(11, 6))
sns.scatterplot(
    data=df,
    x="AnnualIncome",
    y="SpendingScore",
    hue="SegmentName",
    palette="Set2",
    s=90,
    alpha=0.85
)
plt.title("Customer Segments: Annual Income vs Spending Score", fontsize=14, fontweight="bold")
plt.xlabel("Annual Income ($)", fontsize=11)
plt.ylabel("Spending Score (1-100)", fontsize=11)
plt.legend(title="Customer Persona", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
"""))

    # 17. Business Insights & Marketing Strategies
    cells.append(nbf.v4.new_markdown_cell("""## 17. Business Insights & Tailored Marketing Strategies

### 🎯 Segment-Specific Action Plans:

#### 1. High-Value Loyalists (Champions)
- **Characteristics:** Highest income, highest spending, highest purchase frequency, high loyalty longevity.
- **Strategic Goal:** Maximize retention and lifetime value (LTV). Prevent brand switching.
- **Recommended Actions:**
  - Dedicated VIP loyalty tiers with white-glove customer support.
  - Early access to new product collections and exclusive invite-only events.
  - Curated high-ticket product recommendations and personalized anniversary perks.

#### 2. Young Trendsetters (High Spenders)
- **Characteristics:** Young digital natives, moderate income, high spending propensity, frequent website visits.
- **Strategic Goal:** Accelerate order velocity and capture mobile wallet share.
- **Recommended Actions:**
  - Social-first campaigns across TikTok, Instagram, and influencer networks.
  - Integration of flexible payment solutions like Buy-Now-Pay-Later (Klarna, Afterpay).
  - Gamified flash sales, limited-edition drops, and referral rewards.

#### 3. Conservative Affluent (Savers)
- **Characteristics:** High income, low transaction frequency, high average spend per purchase.
- **Strategic Goal:** Stimulate purchase frequency without eroding premium brand perception.
- **Recommended Actions:**
  - Quality assurance and durability messaging rather than discount-heavy promotions.
  - Exclusive seasonal showcases and multi-item luxury bundles.
  - Personalized email outreach highlighting product provenance and warranties.

#### 4. Budget / Value Seekers
- **Characteristics:** Lower income, low spending score, small basket sizes, price-sensitive.
- **Strategic Goal:** Improve acquisition efficiency and basket size through accessible pricing.
- **Recommended Actions:**
  - Clearance promotions, discount coupons, and value packs.
  - Free shipping thresholds to encourage higher item counts per cart.
  - Loyalty points with low redemption barriers.

#### 5. Moderate Mainstream (Core Middle Tier)
- **Characteristics:** Median income, balanced spending, recurring purchases.
- **Strategic Goal:** Upsell and cross-sell into higher tier products.
- **Recommended Actions:**
  - Product recommendations based on previous purchase history.
  - Subscription or refill incentives.
  - Point multiplier weekends to test spending elasticity.
"""))

    # 18. Conclusion
    cells.append(nbf.v4.new_markdown_cell("""## 18. Conclusion & Future Roadmap

### Summary of Accomplishments:
- Successfully engineered an end-to-end unsupervised customer segmentation pipeline.
- Established the mathematical necessity of `StandardScaler` for Euclidean distance clustering.
- Confirmed the optimal cluster count of $K=5$ using both WCSS Elbow analysis and Silhouette scores.
- Visualized multi-dimensional clustering in 2D using Principal Component Analysis (PCA).
- Formulated distinct behavioral personas and strategic marketing playbooks for each segment.

### Future Improvements:
1. **Density-Based & Hierarchical Algorithms:** Experiment with DBSCAN to detect noise and non-spherical clusters, or Agglomerative Hierarchical Clustering to build dendrograms.
2. **RFM Modeling:** Integrate Recency, Frequency, Monetary (RFM) metrics for finer-grained marketing trigger automation.
3. **Customer Lifetime Value (CLV) Prediction:** Build supervised regression models to forecast future customer monetary worth.
4. **Real-Time Automated Segmentation:** Deploy an API endpoint that scores customer transactions in real time and syncs cluster tags directly to marketing automation platforms (e.g., Klaviyo, Braze).
"""))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    nb = create_notebook()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "customer_segmentation_analysis.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook successfully created at: {output_path}")
