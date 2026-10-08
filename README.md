# Customer Segmentation Project

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Unsupervised%20ML-orange.svg)](https://scikit-learn.org/)

An end-to-end Machine Learning and Business Analytics project that segments retail customers based on demographic profiles and behavioral shopping patterns using **K-Means Clustering**, **StandardScaler**, **Silhouette Analysis**, and **Principal Component Analysis (PCA)**. Includes an interactive **Streamlit Analytics Dashboard** and a documented **Jupyter Notebook**.

---

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Clustering Method](#clustering-method)
- [Customer Segments](#customer-segments)
- [Business Insights](#business-insights)
- [Dashboard](#dashboard)
- [Installation](#installation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Improvements](#future-improvements)
- [License](#license)

---

## Overview
Customer segmentation is the process of partitioning an enterprise customer base into mutually exclusive sub-cohorts that share common purchasing habits, financial capacities, and engagement levels. This project develops a quantitative clustering pipeline that groups 1,000 retail customers, justifies the choice of $K$ with mathematical diagnostics, visualizes multidimensional data in 2D via PCA, and translates cluster centroids into strategic marketing playbooks.

---

## Problem Statement
Traditional "one-size-fits-all" marketing strategies suffer from diminishing returns:
- **High Customer Acquisition & Retention Costs:** Marketing budgets are wasted on generic broadcast campaigns.
- **Under-engagement of Affluent Customers:** High-income consumers with low purchase frequency are overlooked without luxury-tier nurturing.
- **Untapped Growth in Younger Demographics:** Impulsive, trend-conscious younger shoppers churn when not engaged across digital channels.
- **Lack of Quantitative Personas:** Retail strategies often rely on qualitative guesswork rather than transaction-driven behavioral data.

---

## Objectives
1. **Clean & Validate Customer Data:** Ingest and verify data integrity across demographic and financial indicators.
2. **Perform Exploratory Data Analysis (EDA):** Identify correlation patterns, skewness, and multi-modal distributions.
3. **Standardize Multi-Scale Features:** Apply `StandardScaler` to prevent high-variance attributes from overpowering distance calculations.
4. **Determine Optimal Clusters Mathematically:** Justify the cluster count using the **Elbow Method (WCSS)** and **Silhouette Score Analysis**.
5. **Dimensionality Reduction:** Project high-dimensional feature spaces into 2D using **Principal Component Analysis (PCA)** for visualization.
6. **Synthesize Business Personas & Recommendations:** Deliver actionable marketing tactics for each discovered segment.
7. **Deploy Interactive Dashboard:** Provide business stakeholders with a real-time Streamlit dashboard.

---

## Features
- **Interactive Web Dashboard:** Upload custom CSV files, select clustering features, adjust cluster counts dynamically, and view real-time diagnostics.
- **Mathematically Justified K Selection:** Side-by-side Elbow curve and Silhouette analysis charts.
- **2D PCA Projection:** Interactive scatter maps rendering multi-dimensional clusters and centroid locations.
- **Automated Persona Generator:** Dynamic algorithm that names and profiles clusters based on relative dataset benchmarks.
- **Executable Jupyter Notebook:** 18 documented sections containing LaTeX mathematical explanations and pre-rendered figures.
- **Modular Production Code:** Clean functions adhering to PEP 8 standards with automated unit tests.

---

## Technologies Used
- **Language:** Python 3.9+
- **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization:** [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Machine Learning & Modeling:** [Scikit-learn](https://scikit-learn.org/) (`KMeans`, `StandardScaler`, `silhouette_score`, `PCA`)
- **Interactive Web Application:** [Streamlit](https://streamlit.io/)
- **Environment & Notebooks:** [JupyterLab / Notebook](https://jupyter.org/), `nbconvert`

---

## Dataset
The project analyzes a dataset of **1,000 customer records** stored in `data/customer_data.csv`.

| Field Name | Type | Range / Format | Description |
| :--- | :--- | :--- | :--- |
| `CustomerID` | String | `CUST-0001` - `CUST-1000` | Unique customer identification code |
| `Age` | Integer | 18 – 70 | Customer age in years |
| `Gender` | Categorical | `Male`, `Female` | Self-reported customer gender |
| `AnnualIncome` | Integer | $18,000 – $145,000 | Annual customer earnings in USD |
| `SpendingScore` | Integer | 1 – 100 | Proprietary behavioral spend propensity rating |
| `PurchaseFrequency` | Integer | 3 – 50 | Total transactions completed per calendar year |
| `AveragePurchaseValue`| Float | $20.00 – $480.00 | Average transaction amount in USD |
| `TotalPurchases` | Integer | 4 – 150 | Cumulative purchases across account lifetime |
| `WebsiteVisits` | Integer | 3 – 58 | Number of storefront visits per month |
| `MembershipYears` | Integer | 1 – 10 | Duration of customer loyalty account membership |

---

## Project Structure
```text
customer-segmentation/
│
├── data/
│   ├── customer_data.csv            # 1,000 customer dataset records
│   └── generate_data.py             # Reproducible synthetic dataset generator
│
├── notebooks/
│   ├── customer_segmentation_analysis.ipynb   # 18-section end-to-end analysis notebook
│   └── build_notebook.py                      # Automated notebook generation script
│
├── src/
│   ├── data_preprocessing.py        # Loading, cleaning, validation, StandardScaler
│   ├── clustering.py                # Elbow, Silhouette, KMeans, PCA, profiling
│   └── visualization.py             # Matplotlib & Seaborn publication-ready charts
│
├── tests/
│   └── test_pipeline.py             # Automated unit tests for data & model pipeline
│
├── app.py                           # Streamlit interactive analytics web application
├── requirements.txt                 # Pinned dependencies
├── README.md                        # Documentation and project manual
├── .gitignore                       # Git exclusion rules
└── LICENSE                          # MIT License
```

---

## Methodology
The analytical lifecycle follows a standard data science methodology:
1. **Data Ingestion & Hygiene:** Verify schema integrity, ensure zero duplicate records, check boundary constraints.
2. **Exploratory Data Analysis:** Compute univariate metrics, assess skewness, and analyze correlation heatmaps.
3. **Feature Selection:** Isolate the 5 core drivers of purchasing behavior: `AnnualIncome`, `SpendingScore`, `PurchaseFrequency`, `AveragePurchaseValue`, and `Age`.
4. **Feature Standardization:** Scale data using `StandardScaler` to ensure unit variance and zero mean.
5. **Model Optimization:** Calculate WCSS across $K \in [2, 10]$ (Elbow Method) and evaluate Silhouette Scores.
6. **Model Training:** Fit K-Means with $K=5$ and `k-means++` initialization.
7. **Dimensionality Reduction:** Transform the 5D feature space into 2 principal components with PCA.
8. **Business Profiling:** Calculate cluster centroids, assign business personas, and formulate targeted retention playbooks.

---

## Exploratory Data Analysis
Key observations from initial data exploration:
- **Income Distribution:** Ranges from $18k to $145k with distinct demographic clusters around entry-level salaries ($30k-$40k), middle management ($60k-$70k), and executive brackets ($100k+).
- **Spending Score Distribution:** Bimodal distribution highlighting distinct populations of budget-conscious shoppers (scores < 40) and active spenders (scores > 70).
- **Correlation Insights:** Strong positive correlation ($r = 0.81$) between `PurchaseFrequency` and `SpendingScore`. Strong correlation ($r = 0.74$) between `AnnualIncome` and `AveragePurchaseValue`.

---

## Clustering Method

### 1. StandardScaler
K-Means relies on the Euclidean Distance metric:
$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
Because `AnnualIncome` is measured in tens of thousands of dollars while `SpendingScore` ranges from 1 to 100, raw distances would be completely overwhelmed by income variance. `StandardScaler` standardizes each feature:
$$z = \frac{x - \mu}{\sigma}$$
ensuring equal weight across all dimensions.

### 2. K-Means Algorithm
K-Means partitions $n$ observations into $K$ clusters by minimizing the within-cluster sum of squares:
$$\arg\min_{\mathbf{S}} \sum_{i=1}^{K} \sum_{\mathbf{x} \in S_i} ||\mathbf{x} - \boldsymbol{\mu}_i||^2$$
We employ `k-means++` centroid initialization to avoid local minima.

### 3. Elbow Method
By plotting Inertia (WCSS) against $K$, the rate of inertia reduction decreases sharply after $K=5$, forming an unmistakable "elbow" inflection point.

### 4. Silhouette Score
The Silhouette Score quantitatively verifies cohesion and separation:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
The maximum average silhouette score is achieved at **$K = 5$ ($s \approx 0.44$)**, confirming 5 well-separated clusters.

### 5. Principal Component Analysis (PCA)
PCA projects the 5 standardized features onto 2 orthogonal eigenvectors explaining over **72% of total variance**, providing a clear 2D visualization of the 5 clusters.

---

## Customer Segments

| Cluster | Discovered Persona | Size (%) | Avg Income | Spending Score | Avg Order Freq | Avg Ticket Size |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | **High-Value Loyalists (Champions)** | 22.0% | $105,400 | 83.2 / 100 | 36.1 / yr | $321.50 |
| **1** | **Budget / Value Seekers** | 22.0% | $32,100 | 25.4 / 100 | 9.2 / yr | $48.20 |
| **2** | **Young Trendsetters (High Spenders)** | 20.0% | $38,200 | 81.9 / 100 | 30.2 / yr | $109.80 |
| **3** | **Conservative Affluent (Savers)** | 18.0% | $103,800 | 23.1 / 100 | 12.0 / yr | $244.70 |
| **4** | **Moderate Mainstream (Core Tier)** | 18.0% | $63,100 | 50.8 / 100 | 19.3 / yr | $134.90 |

---

## Business Insights

### Actionable Strategies by Segment:
1. **High-Value Loyalists (Tier 1: VIP):**
   - **Insight:** Generate the highest revenue per transaction ($320+) and visit frequently.
   - **Strategy:** Exclusive concierge service, early access to new product collections, and invitation-only loyalty tiers.
2. **Young Trendsetters (Tier 2: Growth):**
   - **Insight:** Highly digital (38+ web visits/mo) with high spend propensity despite moderate salaries.
   - **Strategy:** Social commerce campaigns (TikTok/Instagram), Buy-Now-Pay-Later (BNPL) checkout, flash sales.
3. **Conservative Affluent (Tier 3: Opportunity):**
   - **Insight:** High earning power but conservative transaction frequency.
   - **Strategy:** Quality assurance, warranty highlights, craftsmanship storytelling, and seasonal luxury bundles.
4. **Budget / Value Seekers (Tier 4: Price-Sensitive):**
   - **Insight:** Highly sensitive to pricing and shipping fees.
   - **Strategy:** Clearance sales, bundled bulk discounts, and free shipping thresholds to increase basket size.
5. **Moderate Mainstream (Tier 5: Core):**
   - **Insight:** Dependable, steady baseline shoppers.
   - **Strategy:** Cross-sell promotions, loyalty point multipliers, and personalized category recommendations.

---

## Dashboard
The interactive Streamlit dashboard provides non-technical stakeholders with self-service analytics.

### Key Dashboard Sections:
1. **Dataset Overview:** Live KPI counters (Total Customers, Average Age, Income, Spending Score, Purchase Frequency).
2. **Customer Segments:** 2D PCA cluster projection and cluster size distributions.
3. **Cluster Profile Analysis:** Detailed metric summary table with side-by-side bar chart comparisons.
4. **Customer Behavior:** Interactive bivariate scatter plots (Income vs Spending, Frequency vs Spending, Age vs Spending).
5. **Business Insights:** Automatically generated demographic and financial profiling cards.
6. **Marketing Recommendations:** Actionable marketing playbooks for each segment.

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/prathap-ds/customer-segmentation.git
cd customer-segmentation
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
```

**Activate on Windows:**
```bash
venv\Scripts\activate
```

**Activate on macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Dashboard
```bash
streamlit run app.py
```
*The web browser will automatically open at `http://localhost:8501`.*

### 5. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/customer_segmentation_analysis.ipynb
```
*Or launch JupyterLab:*
```bash
jupyter lab
```

### 6. Run Unit Tests
```bash
python -m unittest tests/test_pipeline.py
```

---

## Results
- The **Elbow Method** demonstrated a clear reduction in inertia curvature at $K = 5$.
- The **Silhouette Analysis** confirmed $K = 5$ as optimal with an average silhouette score of **0.44**, outperforming $K=3$ (0.35) and $K=4$ (0.38).
- **PCA dimensionality reduction** successfully projected 5-dimensional customer behaviors into 2 principal components retaining >72% of dataset variance.
- Automated persona generation successfully categorized distinct archetypes without manual label intervention.

---

## Conclusion
This project demonstrates the practical value of unsupervised machine learning in modern business intelligence. By replacing broad demographic assumptions with multidimensional behavioral clustering, enterprises can:
- Minimize customer churn by protecting VIP loyalty tiers.
- Optimize digital marketing spend by routing tailored offers to price-elastic and trend-driven cohorts.
- Convert high-earning savers into recurring buyers through quality-oriented messaging.

---

## Future Improvements
- **Alternative Clustering Algorithms:** Benchmark K-Means against **DBSCAN** for non-linear density clustering and **Hierarchical Agglomerative Clustering** with dendrograms.
- **RFM Integration:** Incorporate Recency, Frequency, and Monetary (RFM) scoring for granular marketing lifecycle triggers.
- **Customer Lifetime Value (CLV):** Train supervised regression models (e.g., XGBoost, LightGBM) to forecast lifetime revenue per cluster.
- **Production API & CRM Webhooks:** Deploy a FastAPI service connecting real-time segmentation directly to marketing automation platforms (HubSpot, Salesforce, Klaviyo).
- **Cloud Deployment:** Containerize with Docker and deploy to Streamlit Community Cloud or AWS ECS.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
