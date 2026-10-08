"""
Script to generate a realistic synthetic customer segmentation dataset.
Generates 1,000 customer records with realistic demographic and behavioral attributes.
"""

import numpy as np
import pandas as pd
import os

def generate_customer_dataset(num_records=1000, random_state=42):
    np.random.seed(random_state)
    
    # Define 5 realistic customer personas / clusters
    # 1. High-Value Loyalists (~22%)
    # 2. Budget / Value Seekers (~22%)
    # 3. Young Trendsetters (~20%)
    # 4. Conservative Affluent (~18%)
    # 5. Moderate Mainstream (~18%)
    
    counts = [220, 220, 200, 180, 180]
    
    records = []
    cust_id = 1
    
    # Segment 1: High-Value Loyalists
    for _ in range(counts[0]):
        age = int(np.clip(np.random.normal(42, 7), 25, 62))
        gender = np.random.choice(["Female", "Male"], p=[0.54, 0.46])
        annual_income = int(np.clip(np.random.normal(105000, 12000), 80000, 145000))
        spending_score = int(np.clip(np.random.normal(83, 7), 68, 99))
        purchase_freq = int(np.clip(np.random.normal(36, 5), 24, 50))
        avg_purchase_val = round(float(np.clip(np.random.normal(320, 45), 200, 480)), 2)
        membership_years = int(np.clip(np.random.normal(6.5, 1.8), 3, 10))
        total_purchases = int(purchase_freq * (membership_years * 0.4 + 1.2) + np.random.randint(-5, 8))
        total_purchases = max(20, total_purchases)
        website_visits = int(np.clip(np.random.normal(32, 6), 18, 52))
        
        records.append({
            "CustomerID": f"CUST-{cust_id:04d}",
            "Age": age,
            "Gender": gender,
            "AnnualIncome": annual_income,
            "SpendingScore": spending_score,
            "PurchaseFrequency": purchase_freq,
            "AveragePurchaseValue": avg_purchase_val,
            "TotalPurchases": total_purchases,
            "WebsiteVisits": website_visits,
            "MembershipYears": membership_years
        })
        cust_id += 1
        
    # Segment 2: Budget / Value Seekers
    for _ in range(counts[1]):
        age = int(np.clip(np.random.normal(46, 12), 22, 68))
        gender = np.random.choice(["Female", "Male"], p=[0.50, 0.50])
        annual_income = int(np.clip(np.random.normal(32000, 7000), 18000, 48000))
        spending_score = int(np.clip(np.random.normal(25, 8), 10, 44))
        purchase_freq = int(np.clip(np.random.normal(9, 3), 3, 16))
        avg_purchase_val = round(float(np.clip(np.random.normal(48, 12), 20, 95)), 2)
        membership_years = int(np.clip(np.random.normal(2.5, 1.2), 1, 6))
        total_purchases = int(purchase_freq * (membership_years * 0.5 + 0.8) + np.random.randint(-3, 4))
        total_purchases = max(4, total_purchases)
        website_visits = int(np.clip(np.random.normal(10, 3.5), 3, 20))
        
        records.append({
            "CustomerID": f"CUST-{cust_id:04d}",
            "Age": age,
            "Gender": gender,
            "AnnualIncome": annual_income,
            "SpendingScore": spending_score,
            "PurchaseFrequency": purchase_freq,
            "AveragePurchaseValue": avg_purchase_val,
            "TotalPurchases": total_purchases,
            "WebsiteVisits": website_visits,
            "MembershipYears": membership_years
        })
        cust_id += 1

    # Segment 3: Young Trendsetters / High Spenders
    for _ in range(counts[2]):
        age = int(np.clip(np.random.normal(24, 3.5), 18, 33))
        gender = np.random.choice(["Female", "Male"], p=[0.58, 0.42])
        annual_income = int(np.clip(np.random.normal(38000, 7500), 22000, 56000))
        spending_score = int(np.clip(np.random.normal(82, 7), 65, 98))
        purchase_freq = int(np.clip(np.random.normal(30, 5), 18, 44))
        avg_purchase_val = round(float(np.clip(np.random.normal(110, 22), 60, 185)), 2)
        membership_years = int(np.clip(np.random.normal(2.2, 1.0), 1, 4))
        total_purchases = int(purchase_freq * (membership_years * 0.6 + 0.9) + np.random.randint(-4, 6))
        total_purchases = max(15, total_purchases)
        website_visits = int(np.clip(np.random.normal(38, 7), 22, 58))
        
        records.append({
            "CustomerID": f"CUST-{cust_id:04d}",
            "Age": age,
            "Gender": gender,
            "AnnualIncome": annual_income,
            "SpendingScore": spending_score,
            "PurchaseFrequency": purchase_freq,
            "AveragePurchaseValue": avg_purchase_val,
            "TotalPurchases": total_purchases,
            "WebsiteVisits": website_visits,
            "MembershipYears": membership_years
        })
        cust_id += 1

    # Segment 4: Conservative Affluent (High Income, Prudent Spenders)
    for _ in range(counts[3]):
        age = int(np.clip(np.random.normal(53, 8), 36, 70))
        gender = np.random.choice(["Female", "Male"], p=[0.48, 0.52])
        annual_income = int(np.clip(np.random.normal(104000, 12500), 78000, 142000))
        spending_score = int(np.clip(np.random.normal(23, 7), 8, 42))
        purchase_freq = int(np.clip(np.random.normal(12, 3.5), 5, 20))
        avg_purchase_val = round(float(np.clip(np.random.normal(245, 42), 140, 380)), 2)
        membership_years = int(np.clip(np.random.normal(5.4, 1.8), 2, 9))
        total_purchases = int(purchase_freq * (membership_years * 0.5 + 1.0) + np.random.randint(-3, 5))
        total_purchases = max(10, total_purchases)
        website_visits = int(np.clip(np.random.normal(13, 3.8), 5, 24))
        
        records.append({
            "CustomerID": f"CUST-{cust_id:04d}",
            "Age": age,
            "Gender": gender,
            "AnnualIncome": annual_income,
            "SpendingScore": spending_score,
            "PurchaseFrequency": purchase_freq,
            "AveragePurchaseValue": avg_purchase_val,
            "TotalPurchases": total_purchases,
            "WebsiteVisits": website_visits,
            "MembershipYears": membership_years
        })
        cust_id += 1

    # Segment 5: Moderate Mainstream
    for _ in range(counts[4]):
        age = int(np.clip(np.random.normal(39, 9), 24, 62))
        gender = np.random.choice(["Female", "Male"], p=[0.51, 0.49])
        annual_income = int(np.clip(np.random.normal(63000, 8500), 45000, 82000))
        spending_score = int(np.clip(np.random.normal(51, 6.5), 35, 68))
        purchase_freq = int(np.clip(np.random.normal(19, 4), 10, 28))
        avg_purchase_val = round(float(np.clip(np.random.normal(135, 25), 75, 210)), 2)
        membership_years = int(np.clip(np.random.normal(3.8, 1.4), 1, 7))
        total_purchases = int(purchase_freq * (membership_years * 0.5 + 1.0) + np.random.randint(-4, 5))
        total_purchases = max(12, total_purchases)
        website_visits = int(np.clip(np.random.normal(19, 4.5), 8, 32))
        
        records.append({
            "CustomerID": f"CUST-{cust_id:04d}",
            "Age": age,
            "Gender": gender,
            "AnnualIncome": annual_income,
            "SpendingScore": spending_score,
            "PurchaseFrequency": purchase_freq,
            "AveragePurchaseValue": avg_purchase_val,
            "TotalPurchases": total_purchases,
            "WebsiteVisits": website_visits,
            "MembershipYears": membership_years
        })
        cust_id += 1
        
    df = pd.DataFrame(records)
    
    # Shuffle records so segments are randomly mixed like a real customer database
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df["CustomerID"] = [f"CUST-{i+1:04d}" for i in range(len(df))]
    
    return df

if __name__ == "__main__":
    df = generate_customer_dataset(1000)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "customer_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} customer records saved to {output_path}")
    print(df.head())
