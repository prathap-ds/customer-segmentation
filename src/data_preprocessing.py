"""
Data Preprocessing Module for Customer Segmentation Project.
Handles data loading, validation, cleaning, encoding, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads customer dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise FileNotFoundError(f"Error loading dataset from {file_path}: {e}")


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs comprehensive data health inspection.
    
    Checks:
    - Shape (rows, columns)
    - Data types
    - Missing / null values
    - Duplicate rows
    - Value ranges and anomalies
    
    Returns:
        Dict[str, Any]: Summary statistics and data health status.
    """
    inspection = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "numeric_summary": df.describe().to_dict()
    }
    
    # Check for invalid values in domain constraints
    invalid_checks = {}
    if "Age" in df.columns:
        invalid_checks["invalid_age"] = int(((df["Age"] < 10) | (df["Age"] > 110)).sum())
    if "AnnualIncome" in df.columns:
        invalid_checks["negative_income"] = int((df["AnnualIncome"] < 0).sum())
    if "SpendingScore" in df.columns:
        invalid_checks["invalid_spending_score"] = int(((df["SpendingScore"] < 1) | (df["SpendingScore"] > 100)).sum())
    if "PurchaseFrequency" in df.columns:
        invalid_checks["negative_purchase_freq"] = int((df["PurchaseFrequency"] < 0).sum())
        
    inspection["invalid_records"] = invalid_checks
    return inspection


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans customer data:
    - Drops duplicates
    - Imputes missing numerical values with median and categorical with mode
    - Removes invalid domain values if any
    
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    cleaned_df = df.copy()
    
    # Remove duplicate records
    cleaned_df = cleaned_df.drop_duplicates()
    
    # Handle missing values
    num_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            
    cat_cols = cleaned_df.select_dtypes(include=["object", "category", "string"]).columns
    for col in cat_cols:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode()[0])
            
    # Filter out impossible domain values
    if "Age" in cleaned_df.columns:
        cleaned_df = cleaned_df[(cleaned_df["Age"] >= 12) & (cleaned_df["Age"] <= 100)]
    if "AnnualIncome" in cleaned_df.columns:
        cleaned_df = cleaned_df[cleaned_df["AnnualIncome"] >= 0]
    if "SpendingScore" in cleaned_df.columns:
        cleaned_df = cleaned_df[(cleaned_df["SpendingScore"] >= 1) & (cleaned_df["SpendingScore"] <= 100)]
        
    return cleaned_df.reset_index(drop=True)


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts categorical columns like 'Gender' to numeric representations.
    
    Returns:
        pd.DataFrame: DataFrame with encoded features.
    """
    encoded_df = df.copy()
    if "Gender" in encoded_df.columns:
        # Binary encoding: Female -> 0, Male -> 1 (or One-Hot for general cases)
        gender_mapping = {"Female": 0, "Male": 1}
        encoded_df["Gender_Code"] = encoded_df["Gender"].map(gender_mapping).fillna(0).astype(int)
    return encoded_df


def scale_features(
    df: pd.DataFrame, 
    feature_cols: List[str]
) -> Tuple[np.ndarray, StandardScaler, pd.DataFrame]:
    """
    Scales numerical features using StandardScaler.
    
    WHY FEATURE SCALING IS ESSENTIAL FOR K-MEANS:
    K-Means is an unsupervised, distance-based clustering algorithm relying on
    Euclidean Distance: d(p, q) = sqrt(sum((p_i - q_i)^2)).
    
    If one feature has a large magnitude/variance (e.g., AnnualIncome spanning
    $20,000 to $140,000), its squared differences will completely overpower features
    with smaller scales (e.g., Age 18-70, SpendingScore 1-100, or MembershipYears 1-10).
    
    StandardScaler standardizes each feature to have a mean of 0 and standard
    deviation of 1 (Z-Score: z = (x - mu) / sigma), ensuring that all attributes
    contribute equally and proportionally to cluster formation.
    
    Args:
        df: Input DataFrame
        feature_cols: List of column names to scale
        
    Returns:
        Tuple:
            - scaled_array (np.ndarray): Scaled numerical matrix
            - scaler (StandardScaler): Fitted scaler instance
            - scaled_df (pd.DataFrame): DataFrame with scaled features
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[feature_cols])
    
    scaled_df = pd.DataFrame(
        scaled_array, 
        columns=[f"{col}_scaled" for col in feature_cols],
        index=df.index
    )
    
    return scaled_array, scaler, scaled_df
