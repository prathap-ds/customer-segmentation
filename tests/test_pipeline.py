"""
Test pipeline verification for Customer Segmentation.
Validates end-to-end data processing, feature scaling, model training, and metrics.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add src to system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

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


class TestCustomerSegmentationPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "customer_data.csv"))
        cls.df = load_data(cls.data_path)
        cls.features = ["AnnualIncome", "SpendingScore", "PurchaseFrequency", "AveragePurchaseValue", "Age"]
        
    def test_load_data(self):
        self.assertGreaterEqual(len(self.df), 500)
        expected_cols = [
            "CustomerID", "Age", "Gender", "AnnualIncome", 
            "SpendingScore", "PurchaseFrequency", "AveragePurchaseValue", 
            "TotalPurchases", "WebsiteVisits", "MembershipYears"
        ]
        for col in expected_cols:
            self.assertIn(col, self.df.columns)
            
    def test_data_inspection_and_cleaning(self):
        inspection = inspect_data(self.df)
        self.assertEqual(inspection["duplicates"], 0)
        self.assertEqual(inspection["total_missing"], 0)
        
        cleaned = clean_data(self.df)
        self.assertEqual(len(cleaned), len(self.df))
        
    def test_feature_scaling(self):
        scaled_array, scaler, scaled_df = scale_features(self.df, self.features)
        self.assertEqual(scaled_array.shape, (len(self.df), len(self.features)))
        # Mean should be close to 0 and std close to 1
        np.testing.assert_almost_equal(scaled_array.mean(axis=0), np.zeros(len(self.features)), decimal=1)
        np.testing.assert_almost_equal(scaled_array.std(axis=0), np.ones(len(self.features)), decimal=1)
        
    def test_clustering_and_diagnostics(self):
        scaled_array, _, _ = scale_features(self.df, self.features)
        
        # Test Elbow and Silhouette
        inertias = calculate_elbow(scaled_array, k_range=range(2, 6))
        self.assertEqual(len(inertias), 4)
        # Inertia should be strictly decreasing as K increases
        self.assertGreater(inertias[2], inertias[3])
        self.assertGreater(inertias[3], inertias[4])
        
        silhouettes = calculate_silhouette(scaled_array, k_range=range(2, 6))
        self.assertEqual(len(silhouettes), 4)
        for s in silhouettes.values():
            self.assertGreater(s, 0.2)
            
        # Test KMeans training
        kmeans, labels = train_kmeans(scaled_array, n_clusters=5)
        self.assertEqual(len(labels), len(self.df))
        self.assertEqual(len(np.unique(labels)), 5)
        
        # Test PCA
        pca_coords, variance_ratio, _ = compute_pca(scaled_array, n_components=2)
        self.assertEqual(pca_coords.shape, (len(self.df), 2))
        self.assertGreater(sum(variance_ratio), 0.5)
        
    def test_profiles_and_interpretation(self):
        scaled_array, _, _ = scale_features(self.df, self.features)
        _, labels = train_kmeans(scaled_array, n_clusters=5)
        df_copy = self.df.copy()
        df_copy["Cluster"] = labels
        
        profiles = compute_cluster_profiles(df_copy)
        self.assertEqual(len(profiles), 5)
        self.assertIn("CustomerCount", profiles.columns)
        self.assertIn("Percentage", profiles.columns)
        
        interpretations = interpret_clusters(profiles)
        self.assertEqual(len(interpretations), 5)
        for cid, info in interpretations.items():
            self.assertIn("name", info)
            self.assertIn("marketing_strategy", info)
            self.assertIn("summary", info)


if __name__ == "__main__":
    unittest.main()
