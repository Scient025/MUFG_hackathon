#!/usr/bin/env python3
"""
Enhanced User Segmentation Model Training with Visualizations
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib
import os
from ml_visualizer import MLVisualizer
from supabase_config import supabase, USER_PROFILES_TABLE

class UserSegmentationTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.visualizer = MLVisualizer()
        self.results = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess data for user segmentation"""
        print("Loading data for User Segmentation Model...")
        
        # Fetch data from Supabase
        response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
        if not response.data:
            raise ValueError("No data found in Supabase database")
        
        df = pd.DataFrame(response.data)
        
        # Handle missing values
        df = df.fillna({
            'Age': 30,
            'Annual_Income': 0,
            'Current_Savings': 0,
            'Contribution_Amount': 0,
            'Years_Contributed': 0,
            'Portfolio_Diversity_Score': 0.5,
            'Risk_Tolerance': 'Medium'
        })
        
        # Convert numeric columns
        numeric_columns = ['Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
                          'Years_Contributed', 'Portfolio_Diversity_Score']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode risk tolerance
        if 'Risk_Tolerance' in df.columns:
            df['Risk_Tolerance'] = df['Risk_Tolerance'].fillna('Medium')
            le = LabelEncoder()
            df['Risk_Tolerance_encoded'] = le.fit_transform(df['Risk_Tolerance'].astype(str))
        
        return df
    
    def train_model(self, n_clusters=5):
        """Train the user segmentation model with visualizations"""
        print(f"Training User Segmentation Model with {n_clusters} clusters...")
        
        # Load data
        df = self.load_and_preprocess_data()
        
        # Features for clustering
        clustering_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Years_Contributed', 'Portfolio_Diversity_Score'
        ]
        
        # Filter out missing values
        clustering_data = df[clustering_features].dropna()
        
        # Scale features
        self.scaler = StandardScaler()
        scaled_features = self.scaler.fit_transform(clustering_data)
        
        # Train KMeans
        self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = self.model.fit_predict(scaled_features)
        
        # Add cluster labels to dataframe
        df.loc[clustering_data.index, 'Cluster'] = cluster_labels
        
        # Calculate clustering metrics
        silhouette_avg = silhouette_score(scaled_features, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(scaled_features, cluster_labels)
        
        print(f"\n=== User Segmentation Model Metrics ===")
        print(f"Number of clusters: {n_clusters}")
        print(f"Silhouette Score: {silhouette_avg:.4f}")
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f}")
        
        # Cluster analysis
        cluster_analysis = clustering_data.copy()
        cluster_analysis['Cluster'] = cluster_labels
        
        print(f"\nCluster Analysis:")
        for i in range(n_clusters):
            cluster_data = cluster_analysis[cluster_analysis['Cluster'] == i]
            print(f"\nCluster {i} ({len(cluster_data)} users):")
            print(f"  Average Age: {cluster_data['Age'].mean():.1f}")
            print(f"  Average Income: ${cluster_data['Annual_Income'].mean():,.0f}")
            print(f"  Average Savings: ${cluster_data['Current_Savings'].mean():,.0f}")
            print(f"  Average Contribution: ${cluster_data['Contribution_Amount'].mean():,.0f}")
        
        # Feature importance (using cluster centers)
        feature_importance = np.abs(self.model.cluster_centers_).mean(axis=0)
        self.visualizer.plot_feature_importance(clustering_features, feature_importance, 
                                              "User Segmentation")
        
        metrics = {
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'n_clusters': n_clusters
        }
        
        self.results['User Segmentation'] = metrics
        return metrics
    
    def save_model(self):
        """Save the trained model and scaler"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/kmeans_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        print("User segmentation model saved successfully!")

if __name__ == "__main__":
    trainer = UserSegmentationTrainer()
    metrics = trainer.train_model()
    trainer.save_model()
    
    print("\n🎉 User Segmentation Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
