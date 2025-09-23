#!/usr/bin/env python3
"""
Enhanced Churn Risk Model Training with Visualizations
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from ml_visualizer import MLVisualizer
from supabase_config import supabase, USER_PROFILES_TABLE

class ChurnRiskTrainer:
    def __init__(self):
        self.model = None
        self.visualizer = MLVisualizer()
        self.results = {}
    
    def create_churn_label(self, row):
        """Create churn label based on business logic"""
        churn_score = 0
        
        # Contribution frequency (irregular = higher churn risk)
        if row['Contribution_Frequency_encoded'] == 0: churn_score += 1
        
        # Suspicious activity
        if row.get('Suspicious_Flag', 'No') == 'Yes': churn_score += 1
        
        # Low contribution percentage
        if row['Contribution_Percent_of_Income'] < 0.02: churn_score += 1
        
        # High debt-to-income ratio
        if row['DTI_Ratio'] > 0.4: churn_score += 1
        
        # Employment status (unemployed = higher risk)
        if row['Employment_Status_encoded'] == 0: churn_score += 1
        
        # Low savings rate
        if row['Savings_Rate'] < 0.05: churn_score += 1
        
        # Young age with low contributions
        if row['Age'] < 30 and row['Contribution_Amount'] < 500: churn_score += 1
        
        return 1 if churn_score >= 2 else 0
    
    def load_and_preprocess_data(self):
        """Load and preprocess data for churn risk prediction"""
        print("Loading data for Churn Risk Model...")
        
        # Fetch data from Supabase
        response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
        if not response.data:
            raise ValueError("No data found in Supabase database")
        
        df = pd.DataFrame(response.data)
        
        # Handle missing values
        df = df.fillna({
            'Age': 30,
            'Annual_Income': 0,
            'Contribution_Amount': 0,
            'Years_Contributed': 0,
            'Savings_Rate': 0.1,
            'Portfolio_Diversity_Score': 0.5,
            'Employment_Status': 'Full-time',
            'Contribution_Frequency': 'Monthly',
            'Suspicious_Flag': 'No',
            'Debt_Level': 'Low'
        })
        
        # Convert numeric columns
        numeric_columns = ['Age', 'Annual_Income', 'Contribution_Amount', 'Years_Contributed',
                          'Savings_Rate', 'Portfolio_Diversity_Score']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode categorical variables
        categorical_columns = ['Employment_Status', 'Contribution_Frequency', 'Debt_Level']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # Create derived features
        df['DTI_Ratio'] = np.where(
            df['Annual_Income'] > 0,
            df['Debt_Level'].map({'Low': 0.1, 'Medium': 0.3, 'High': 0.5, 'Unknown': 0.2}).fillna(0.2),
            np.nan
        )
        
        df['Contribution_Percent_of_Income'] = np.where(
            df['Annual_Income'] > 0,
            (df['Contribution_Amount'] * 12) / df['Annual_Income'], 0
        )
        
        return df
    
    def train_model(self):
        """Train the churn risk model with visualizations"""
        print("Training Churn Risk Model...")
        
        # Load data
        df = self.load_and_preprocess_data()
        
        # Features for churn prediction
        churn_features = [
            'Age', 'Annual_Income', 'Employment_Status_encoded', 'Debt_Level_encoded',
            'Contribution_Frequency_encoded', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Contribution_Percent_of_Income', 'DTI_Ratio'
        ]
        
        # Create target variable
        df['Churn_Risk'] = df.apply(self.create_churn_label, axis=1)
        
        # Prepare data
        churn_data = df[churn_features + ['Churn_Risk']].dropna()
        X = churn_data[churn_features]
        y = churn_data['Churn_Risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Visualizations
        class_names = ['No Churn', 'Churn Risk']
        
        # Confusion Matrix
        self.visualizer.plot_confusion_matrix(y_test, y_pred, "Churn Risk", class_names)
        
        # ROC Curve
        self.visualizer.plot_roc_curve(y_test, y_pred_proba, "Churn Risk", class_names)
        
        # Learning Curve
        self.visualizer.plot_learning_curve(self.model, X_train, y_train, "Churn Risk")
        
        # Feature Importance
        self.visualizer.plot_feature_importance(churn_features, self.model.feature_importances_, 
                                              "Churn Risk")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        print(f"\n=== Churn Risk Model Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.results['Churn Risk'] = metrics
        return metrics
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/churn_risk_model.pkl')
        
        print("Churn risk model saved successfully!")

if __name__ == "__main__":
    trainer = ChurnRiskTrainer()
    metrics = trainer.train_model()
    trainer.save_model()
    
    print("\n🎉 Churn Risk Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
