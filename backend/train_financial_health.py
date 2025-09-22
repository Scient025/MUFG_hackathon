#!/usr/bin/env python3
"""
Enhanced Financial Health Model Training with Visualizations
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from ml_visualizer import MLVisualizer
from supabase_config import supabase, USER_PROFILES_TABLE

class FinancialHealthTrainer:
    def __init__(self):
        self.model = None
        self.visualizer = MLVisualizer()
        self.results = {}
    
    def calculate_health_score(self, row):
        """Calculate financial health score using business logic"""
        score = 0
        
        # Income component (20 points)
        if row['Annual_Income'] > 100000: score += 20
        elif row['Annual_Income'] > 75000: score += 15
        elif row['Annual_Income'] > 50000: score += 10
        else: score += 5
        
        # Savings component (25 points)
        savings_ratio = row['Savings_to_Income_Ratio']
        if savings_ratio > 2.0: score += 25
        elif savings_ratio > 1.0: score += 20
        elif savings_ratio > 0.5: score += 15
        elif savings_ratio > 0.2: score += 10
        else: score += 5
        
        # Contribution component (20 points)
        contrib_ratio = row['Contribution_Percent_of_Income']
        if contrib_ratio > 0.15: score += 20
        elif contrib_ratio > 0.10: score += 15
        elif contrib_ratio > 0.05: score += 10
        else: score += 5
        
        # Debt component (15 points)
        dti = row['DTI_Ratio']
        if dti < 0.2: score += 15
        elif dti < 0.3: score += 12
        elif dti < 0.4: score += 8
        else: score += 3
        
        # Diversity component (10 points)
        diversity = row['Portfolio_Diversity_Score']
        if diversity > 0.8: score += 10
        elif diversity > 0.6: score += 8
        elif diversity > 0.4: score += 5
        else: score += 2
        
        # Experience component (10 points)
        experience = row['Investment_Experience_Level_encoded']
        if experience >= 2: score += 10
        elif experience >= 1: score += 7
        else: score += 4
        
        return min(100, max(0, score))
    
    def load_and_preprocess_data(self):
        """Load and preprocess data for financial health prediction"""
        print("Loading data for Financial Health Model...")
        
        # Fetch data from Supabase
        response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
        if not response.data:
            raise ValueError("No data found in Supabase database")
        
        df = pd.DataFrame(response.data)
        
        # Handle missing values
        df = df.fillna({
            'Annual_Income': 0,
            'Current_Savings': 0,
            'Contribution_Amount': 0,
            'Years_Contributed': 0,
            'Age': 30,
            'Portfolio_Diversity_Score': 0.5,
            'Savings_Rate': 0.1,
            'Debt_Level': 'Low',
            'Investment_Experience_Level': 'Beginner',
            'Contribution_Frequency': 'Monthly'
        })
        
        # Convert numeric columns
        numeric_columns = ['Annual_Income', 'Current_Savings', 'Contribution_Amount', 
                          'Years_Contributed', 'Portfolio_Diversity_Score', 'Savings_Rate', 'Age']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode categorical variables
        from sklearn.preprocessing import LabelEncoder
        categorical_columns = ['Debt_Level', 'Investment_Experience_Level', 'Contribution_Frequency']
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
        
        df['Savings_to_Income_Ratio'] = np.where(
            df['Annual_Income'] > 0,
            df['Current_Savings'] / df['Annual_Income'], 0
        )
        
        df['Contribution_Percent_of_Income'] = np.where(
            df['Annual_Income'] > 0,
            (df['Contribution_Amount'] * 12) / df['Annual_Income'], 0
        )
        
        df['Risk_Adjusted_Return'] = np.where(
            df['Volatility'] > 0,
            df['Annual_Return_Rate'] / df['Volatility'], 0
        )
        
        return df
    
    def train_model(self):
        """Train the financial health model with visualizations"""
        print("Training Financial Health Model...")
        
        # Load data
        df = self.load_and_preprocess_data()
        
        # Features for financial health
        health_features = [
            'Annual_Income', 'Current_Savings', 'Savings_Rate', 'Debt_Level_encoded',
            'Portfolio_Diversity_Score', 'Contribution_Amount', 'Contribution_Frequency_encoded',
            'Years_Contributed', 'DTI_Ratio', 'Savings_to_Income_Ratio',
            'Contribution_Percent_of_Income', 'Risk_Adjusted_Return', 'Age',
            'Investment_Experience_Level_encoded'
        ]
        
        # Create target variable
        df['Financial_Health_Score'] = df.apply(self.calculate_health_score, axis=1)
        
        # Prepare data
        health_data = df[health_features + ['Financial_Health_Score']].dropna()
        X = health_data[health_features]
        y = health_data['Financial_Health_Score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n=== Financial Health Model Metrics ===")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance plot
        self.visualizer.plot_feature_importance(health_features, self.model.feature_importances_, 
                                              "Financial Health")
        
        # Learning curve
        self.visualizer.plot_learning_curve(self.model, X_train, y_train, "Financial Health")
        
        metrics = {'rmse': rmse, 'r2': r2}
        self.results['Financial Health'] = metrics
        return metrics
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/financial_health_model.pkl')
        
        print("Financial health model saved successfully!")

if __name__ == "__main__":
    trainer = FinancialHealthTrainer()
    metrics = trainer.train_model()
    trainer.save_model()
    
    print("\n🎉 Financial Health Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
