#!/usr/bin/env python3
"""
Enhanced Investment Recommendation Model Training with Visualizations
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os
from ml_visualizer import MLVisualizer
from supabase_config import supabase, USER_PROFILES_TABLE

class InvestmentRecommendationTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.visualizer = MLVisualizer()
        self.results = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess data for investment recommendation"""
        print("Loading data for Investment Recommendation Model...")
        
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
            'Savings_Rate': 0.1,
            'Portfolio_Diversity_Score': 0.5,
            'Risk_Tolerance': 'Medium',
            'Investment_Type': 'ETF',
            'Investment_Experience_Level': 'Beginner',
            'Annual_Return_Rate': 7.0,
            'Volatility': 2.0,
            'Fees_Percentage': 1.0,
            'Projected_Pension_Amount': 0
        })
        
        # Convert numeric columns
        numeric_columns = ['Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
                          'Years_Contributed', 'Savings_Rate', 'Portfolio_Diversity_Score',
                          'Annual_Return_Rate', 'Volatility', 'Fees_Percentage', 'Projected_Pension_Amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode categorical variables
        categorical_columns = ['Risk_Tolerance', 'Investment_Type', 'Investment_Experience_Level']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        return df
    
    def train_model(self):
        """Train the investment recommendation model with visualizations"""
        print("Training Investment Recommendation Model...")
        
        # Load data
        df = self.load_and_preprocess_data()
        
        # Features for investment recommendation
        investment_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Investment_Type_encoded', 'Annual_Return_Rate',
            'Volatility', 'Fees_Percentage', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded'
        ]
        
        # Target: Projected_Pension_Amount
        investment_data = df[investment_features + ['Projected_Pension_Amount']].dropna()
        X = investment_data[investment_features]
        y = investment_data['Projected_Pension_Amount']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n=== Investment Recommendation Model Metrics ===")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Feature importance plot
        self.visualizer.plot_feature_importance(investment_features, self.model.feature_importances_, 
                                              "Investment Recommendation")
        
        # Learning curve
        self.visualizer.plot_learning_curve(self.model, X_train, y_train, "Investment Recommendation")
        
        metrics = {'rmse': rmse, 'r2': r2}
        self.results['Investment Recommendation'] = metrics
        return metrics
    
    def save_model(self):
        """Save the trained model and scaler"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/investment_recommendation_model.pkl')
        
        print("Investment recommendation model saved successfully!")

if __name__ == "__main__":
    trainer = InvestmentRecommendationTrainer()
    metrics = trainer.train_model()
    trainer.save_model()
    
    print("\n🎉 Investment Recommendation Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
