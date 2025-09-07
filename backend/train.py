import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class SuperannuationMLPipeline:
    def __init__(self, csv_path: str = "case1.csv"):
        self.csv_path = csv_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.cluster_labels = None
        
    def load_and_preprocess_data(self):
        """Load CSV data and preprocess for ML models"""
        print("Loading data from CSV...")
        self.df = pd.read_csv(self.csv_path)
        
        # Handle missing values
        self.df = self.df.fillna({
            'Risk_Tolerance': 'Medium',
            'Investment_Type': 'ETF',
            'Fund_Name': 'Unknown',
            'Marital_Status': 'Single',
            'Education_Level': 'Bachelor\'s',
            'Health_Status': 'Average'
        })
        
        # Encode categorical variables
        categorical_columns = [
            'Gender', 'Country', 'Employment_Status', 'Risk_Tolerance',
            'Investment_Type', 'Fund_Name', 'Marital_Status', 'Education_Level',
            'Health_Status', 'Home_Ownership_Status', 'Investment_Experience_Level',
            'Financial_Goals', 'Insurance_Coverage', 'Pension_Type', 'Withdrawal_Strategy'
        ]
        
        for col in categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Data loaded: {len(self.df)} users, {len(self.df.columns)} features")
        return self.df
    
    def train_user_segmentation_model(self, n_clusters: int = 5):
        """Train KMeans clustering for user segmentation"""
        print("Training user segmentation model...")
        
        # Features for clustering
        clustering_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Years_Contributed', 'Portfolio_Diversity_Score'
        ]
        
        # Filter out missing values
        clustering_data = self.df[clustering_features].dropna()
        
        # Scale features
        scaled_features = self.scaler.fit_transform(clustering_data)
        
        # Train KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(scaled_features)
        
        # Add cluster labels to dataframe
        self.df.loc[clustering_data.index, 'Cluster'] = self.cluster_labels
        
        # Save model
        self.models['kmeans'] = kmeans
        joblib.dump(kmeans, 'models/kmeans_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        print(f"User segmentation complete. {n_clusters} clusters created.")
        return kmeans
    
    def train_risk_prediction_model(self):
        """Train Logistic Regression for risk tolerance prediction"""
        print("Training risk prediction model...")
        
        # Features for risk prediction
        risk_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Years_Contributed', 'Investment_Experience_Level_encoded',
            'Portfolio_Diversity_Score', 'Savings_Rate', 'Debt_Level'
        ]
        
        # Prepare data
        risk_data = self.df[risk_features + ['Risk_Tolerance_encoded']].dropna()
        X = risk_data[risk_features]
        y = risk_data['Risk_Tolerance_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and fit separate scaler for risk prediction
        risk_scaler = StandardScaler()
        X_train_scaled = risk_scaler.fit_transform(X_train)
        X_test_scaled = risk_scaler.transform(X_test)
        
        # Train model
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = lr_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Risk prediction model accuracy: {accuracy:.3f}")
        
        # Save model and scaler
        self.models['risk_prediction'] = lr_model
        joblib.dump(lr_model, 'models/risk_prediction_model.pkl')
        joblib.dump(risk_scaler, 'models/risk_scaler.pkl')
        
        return lr_model
    
    def train_investment_recommendation_model(self):
        """Train XGBoost for investment recommendations and pension projections"""
        print("Training investment recommendation model...")
        
        # Features for investment recommendation
        investment_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Investment_Type_encoded', 'Annual_Return_Rate',
            'Volatility', 'Fees_Percentage', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded'
        ]
        
        # Target: Projected_Pension_Amount
        investment_data = self.df[investment_features + ['Projected_Pension_Amount']].dropna()
        X = investment_data[investment_features]
        y = investment_data['Projected_Pension_Amount']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and fit scaler for investment features
        from sklearn.preprocessing import StandardScaler
        investment_scaler = StandardScaler()
        X_train_scaled = investment_scaler.fit_transform(X_train)
        X_test_scaled = investment_scaler.transform(X_test)
        
        # Train XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        xgb_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = xgb_model.predict(X_test_scaled)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        print(f"Investment recommendation model RMSE: {rmse:.2f}")
        
        # Save model and scaler
        self.models['investment_recommendation'] = xgb_model
        joblib.dump(xgb_model, 'models/investment_recommendation_model.pkl')
        joblib.dump(investment_scaler, 'models/investment_scaler.pkl')
        
        return xgb_model
    
    def train_all_models(self):
        """Train all ML models"""
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Train models
        self.train_user_segmentation_model()
        self.train_risk_prediction_model()
        self.train_investment_recommendation_model()
        
        # Save label encoders
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        
        print("All models trained and saved successfully!")
        return self.models

if __name__ == "__main__":
    # Train all models
    pipeline = SuperannuationMLPipeline()
    models = pipeline.train_all_models()
    
    print("\nModel training complete!")
    print("Models saved to 'models/' directory:")
    print("- kmeans_model.pkl: User segmentation")
    print("- risk_prediction_model.pkl: Risk tolerance prediction")
    print("- investment_recommendation_model.pkl: Investment recommendations")
    print("- scaler.pkl: Feature scaler")
    print("- label_encoders.pkl: Categorical encoders")
