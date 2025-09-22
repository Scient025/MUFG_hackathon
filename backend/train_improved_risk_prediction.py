#!/usr/bin/env python3
"""
Improved Risk Prediction Model with Better Performance
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from ml_visualizer import MLVisualizer
from supabase_config import supabase, USER_PROFILES_TABLE

class ImprovedRiskPredictionTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.visualizer = MLVisualizer()
        self.results = {}
    
    def load_and_preprocess_data(self):
        """Load and preprocess data with improved feature engineering"""
        print("Loading data for Improved Risk Prediction Model...")
        
        # Fetch data from Supabase
        response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
        if not response.data:
            raise ValueError("No data found in Supabase database")
        
        df = pd.DataFrame(response.data)
        
        # Handle missing values
        df = df.fillna({
            'Risk_Tolerance': 'Medium',
            'Investment_Experience_Level': 'Beginner',
            'Debt_Level': 'Low',
            'Annual_Income': 0,
            'Current_Savings': 0,
            'Contribution_Amount': 0,
            'Years_Contributed': 0,
            'Age': 30,
            'Portfolio_Diversity_Score': 0.5,
            'Savings_Rate': 0.1
        })
        
        # Convert numeric columns
        numeric_columns = ['Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount', 
                          'Years_Contributed', 'Portfolio_Diversity_Score', 'Savings_Rate']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode categorical variables
        categorical_columns = ['Investment_Experience_Level', 'Debt_Level']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # IMPROVED FEATURE ENGINEERING
        # 1. Create interaction features
        df['Age_Income_Interaction'] = df['Age'] * df['Annual_Income']
        df['Savings_Income_Ratio'] = df['Current_Savings'] / (df['Annual_Income'] + 1)
        df['Contribution_Income_Ratio'] = (df['Contribution_Amount'] * 12) / (df['Annual_Income'] + 1)
        
        # 2. Create age groups
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 100], labels=[0, 1, 2, 3])
        df['Age_Group'] = df['Age_Group'].astype(int)
        
        # 3. Create income brackets
        df['Income_Bracket'] = pd.cut(df['Annual_Income'], 
                                     bins=[0, 50000, 75000, 100000, float('inf')], 
                                     labels=[0, 1, 2, 3])
        df['Income_Bracket'] = df['Income_Bracket'].astype(int)
        
        # 4. Create financial stability score
        df['Financial_Stability'] = (
            df['Savings_Income_Ratio'] * 0.4 +
            df['Contribution_Income_Ratio'] * 0.3 +
            df['Portfolio_Diversity_Score'] * 0.2 +
            (df['Years_Contributed'] / 10) * 0.1
        )
        
        return df
    
    def train_model(self):
        """Train improved risk prediction model"""
        print("Training Improved Risk Prediction Model...")
        
        # Load data
        df = self.load_and_preprocess_data()
        
        # IMPROVED FEATURES
        risk_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Years_Contributed', 'Investment_Experience_Level_encoded',
            'Portfolio_Diversity_Score', 'Savings_Rate', 'Debt_Level_encoded',
            # New engineered features
            'Age_Income_Interaction', 'Savings_Income_Ratio', 'Contribution_Income_Ratio',
            'Age_Group', 'Income_Bracket', 'Financial_Stability'
        ]
        
        # Prepare data
        risk_data = df[risk_features + ['Risk_Tolerance_encoded']].dropna()
        X = risk_data[risk_features]
        y = risk_data['Risk_Tolerance_encoded']
        
        # Check class distribution
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # IMPROVED MODEL: Random Forest with class balancing
        # Compute class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        # Use Random Forest instead of Logistic Regression
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight_dict,
            random_state=42
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Visualizations
        class_names = ['Low', 'Medium', 'High']
        
        # Confusion Matrix
        self.visualizer.plot_confusion_matrix(y_test, y_pred, "Improved Risk Prediction", class_names)
        
        # ROC Curve
        self.visualizer.plot_roc_curve(y_test, y_pred_proba, "Improved Risk Prediction", class_names)
        
        # Learning Curve
        self.visualizer.plot_learning_curve(self.model, X_train, y_train, "Improved Risk Prediction")
        
        # Feature Importance
        self.visualizer.plot_feature_importance(risk_features, self.model.feature_importances_, 
                                              "Improved Risk Prediction")
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"\n=== Improved Risk Prediction Model Metrics ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"CV Accuracy (mean ± std): {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\nFeature Importance (Top 10):")
        feature_importance = list(zip(risk_features, self.model.feature_importances_))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"  {i+1}. {feature}: {importance:.4f}")
        
        self.results['Improved Risk Prediction'] = metrics
        return metrics
    
    def save_model(self):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/improved_risk_prediction_model.pkl')
        
        print("Improved risk prediction model saved successfully!")

if __name__ == "__main__":
    trainer = ImprovedRiskPredictionTrainer()
    metrics = trainer.train_model()
    trainer.save_model()
    
    print("\n🎉 Improved Risk Prediction Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
