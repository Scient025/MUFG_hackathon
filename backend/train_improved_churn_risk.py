#!/usr/bin/env python3
"""
Improved Churn Risk Model Training with Better Recall
Addresses the 57% recall issue by using class-weighted XGBoost and threshold optimization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os
from ml_visualizer import MLVisualizer
from supabase_config import supabase, USER_PROFILES_TABLE
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

class ImprovedChurnRiskTrainer:
    def __init__(self):
        self.model = None
        self.visualizer = MLVisualizer()
        self.results = {}
        self.best_threshold = 0.5
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for churn risk prediction"""
        print("Loading data for Improved Churn Risk Model...")
        
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
            'Debt_Level': 'Low',
            'Employment_Status': 'Full-time',
            'Investment_Experience_Level': 'Beginner',
            'Contribution_Frequency': 'Monthly'
        })
        
        # Convert numeric columns
        numeric_columns = ['Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
                          'Years_Contributed', 'Portfolio_Diversity_Score', 'Savings_Rate']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode categorical variables
        categorical_columns = ['Employment_Status', 'Debt_Level', 'Investment_Experience_Level', 'Contribution_Frequency']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # Create derived features for better churn prediction
        df['DTI_Ratio'] = np.where(
            df['Annual_Income'] > 0,
            df['Debt_Level'].map({'Low': 0.1, 'Medium': 0.3, 'High': 0.5, 'Unknown': 0.2}).fillna(0.2),
            np.nan
        )
        
        df['Contribution_Percent_of_Income'] = np.where(
            df['Annual_Income'] > 0,
            (df['Contribution_Amount'] * 12) / df['Annual_Income'], 0
        )
        
        # Enhanced churn indicators
        df['Low_Contribution_Flag'] = (df['Contribution_Amount'] < 500).astype(int)
        df['High_Debt_Flag'] = (df['DTI_Ratio'] > 0.4).astype(int)
        df['Low_Savings_Rate_Flag'] = (df['Savings_Rate'] < 0.05).astype(int)
        df['Young_Low_Contributor'] = ((df['Age'] < 30) & (df['Contribution_Amount'] < 500)).astype(int)
        
        return df
    
    def create_churn_label(self, row):
        """Enhanced churn label creation with more sophisticated logic"""
        churn_score = 0
        
        # Primary indicators
        if row['Contribution_Frequency_encoded'] == 0: churn_score += 2  # No contributions
        if row.get('Suspicious_Flag', 'No') == 'Yes': churn_score += 2  # Suspicious activity
        if row['Contribution_Percent_of_Income'] < 0.02: churn_score += 2  # Very low contribution rate
        if row['DTI_Ratio'] > 0.4: churn_score += 1  # High debt
        if row['Employment_Status_encoded'] == 0: churn_score += 1  # Unemployed
        if row['Savings_Rate'] < 0.05: churn_score += 1  # Low savings rate
        
        # Enhanced indicators
        if row['Low_Contribution_Flag']: churn_score += 1
        if row['High_Debt_Flag']: churn_score += 1
        if row['Low_Savings_Rate_Flag']: churn_score += 1
        if row['Young_Low_Contributor']: churn_score += 1
        
        # Portfolio diversity issues
        if row['Portfolio_Diversity_Score'] < 0.3: churn_score += 1
        
        # Return 1 if churn_score >= 3 (more sensitive threshold)
        return 1 if churn_score >= 3 else 0
    
    def optimize_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold to maximize F1 score"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        # Find threshold that maximizes F1 score
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        
        return best_threshold, f1_scores[best_idx]
    
    def train_model(self):
        """Train improved churn risk model with better recall"""
        print("Training Improved Churn Risk Model...")
        
        # Load data
        df = self.load_and_preprocess_data()
        
        # Enhanced features for churn prediction
        churn_features = [
            'Age', 'Annual_Income', 'Employment_Status_encoded', 'Debt_Level_encoded',
            'Contribution_Frequency_encoded', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded',
            'Contribution_Percent_of_Income', 'DTI_Ratio',
            # New engineered features
            'Low_Contribution_Flag', 'High_Debt_Flag', 'Low_Savings_Rate_Flag', 'Young_Low_Contributor'
        ]
        
        # Create target variable
        df['Churn_Risk'] = df.apply(self.create_churn_label, axis=1)
        
        # Prepare data
        churn_data = df[churn_features + ['Churn_Risk']].dropna()
        X = churn_data[churn_features]
        y = churn_data['Churn_Risk']
        
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = dict(zip(np.unique(y_train), class_weights))
        
        print(f"Class weights: {class_weight_dict}")
        
        # Train improved XGBoost model with class weights
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=class_weight_dict[1] / class_weight_dict[0],  # Handle class imbalance
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        # Get probabilities for threshold optimization
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Optimize threshold
        self.best_threshold, best_f1 = self.optimize_threshold(y_test, y_pred_proba)
        print(f"Optimal threshold: {self.best_threshold:.4f}")
        print(f"Best F1 score at optimal threshold: {best_f1:.4f}")
        
        # Predictions with optimal threshold
        y_pred_optimal = (y_pred_proba >= self.best_threshold).astype(int)
        y_pred_default = self.model.predict(X_test)
        
        # Calculate metrics for both thresholds
        metrics_default = {
            'accuracy': accuracy_score(y_test, y_pred_default),
            'precision': precision_score(y_test, y_pred_default, zero_division=0),
            'recall': recall_score(y_test, y_pred_default, zero_division=0),
            'f1': f1_score(y_test, y_pred_default, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        metrics_optimal = {
            'accuracy': accuracy_score(y_test, y_pred_optimal),
            'precision': precision_score(y_test, y_pred_optimal, zero_division=0),
            'recall': recall_score(y_test, y_pred_optimal, zero_division=0),
            'f1': f1_score(y_test, y_pred_optimal, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
        
        print(f"\n=== Improved Churn Risk Model Metrics ===")
        print(f"Default Threshold (0.5):")
        print(f"  Accuracy: {metrics_default['accuracy']:.4f}")
        print(f"  Precision: {metrics_default['precision']:.4f}")
        print(f"  Recall: {metrics_default['recall']:.4f}")
        print(f"  F1 Score: {metrics_default['f1']:.4f}")
        print(f"  AUC: {metrics_default['auc']:.4f}")
        
        print(f"\nOptimal Threshold ({self.best_threshold:.4f}):")
        print(f"  Accuracy: {metrics_optimal['accuracy']:.4f}")
        print(f"  Precision: {metrics_optimal['precision']:.4f}")
        print(f"  Recall: {metrics_optimal['recall']:.4f}")
        print(f"  F1 Score: {metrics_optimal['f1']:.4f}")
        print(f"  AUC: {metrics_optimal['auc']:.4f}")
        
        print(f"\nCV F1 Score (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Visualizations
        class_names = ['No Churn', 'Churn Risk']
        
        # Confusion Matrix
        self.visualizer.plot_confusion_matrix(y_test, y_pred_optimal, "Improved Churn Risk", class_names)
        
        # ROC Curve (only for binary classification)
        if len(np.unique(y_test)) == 2:
            self.visualizer.plot_roc_curve(y_test, y_pred_proba, "Improved Churn Risk", class_names)
        
        # Learning Curve
        self.visualizer.plot_learning_curve(self.model, X_train, y_train, "Improved Churn Risk")
        
        # Feature Importance
        self.visualizer.plot_feature_importance(churn_features, self.model.feature_importances_, 
                                              "Improved Churn Risk")
        
        # Precision-Recall Curve
        self.plot_precision_recall_curve(y_test, y_pred_proba)
        
        # Threshold optimization plot
        self.plot_threshold_optimization(y_test, y_pred_proba)
        
        metrics = {
            'default_threshold': metrics_default,
            'optimal_threshold': metrics_optimal,
            'best_threshold': self.best_threshold,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        }
        
        self.results['Improved Churn Risk'] = metrics
        return metrics
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba):
        """Plot precision-recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, label='Precision-Recall Curve')
        plt.axvline(x=recall[np.argmax(precision)], color='r', linestyle='--', 
                   label=f'Best Recall: {recall[np.argmax(precision)]:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Improved Churn Risk')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/precision_recall_curve_improved_churn_risk.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_threshold_optimization(self, y_true, y_pred_proba):
        """Plot threshold optimization"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
        plt.plot(thresholds, recall[:-1], 'r-', label='Recall')
        plt.plot(thresholds, f1_scores[:-1], 'g-', label='F1 Score')
        plt.axvline(x=self.best_threshold, color='k', linestyle='--', label=f'Optimal: {self.best_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Optimization')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='No Churn', color='blue')
        plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Churn Risk', color='red')
        plt.axvline(x=self.best_threshold, color='k', linestyle='--', label=f'Threshold: {self.best_threshold:.3f}')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 4)
        plt.plot(thresholds, f1_scores[:-1], 'g-', linewidth=2)
        plt.axvline(x=self.best_threshold, color='k', linestyle='--', label=f'Optimal: {self.best_threshold:.3f}')
        plt.xlabel('Threshold')
        plt.ylabel('F1 Score')
        plt.title('F1 Score vs Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/threshold_optimization_improved_churn_risk.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self):
        """Save the trained model and threshold"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/improved_churn_risk_model.pkl')
        joblib.dump(self.best_threshold, 'models/churn_risk_threshold.pkl')
        
        print("Improved churn risk model and threshold saved successfully!")

if __name__ == "__main__":
    trainer = ImprovedChurnRiskTrainer()
    metrics = trainer.train_model()
    trainer.save_model()
    
    print("\n🎉 Improved Churn Risk Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
