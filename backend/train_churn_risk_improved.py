#!/usr/bin/env python3
"""
Improved Churn Risk Model Training
Addresses low recall (66.7%) by implementing:
- Class weighting in XGBoost
- Threshold tuning for better recall
- SMOTE oversampling
- Balanced ensemble methods
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ImprovedChurnRiskTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.best_threshold = 0.5
        self.models_dir = "models"
        self.visualizations_dir = "visualizations"
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
    
    def load_data(self):
        """Load and prepare churn risk data"""
        print("Loading data for Improved Churn Risk Model...")
        
        # Load CSV data
        df = pd.read_csv('case1.csv')
        
        # Create churn risk target (simplified logic)
        # High risk if: low savings rate, high debt, or low contribution frequency
        df['Churn_Risk'] = 0  # Default: no churn risk
        
        # Calculate debt-to-income ratio from available data
        df['DTI_Ratio'] = df['Monthly_Expenses'] * 12 / df['Annual_Income']
        df['DTI_Ratio'] = df['DTI_Ratio'].fillna(df['DTI_Ratio'].median())
        
        # High churn risk conditions
        high_risk_conditions = (
            (df['Savings_Rate'] < 0.1) |  # Low savings rate
            (df['DTI_Ratio'] > 0.4) |     # High debt-to-income
            (df['Contribution_Frequency'] == 'Never')  # No contributions
        )
        
        df.loc[high_risk_conditions, 'Churn_Risk'] = 1
        
        # Select features for churn risk prediction
        feature_columns = [
            'Age', 'Annual_Income', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'DTI_Ratio', 'Debt_Level',
            'Monthly_Expenses', 'Investment_Experience_Level_encoded'
        ]
        
        # Handle missing values for numeric columns only
        numeric_columns = [col for col in feature_columns if col != 'Debt_Level']
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        X = df[feature_columns].copy()
        y = df['Churn_Risk'].copy()
        
        # Encode categorical variables
        categorical_columns = ['Debt_Level']
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
        
        self.feature_names = X.columns.tolist()
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Churn distribution: {y.value_counts().to_dict()}")
        print(f"Churn rate: {y.mean():.2%}")
        
        return X, y
    
    def train_model(self):
        """Train improved churn risk model with multiple strategies"""
        X, y = self.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("\n=== Training Multiple Churn Risk Models ===")
        
        # Strategy 1: Class-weighted XGBoost
        print("\n1. Training Class-Weighted XGBoost...")
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        
        xgb_weighted = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        
        xgb_weighted.fit(X_train_scaled, y_train)
        
        # Strategy 2: SMOTE + XGBoost
        print("2. Training SMOTE + XGBoost...")
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
        
        xgb_smote = xgb.XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
        
        xgb_smote.fit(X_resampled, y_resampled)
        
        # Strategy 3: Balanced Random Forest
        print("3. Training Balanced Random Forest...")
        brf = BalancedRandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        brf.fit(X_train_scaled, y_train)
        
        # Evaluate all models
        models = {
            'XGBoost_Weighted': xgb_weighted,
            'XGBoost_SMOTE': xgb_smote,
            'Balanced_RF': brf
        }
        
        best_model = None
        best_recall = 0
        best_model_name = ""
        
        print("\n=== Model Comparison ===")
        for name, model in models.items():
            # Get predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            print(f"\n{name}:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            # Track best recall
            if recall > best_recall:
                best_recall = recall
                best_model = model
                best_model_name = name
        
        # Threshold tuning for best model
        print(f"\n=== Threshold Tuning for {best_model_name} ===")
        y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1_threshold = 0
        
        print("Threshold | Recall | Precision | F1 Score")
        print("-" * 45)
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba_best >= threshold).astype(int)
            recall_thresh = recall_score(y_test, y_pred_thresh, zero_division=0)
            precision_thresh = precision_score(y_test, y_pred_thresh, zero_division=0)
            f1_thresh = f1_score(y_test, y_pred_thresh, zero_division=0)
            
            print(f"{threshold:8.2f} | {recall_thresh:6.3f} | {precision_thresh:9.3f} | {f1_thresh:8.3f}")
            
            # Choose threshold that maximizes F1 score
            if f1_thresh > best_f1_threshold:
                best_f1_threshold = f1_thresh
                best_threshold = threshold
        
        print(f"\nBest threshold: {best_threshold:.2f} (F1: {best_f1_threshold:.3f})")
        
        # Final evaluation with best threshold
        y_pred_final = (y_pred_proba_best >= best_threshold).astype(int)
        
        final_accuracy = accuracy_score(y_test, y_pred_final)
        final_precision = precision_score(y_test, y_pred_final, zero_division=0)
        final_recall = recall_score(y_test, y_pred_final, zero_division=0)
        final_f1 = f1_score(y_test, y_pred_final, zero_division=0)
        
        print(f"\n=== Final Improved Churn Risk Model Metrics ===")
        print(f"Model: {best_model_name}")
        print(f"Threshold: {best_threshold:.2f}")
        print(f"Accuracy: {final_accuracy:.4f}")
        print(f"Precision: {final_precision:.4f}")
        print(f"Recall: {final_recall:.4f}")
        print(f"F1 Score: {final_f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred_final, target_names=['No Churn', 'Churn Risk']))
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='recall')
        print(f"\nCross-validation Recall (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Save model and components
        self.model = best_model
        self.best_threshold = best_threshold
        
        model_path = os.path.join(self.models_dir, 'improved_churn_risk_model.pkl')
        threshold_path = os.path.join(self.models_dir, 'churn_risk_threshold.pkl')
        scaler_path = os.path.join(self.models_dir, 'churn_risk_scaler.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.best_threshold, threshold_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\nImproved churn risk model saved to: {model_path}")
        print(f"Best threshold saved to: {threshold_path}")
        print(f"Scaler saved to: {scaler_path}")
        
        # Create visualizations
        self.create_visualizations(y_test, y_pred_final, y_pred_proba_best, best_model_name)
        
        return {
            'accuracy': final_accuracy,
            'precision': final_precision,
            'recall': final_recall,
            'f1_score': final_f1,
            'best_threshold': best_threshold,
            'model_name': best_model_name,
            'cv_recall_mean': cv_scores.mean(),
            'cv_recall_std': cv_scores.std()
        }
    
    def create_visualizations(self, y_test, y_pred, y_pred_proba, model_name):
        """Create visualization plots"""
        print("\nCreating visualizations...")
        
        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn Risk'],
                   yticklabels=['No Churn', 'Churn Risk'])
        plt.title(f'Improved Churn Risk - Confusion Matrix ({model_name})')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'confusion_matrix_improved_churn_risk.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(10, 8))
            importances = self.model.feature_importances_
            n_features = min(10, len(importances))
            indices = np.argsort(importances)[::-1][:n_features]
            
            plt.title(f'Improved Churn Risk - Feature Importance ({model_name})')
            plt.bar(range(n_features), importances[indices])
            plt.xticks(range(n_features), [self.feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizations_dir, 'feature_importance_improved_churn_risk.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Threshold Analysis
        plt.figure(figsize=(10, 6))
        thresholds = np.arange(0.1, 0.9, 0.05)
        recalls = []
        precisions = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            recalls.append(recall_score(y_test, y_pred_thresh, zero_division=0))
            precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred_thresh, zero_division=0))
        
        plt.plot(thresholds, recalls, 'b-', label='Recall', linewidth=2)
        plt.plot(thresholds, precisions, 'r-', label='Precision', linewidth=2)
        plt.plot(thresholds, f1_scores, 'g-', label='F1 Score', linewidth=2)
        plt.axvline(x=self.best_threshold, color='k', linestyle='--', label=f'Best Threshold ({self.best_threshold:.2f})')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Improved Churn Risk - Threshold Analysis ({model_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'threshold_analysis_improved_churn_risk.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to visualizations/ directory")

def main():
    """Main training execution"""
    print("Training Improved Churn Risk Model...")
    print("=" * 50)
    
    trainer = ImprovedChurnRiskTrainer()
    metrics = trainer.train_model()
    
    print("\n" + "=" * 50)
    print("Improved Churn Risk Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
    
    return metrics

if __name__ == "__main__":
    main()
