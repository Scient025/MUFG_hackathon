#!/usr/bin/env python3
"""
Investment Recommendation Model - Complete Overhaul
Addresses R² = -1.29 by implementing:
- Target transformation (log, normalized ratios)
- Simpler baseline models first
- Domain-specific feature engineering
- Multiple algorithm comparison
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImprovedInvestmentRecommendationTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.target_transformer = PowerTransformer(method='yeo-johnson')
        self.label_encoders = {}
        self.feature_names = None
        self.models_dir = "models"
        self.visualizations_dir = "visualizations"
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.visualizations_dir, exist_ok=True)
    
    def load_data(self):
        """Load and prepare investment recommendation data"""
        print("Loading data for Improved Investment Recommendation Model...")
        
        # Load CSV data
        df = pd.read_csv('case1.csv')
        
        # Create investment recommendation target
        # Use Current_Savings as the target (what we want to recommend)
        target_column = 'Current_Savings'
        
        # Select features for investment recommendation
        feature_columns = [
            'Age', 'Annual_Income', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Contribution_Amount', 'Monthly_Expenses',
            'Debt_Level', 'Investment_Experience_Level_encoded'
        ]
        
        # Handle missing values for numeric columns only
        numeric_columns = [col for col in feature_columns if col != 'Debt_Level']
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        df[target_column] = df[target_column].fillna(df[target_column].median())
        
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Remove outliers (top and bottom 1%)
        q1 = y.quantile(0.01)
        q99 = y.quantile(0.99)
        mask = (y >= q1) & (y <= q99)
        X = X[mask]
        y = y[mask]
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")
        print(f"Target mean: ${y.mean():,.0f}")
        print(f"Target std: ${y.std():,.0f}")
        
        return X, y
    
    def create_domain_features(self, X, y=None):
        """Create domain-specific features for investment recommendation"""
        print("Creating domain-specific features...")
        
        X_enhanced = X.copy()
        
        # Calculate DTI ratio from available data
        X_enhanced['DTI_Ratio'] = X_enhanced['Monthly_Expenses'] * 12 / X_enhanced['Annual_Income']
        
        # Risk tolerance features
        X_enhanced['Age_Risk_Factor'] = np.where(X_enhanced['Age'] < 30, 1.2, 
                                                np.where(X_enhanced['Age'] < 50, 1.0, 0.8))
        
        # Income stability features (simplified)
        X_enhanced['Income_Stability'] = 1.0  # Assume stable for now
        
        # Debt burden features
        X_enhanced['Debt_Burden'] = np.where(X_enhanced['DTI_Ratio'] < 0.2, 1.0,
                                           np.where(X_enhanced['DTI_Ratio'] < 0.4, 0.8, 0.6))
        
        # Investment capacity features
        X_enhanced['Investment_Capacity'] = X_enhanced['Annual_Income'] * (1 - X_enhanced['DTI_Ratio'])
        X_enhanced['Investment_Ratio'] = X_enhanced['Contribution_Amount'] / X_enhanced['Annual_Income']
        
        # Portfolio diversity features
        X_enhanced['Diversity_Score_Normalized'] = X_enhanced['Portfolio_Diversity_Score'] / 10.0
        
        # Time horizon features
        X_enhanced['Time_Horizon'] = 65 - X_enhanced['Age']  # Years to retirement
        X_enhanced['Contribution_Consistency'] = X_enhanced['Years_Contributed'] / X_enhanced['Age']
        
        # Financial health composite score
        X_enhanced['Financial_Health_Score'] = (
            X_enhanced['Savings_Rate'] * 0.3 +
            (1 - X_enhanced['DTI_Ratio']) * 0.3 +
            X_enhanced['Portfolio_Diversity_Score'] * 0.2 +
            X_enhanced['Income_Stability'] * 0.2
        )
        
        # Investment aggressiveness score
        X_enhanced['Investment_Aggressiveness'] = (
            X_enhanced['Age_Risk_Factor'] * 0.4 +
            X_enhanced['Financial_Health_Score'] * 0.3 +
            X_enhanced['Diversity_Score_Normalized'] * 0.3
        )
        
        print(f"Enhanced features: {X_enhanced.shape[1]} features")
        return X_enhanced
    
    def train_multiple_models(self):
        """Train multiple models with different approaches"""
        X, y = self.load_data()
        
        # Create enhanced features
        X_enhanced = self.create_domain_features(X, y)
        
        # Encode categorical variables
        categorical_columns = ['Debt_Level']
        for col in categorical_columns:
            if col in X_enhanced.columns:
                le = LabelEncoder()
                X_enhanced[col] = le.fit_transform(X_enhanced[col].astype(str))
                self.label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_enhanced, y, test_size=0.2, random_state=42
        )
        
        self.feature_names = X_train.columns.tolist()
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Try different target transformations
        target_transformations = {
            'Original': (y_train, y_test),
            'Log_Transform': (np.log1p(y_train), np.log1p(y_test)),
            'Yeo_Johnson': (self.target_transformer.fit_transform(y_train.values.reshape(-1, 1)).flatten(),
                          self.target_transformer.transform(y_test.values.reshape(-1, 1)).flatten())
        }
        
        models = {
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        best_model = None
        best_r2 = -np.inf
        best_model_name = ""
        best_transformation = ""
        
        print("\n=== Model Comparison Across Target Transformations ===")
        
        for transform_name, (y_train_transformed, y_test_transformed) in target_transformations.items():
            print(f"\n--- {transform_name} Target Transformation ---")
            
            for model_name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train_scaled, y_train_transformed)
                    
                    # Predict
                    y_pred_transformed = model.predict(X_test_scaled)
                    
                    # Transform predictions back to original scale if needed
                    if transform_name == 'Log_Transform':
                        y_pred = np.expm1(y_pred_transformed)
                    elif transform_name == 'Yeo_Johnson':
                        y_pred = self.target_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()
                    else:
                        y_pred = y_pred_transformed
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    mape = mean_absolute_percentage_error(y_test, y_pred)
                    
                    print(f"{model_name:15} | R²: {r2:7.3f} | RMSE: ${rmse:8,.0f} | MAE: ${mae:8,.0f} | MAPE: {mape:6.1%}")
                    
                    # Track best model
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model
                        best_model_name = model_name
                        best_transformation = transform_name
                        best_y_pred = y_pred
                        best_y_test = y_test
                        
                except Exception as e:
                    print(f"{model_name:15} | Error: {str(e)[:50]}...")
        
        print(f"\n=== Best Model: {best_model_name} with {best_transformation} transformation ===")
        print(f"R² Score: {best_r2:.4f}")
        
        # Final evaluation
        final_mse = mean_squared_error(best_y_test, best_y_pred)
        final_rmse = np.sqrt(final_mse)
        final_r2 = r2_score(best_y_test, best_y_pred)
        final_mae = mean_absolute_error(best_y_test, best_y_pred)
        final_mape = mean_absolute_percentage_error(best_y_test, best_y_pred)
        
        print(f"\n=== Final Improved Investment Recommendation Model Metrics ===")
        print(f"Model: {best_model_name}")
        print(f"Transformation: {best_transformation}")
        print(f"R² Score: {final_r2:.4f}")
        print(f"RMSE: ${final_rmse:,.0f}")
        print(f"MAE: ${final_mae:,.0f}")
        print(f"MAPE: {final_mape:.2%}")
        
        # Cross-validation
        if best_transformation == 'Log_Transform':
            cv_y = np.log1p(y_train)
        elif best_transformation == 'Yeo_Johnson':
            cv_y = self.target_transformer.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        else:
            cv_y = y_train
        
        cv_scores = cross_val_score(best_model, X_train_scaled, cv_y, cv=5, scoring='r2')
        print(f"\nCross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Save model and components
        self.model = best_model
        
        model_path = os.path.join(self.models_dir, 'improved_investment_recommendation_model.pkl')
        scaler_path = os.path.join(self.models_dir, 'investment_recommendation_scaler.pkl')
        transformer_path = os.path.join(self.models_dir, 'investment_target_transformer.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.target_transformer, transformer_path)
        
        print(f"\nImproved investment recommendation model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")
        print(f"Target transformer saved to: {transformer_path}")
        
        # Create visualizations
        self.create_visualizations(best_y_test, best_y_pred, best_model_name, best_transformation)
        
        return {
            'r2_score': final_r2,
            'rmse': final_rmse,
            'mae': final_mae,
            'mape': final_mape,
            'model_name': best_model_name,
            'transformation': best_transformation,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
    
    def create_visualizations(self, y_test, y_pred, model_name, transformation):
        """Create visualization plots"""
        print("\nCreating visualizations...")
        
        # Prediction vs Actual scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Investment Amount ($)')
        plt.ylabel('Predicted Investment Amount ($)')
        plt.title(f'Improved Investment Recommendation - Predictions vs Actual\n{model_name} ({transformation})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'prediction_vs_actual_improved_investment.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Residual plot
        residuals = y_test - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, s=50)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Investment Amount ($)')
        plt.ylabel('Residuals ($)')
        plt.title(f'Improved Investment Recommendation - Residual Plot\n{model_name} ({transformation})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'residual_plot_improved_investment.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Error distribution
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals ($)')
        plt.ylabel('Frequency')
        plt.title(f'Improved Investment Recommendation - Error Distribution\n{model_name} ({transformation})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.visualizations_dir, 'error_distribution_improved_investment.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:15]
            
            plt.title(f'Improved Investment Recommendation - Feature Importance\n{model_name} ({transformation})')
            plt.bar(range(15), importances[indices])
            plt.xticks(range(15), [self.feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.visualizations_dir, 'feature_importance_improved_investment.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Visualizations saved to visualizations/ directory")

def main():
    """Main training execution"""
    print("Training Improved Investment Recommendation Model...")
    print("=" * 60)
    
    trainer = ImprovedInvestmentRecommendationTrainer()
    metrics = trainer.train_multiple_models()
    
    print("\n" + "=" * 60)
    print("Improved Investment Recommendation Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
    
    return metrics

if __name__ == "__main__":
    main()
