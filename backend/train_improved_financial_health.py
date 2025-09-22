#!/usr/bin/env python3
"""
Improved Financial Health Model Training
Addresses negative R² by implementing better feature engineering, nonlinear models, and regularization
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
import joblib
import os
from ml_visualizer import MLVisualizer
from supabase_config import supabase, USER_PROFILES_TABLE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class ImprovedFinancialHealthTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.visualizer = MLVisualizer()
        self.results = {}
        self.best_model_name = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess data with enhanced feature engineering"""
        print("Loading data for Improved Financial Health Model...")
        
        # Fetch data from Supabase
        response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
        if not response.data:
            raise ValueError("No data found in Supabase database")
        
        df = pd.DataFrame(response.data)
        
        # Handle missing values with more sophisticated imputation
        df = df.fillna({
            'Annual_Income': df['Annual_Income'].median() if 'Annual_Income' in df.columns else 50000,
            'Current_Savings': df['Current_Savings'].median() if 'Current_Savings' in df.columns else 10000,
            'Contribution_Amount': df['Contribution_Amount'].median() if 'Contribution_Amount' in df.columns else 500,
            'Years_Contributed': df['Years_Contributed'].median() if 'Years_Contributed' in df.columns else 5,
            'Age': df['Age'].median() if 'Age' in df.columns else 35,
            'Portfolio_Diversity_Score': 0.5,
            'Savings_Rate': 0.1,
            'Debt_Level': 'Low',
            'Investment_Experience_Level': 'Beginner',
            'Contribution_Frequency': 'Monthly',
            'Employment_Status': 'Full-time',
            'Marital_Status': 'Single',
            'Education_Level': "Bachelor's",
            'Health_Status': 'Average'
        })
        
        # Convert numeric columns
        numeric_columns = ['Annual_Income', 'Current_Savings', 'Contribution_Amount', 
                          'Years_Contributed', 'Portfolio_Diversity_Score', 'Savings_Rate', 'Age']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode categorical variables
        categorical_columns = ['Debt_Level', 'Investment_Experience_Level', 'Contribution_Frequency',
                              'Employment_Status', 'Marital_Status', 'Education_Level', 'Health_Status']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # Enhanced feature engineering
        self.create_enhanced_features(df)
        
        return df
    
    def create_enhanced_features(self, df):
        """Create sophisticated financial health features"""
        
        # Basic ratios
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
        
        # Advanced financial ratios
        df['Emergency_Fund_Ratio'] = np.where(
            df['Annual_Income'] > 0,
            df['Current_Savings'] / (df['Annual_Income'] / 12 * 3), 0  # 3 months of expenses
        )
        
        df['Retirement_Savings_Rate'] = np.where(
            df['Age'] > 0,
            df['Current_Savings'] / (df['Age'] * df['Annual_Income']), 0
        )
        
        df['Income_Stability_Score'] = np.where(
            df['Employment_Status_encoded'] == 0, 0.5,  # Unemployed
            np.where(df['Employment_Status_encoded'] == 1, 1.0,  # Full-time
            np.where(df['Employment_Status_encoded'] == 2, 0.8, 0.6))  # Part-time, Other
        )
        
        # Age-based features
        df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 55, 100], labels=[0, 1, 2, 3, 4])
        df['Age_Group'] = df['Age_Group'].astype(int)
        
        df['Years_to_Retirement'] = np.maximum(0, 65 - df['Age'])
        
        # Income brackets
        df['Income_Bracket'] = pd.cut(df['Annual_Income'], 
                                    bins=[0, 30000, 50000, 75000, 100000, float('inf')], 
                                    labels=[0, 1, 2, 3, 4])
        df['Income_Bracket'] = df['Income_Bracket'].astype(int)
        
        # Interaction features
        df['Age_Income_Interaction'] = df['Age'] * df['Annual_Income'] / 1000000
        df['Savings_Experience_Interaction'] = df['Current_Savings'] * df['Investment_Experience_Level_encoded']
        df['Contribution_Years_Interaction'] = df['Contribution_Amount'] * df['Years_Contributed']
        
        # Polynomial features for non-linear relationships
        df['Income_Squared'] = (df['Annual_Income'] / 100000) ** 2
        df['Savings_Squared'] = (df['Current_Savings'] / 10000) ** 2
        df['Age_Squared'] = (df['Age'] / 100) ** 2
        
        # Financial health indicators
        df['High_Savings_Rate'] = (df['Savings_Rate'] > 0.15).astype(int)
        df['Low_Debt'] = (df['DTI_Ratio'] < 0.2).astype(int)
        df['Diversified_Portfolio'] = (df['Portfolio_Diversity_Score'] > 0.7).astype(int)
        df['Experienced_Investor'] = (df['Investment_Experience_Level_encoded'] >= 2).astype(int)
        
        # Composite scores
        df['Financial_Discipline_Score'] = (
            df['High_Savings_Rate'] * 0.3 +
            df['Low_Debt'] * 0.3 +
            df['Diversified_Portfolio'] * 0.2 +
            df['Experienced_Investor'] * 0.2
        )
        
        df['Income_Quality_Score'] = (
            df['Income_Stability_Score'] * 0.4 +
            (df['Income_Bracket'] / 4) * 0.3 +
            (df['Education_Level_encoded'] / 3) * 0.3
        )
        
        print("Enhanced features created successfully")
    
    def calculate_enhanced_health_score(self, row):
        """Enhanced financial health score calculation with more sophisticated logic"""
        score = 0
        
        # Income component (25 points) - more nuanced
        income = row['Annual_Income']
        if income > 100000: score += 25
        elif income > 75000: score += 22
        elif income > 50000: score += 18
        elif income > 30000: score += 12
        else: score += 6
        
        # Savings component (30 points) - enhanced
        savings_ratio = row['Savings_to_Income_Ratio']
        emergency_ratio = row['Emergency_Fund_Ratio']
        
        if savings_ratio > 2.0: score += 20
        elif savings_ratio > 1.0: score += 16
        elif savings_ratio > 0.5: score += 12
        elif savings_ratio > 0.2: score += 8
        else: score += 4
        
        if emergency_ratio > 1.0: score += 10
        elif emergency_ratio > 0.5: score += 7
        elif emergency_ratio > 0.25: score += 4
        else: score += 1
        
        # Contribution component (20 points)
        contrib_ratio = row['Contribution_Percent_of_Income']
        if contrib_ratio > 0.15: score += 20
        elif contrib_ratio > 0.10: score += 16
        elif contrib_ratio > 0.05: score += 12
        elif contrib_ratio > 0.02: score += 8
        else: score += 4
        
        # Debt component (15 points)
        dti = row['DTI_Ratio']
        if dti < 0.2: score += 15
        elif dti < 0.3: score += 12
        elif dti < 0.4: score += 8
        elif dti < 0.5: score += 4
        else: score += 1
        
        # Portfolio and experience (10 points)
        diversity = row['Portfolio_Diversity_Score']
        if diversity > 0.8: score += 6
        elif diversity > 0.6: score += 4
        elif diversity > 0.4: score += 2
        else: score += 1
        
        experience = row['Investment_Experience_Level_encoded']
        if experience >= 2: score += 4
        elif experience >= 1: score += 3
        else: score += 2
        
        return min(100, max(0, score))
    
    def train_multiple_models(self):
        """Train and compare multiple models"""
        print("Training Multiple Financial Health Models...")
        
        # Load data
        df = self.load_and_preprocess_data()
        
        # Enhanced features for financial health
        health_features = [
            'Annual_Income', 'Current_Savings', 'Savings_Rate', 'Debt_Level_encoded',
            'Portfolio_Diversity_Score', 'Contribution_Amount', 'Contribution_Frequency_encoded',
            'Years_Contributed', 'DTI_Ratio', 'Savings_to_Income_Ratio',
            'Contribution_Percent_of_Income', 'Age', 'Investment_Experience_Level_encoded',
            # Enhanced features
            'Emergency_Fund_Ratio', 'Retirement_Savings_Rate', 'Income_Stability_Score',
            'Age_Group', 'Income_Bracket', 'Years_to_Retirement',
            'Age_Income_Interaction', 'Savings_Experience_Interaction', 'Contribution_Years_Interaction',
            'Income_Squared', 'Savings_Squared', 'Age_Squared',
            'High_Savings_Rate', 'Low_Debt', 'Diversified_Portfolio', 'Experienced_Investor',
            'Financial_Discipline_Score', 'Income_Quality_Score'
        ]
        
        # Create target variable
        df['Financial_Health_Score'] = df.apply(self.calculate_enhanced_health_score, axis=1)
        
        # Prepare data
        health_data = df[health_features + ['Financial_Health_Score']].dropna()
        X = health_data[health_features]
        y = health_data['Financial_Health_Score']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(f_regression, k=20)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        selected_features = [health_features[i] for i in self.feature_selector.get_support(indices=True)]
        print(f"Selected features: {selected_features}")
        
        # Define models to test
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name in ['Ridge', 'Lasso', 'ElasticNet']:
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            if name in ['Ridge', 'Lasso', 'ElasticNet']:
                cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'y_pred': y_pred
            }
            
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_r2_mean'])
        self.best_model_name = best_model_name
        self.model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"CV R² Score: {results[best_model_name]['cv_r2_mean']:.4f}")
        
        # Visualizations
        self.plot_model_comparison(results, y_test)
        self.plot_residuals(results[best_model_name]['y_pred'], y_test, best_model_name)
        
        # Feature importance for tree-based models
        if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting']:
            if best_model_name in ['Ridge', 'Lasso', 'ElasticNet']:
                feature_names = selected_features
                importances = np.abs(self.model.coef_)
            else:
                feature_names = health_features
                importances = self.model.feature_importances_
            
            self.visualizer.plot_feature_importance(feature_names, importances, f"Financial Health - {best_model_name}")
        
        # Learning curve
        self.visualizer.plot_learning_curve(self.model, X_train, y_train, f"Financial Health - {best_model_name}")
        
        metrics = {
            'best_model': best_model_name,
            'rmse': results[best_model_name]['rmse'],
            'mae': results[best_model_name]['mae'],
            'r2': results[best_model_name]['r2'],
            'cv_r2_mean': results[best_model_name]['cv_r2_mean'],
            'cv_r2_std': results[best_model_name]['cv_r2_std'],
            'all_results': results
        }
        
        self.results['Improved Financial Health'] = metrics
        return metrics
    
    def plot_model_comparison(self, results, y_test):
        """Plot comparison of all models"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² comparison
        models = list(results.keys())
        r2_scores = [results[model]['r2'] for model in models]
        cv_r2_scores = [results[model]['cv_r2_mean'] for model in models]
        
        axes[0, 0].bar(models, r2_scores, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('R² Score Comparison')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        axes[0, 1].bar(models, cv_r2_scores, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Cross-Validation R² Score Comparison')
        axes[0, 1].set_ylabel('CV R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        rmse_scores = [results[model]['rmse'] for model in models]
        axes[1, 0].bar(models, rmse_scores, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('RMSE Comparison')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        mae_scores = [results[model]['mae'] for model in models]
        axes[1, 1].bar(models, mae_scores, alpha=0.7, color='gold')
        axes[1, 1].set_title('MAE Comparison')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison_financial_health.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_residuals(self, y_pred, y_test, model_name):
        """Plot residual analysis"""
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        axes[1, 1].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig('visualizations/residual_analysis_financial_health.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/improved_financial_health_model.pkl')
        joblib.dump(self.scaler, 'models/financial_health_scaler.pkl')
        joblib.dump(self.feature_selector, 'models/financial_health_selector.pkl')
        joblib.dump(self.best_model_name, 'models/financial_health_best_model.pkl')
        
        print("Improved financial health model and preprocessing objects saved successfully!")

if __name__ == "__main__":
    trainer = ImprovedFinancialHealthTrainer()
    metrics = trainer.train_multiple_models()
    trainer.save_model()
    
    print("\n🎉 Improved Financial Health Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
