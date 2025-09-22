#!/usr/bin/env python3
"""
Improved Investment Recommendation Model Training
Addresses high MPE (67.5%) by implementing target transformation, stratified training, and alternative algorithms
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import joblib
import os
from ml_visualizer import MLVisualizer
from supabase_config import supabase, USER_PROFILES_TABLE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class ImprovedInvestmentRecommendationTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.target_transformer = None
        self.feature_selector = None
        self.visualizer = MLVisualizer()
        self.results = {}
        self.best_model_name = None
        self.stratified_models = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data with enhanced feature engineering"""
        print("Loading data for Improved Investment Recommendation Model...")
        
        # Fetch data from Supabase
        response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
        if not response.data:
            raise ValueError("No data found in Supabase database")
        
        df = pd.DataFrame(response.data)
        
        # Handle missing values with more sophisticated imputation
        df = df.fillna({
            'Age': df['Age'].median() if 'Age' in df.columns else 35,
            'Annual_Income': df['Annual_Income'].median() if 'Annual_Income' in df.columns else 50000,
            'Current_Savings': df['Current_Savings'].median() if 'Current_Savings' in df.columns else 10000,
            'Contribution_Amount': df['Contribution_Amount'].median() if 'Contribution_Amount' in df.columns else 500,
            'Years_Contributed': df['Years_Contributed'].median() if 'Years_Contributed' in df.columns else 5,
            'Savings_Rate': 0.1,
            'Portfolio_Diversity_Score': 0.5,
            'Risk_Tolerance': 'Medium',
            'Investment_Type': 'ETF',
            'Investment_Experience_Level': 'Beginner',
            'Annual_Return_Rate': 7.0,
            'Volatility': 2.0,
            'Fees_Percentage': 1.0,
            'Projected_Pension_Amount': df['Projected_Pension_Amount'].median() if 'Projected_Pension_Amount' in df.columns else 300000
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
        
        # Enhanced feature engineering
        self.create_enhanced_features(df)
        
        return df
    
    def create_enhanced_features(self, df):
        """Create sophisticated investment-related features"""
        
        # Basic ratios
        df['Savings_to_Income_Ratio'] = np.where(
            df['Annual_Income'] > 0,
            df['Current_Savings'] / df['Annual_Income'], 0
        )
        
        df['Contribution_Percent_of_Income'] = np.where(
            df['Annual_Income'] > 0,
            (df['Contribution_Amount'] * 12) / df['Annual_Income'], 0
        )
        
        # Time-based features
        df['Years_to_Retirement'] = np.maximum(0, 65 - df['Age'])
        df['Retirement_Horizon'] = pd.cut(df['Years_to_Retirement'], 
                                         bins=[0, 10, 20, 30, 100], 
                                         labels=[0, 1, 2, 3])
        df['Retirement_Horizon'] = df['Retirement_Horizon'].astype(int)
        
        # Investment capacity features
        df['Investment_Capacity'] = df['Annual_Income'] * df['Savings_Rate']
        df['Total_Annual_Contribution'] = df['Contribution_Amount'] * 12
        
        # Risk-adjusted features
        df['Risk_Adjusted_Return'] = np.where(
            df['Volatility'] > 0,
            df['Annual_Return_Rate'] / df['Volatility'], 0
        )
        
        df['Sharpe_Ratio'] = np.where(
            df['Volatility'] > 0,
            (df['Annual_Return_Rate'] - 2.0) / df['Volatility'], 0  # Assuming 2% risk-free rate
        )
        
        # Compound growth features
        df['Compound_Growth_Factor'] = (1 + df['Annual_Return_Rate'] / 100) ** df['Years_to_Retirement']
        df['Future_Value_Current_Savings'] = df['Current_Savings'] * df['Compound_Growth_Factor']
        
        # Interaction features
        df['Age_Income_Interaction'] = df['Age'] * df['Annual_Income'] / 1000000
        df['Savings_Experience_Interaction'] = df['Current_Savings'] * df['Investment_Experience_Level_encoded']
        df['Risk_Contribution_Interaction'] = df['Risk_Tolerance_encoded'] * df['Contribution_Amount']
        
        # Polynomial features
        df['Income_Squared'] = (df['Annual_Income'] / 100000) ** 2
        df['Savings_Squared'] = (df['Current_Savings'] / 10000) ** 2
        df['Age_Squared'] = (df['Age'] / 100) ** 2
        
        # Investment efficiency features
        df['Fees_Impact'] = df['Fees_Percentage'] * df['Current_Savings'] / 100
        df['Net_Return_Rate'] = df['Annual_Return_Rate'] - df['Fees_Percentage']
        
        # Portfolio quality indicators
        df['High_Diversity'] = (df['Portfolio_Diversity_Score'] > 0.7).astype(int)
        df['Low_Fees'] = (df['Fees_Percentage'] < 1.0).astype(int)
        df['High_Return'] = (df['Annual_Return_Rate'] > 8.0).astype(int)
        
        # Composite investment score
        df['Investment_Quality_Score'] = (
            df['High_Diversity'] * 0.3 +
            df['Low_Fees'] * 0.3 +
            df['High_Return'] * 0.2 +
            (df['Investment_Experience_Level_encoded'] / 3) * 0.2
        )
        
        print("Enhanced investment features created successfully")
    
    def create_savings_strata(self, df):
        """Create stratified groups based on current savings"""
        df['Savings_Strata'] = pd.cut(df['Current_Savings'], 
                                     bins=[0, 50000, 150000, 300000, float('inf')], 
                                     labels=['Low', 'Medium', 'High', 'Very_High'])
        return df
    
    def train_stratified_models(self, df):
        """Train separate models for different savings strata"""
        print("Training Stratified Investment Recommendation Models...")
        
        # Create strata
        df = self.create_savings_strata(df)
        
        # Features for investment recommendation
        investment_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Investment_Type_encoded', 'Annual_Return_Rate',
            'Volatility', 'Fees_Percentage', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded',
            # Enhanced features
            'Savings_to_Income_Ratio', 'Contribution_Percent_of_Income', 'Years_to_Retirement',
            'Retirement_Horizon', 'Investment_Capacity', 'Total_Annual_Contribution',
            'Risk_Adjusted_Return', 'Sharpe_Ratio', 'Compound_Growth_Factor',
            'Future_Value_Current_Savings', 'Age_Income_Interaction', 'Savings_Experience_Interaction',
            'Risk_Contribution_Interaction', 'Income_Squared', 'Savings_Squared', 'Age_Squared',
            'Fees_Impact', 'Net_Return_Rate', 'High_Diversity', 'Low_Fees', 'High_Return',
            'Investment_Quality_Score'
        ]
        
        # Prepare data
        investment_data = df[investment_features + ['Projected_Pension_Amount', 'Savings_Strata']].dropna()
        X = investment_data[investment_features]
        y = investment_data['Projected_Pension_Amount']
        strata = investment_data['Savings_Strata']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
        print(f"Strata distribution: {strata.value_counts().to_dict()}")
        
        # Train models for each stratum
        for stratum in ['Low', 'Medium', 'High', 'Very_High']:
            print(f"\nTraining model for {stratum} savings stratum...")
            
            stratum_mask = strata == stratum
            X_stratum = X[stratum_mask]
            y_stratum = y[stratum_mask]
            
            if len(X_stratum) < 10:  # Skip if too few samples
                print(f"  Skipping {stratum} stratum - too few samples ({len(X_stratum)})")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_stratum, y_stratum, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Apply target transformation (log transformation)
            y_train_transformed = np.log1p(y_train)  # log(1 + y) to handle zeros
            y_test_transformed = np.log1p(y_test)
            
            # Train XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train_transformed)
            
            # Predictions (transform back)
            y_pred_transformed = model.predict(X_test_scaled)
            y_pred = np.expm1(y_pred_transformed)  # exp(y) - 1
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  MAPE: {mape:.2%}")
            print(f"  R²: {r2:.4f}")
            
            # Store model
            self.stratified_models[stratum] = {
                'model': model,
                'scaler': scaler,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'y_test': y_test,
                'y_pred': y_pred
            }
    
    def train_multiple_models(self):
        """Train and compare multiple models with target transformation"""
        print("Training Multiple Investment Recommendation Models...")
        
        # Load data
        df = self.load_and_preprocess_data()
        
        # Enhanced features for investment recommendation
        investment_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Investment_Type_encoded', 'Annual_Return_Rate',
            'Volatility', 'Fees_Percentage', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded',
            # Enhanced features
            'Savings_to_Income_Ratio', 'Contribution_Percent_of_Income', 'Years_to_Retirement',
            'Retirement_Horizon', 'Investment_Capacity', 'Total_Annual_Contribution',
            'Risk_Adjusted_Return', 'Sharpe_Ratio', 'Compound_Growth_Factor',
            'Future_Value_Current_Savings', 'Age_Income_Interaction', 'Savings_Experience_Interaction',
            'Risk_Contribution_Interaction', 'Income_Squared', 'Savings_Squared', 'Age_Squared',
            'Fees_Impact', 'Net_Return_Rate', 'High_Diversity', 'Low_Fees', 'High_Return',
            'Investment_Quality_Score'
        ]
        
        # Prepare data
        investment_data = df[investment_features + ['Projected_Pension_Amount']].dropna()
        X = investment_data[investment_features]
        y = investment_data['Projected_Pension_Amount']
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
        
        # Analyze target distribution
        self.plot_target_distribution(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(f_regression, k=25)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        selected_features = [investment_features[i] for i in self.feature_selector.get_support(indices=True)]
        print(f"Selected features: {selected_features}")
        
        # Apply target transformation
        self.target_transformer = PowerTransformer(method='yeo-johnson')
        y_train_transformed = self.target_transformer.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_transformed = self.target_transformer.transform(y_test.values.reshape(-1, 1)).flatten()
        
        print(f"Transformed target distribution: min={y_train_transformed.min():.2f}, max={y_train_transformed.max():.2f}, mean={y_train_transformed.mean():.2f}")
        
        # Define models to test
        models = {
            'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model on transformed target
            model.fit(X_train_selected, y_train_transformed)
            
            # Predictions (transform back)
            y_pred_transformed = model.predict(X_test_selected)
            y_pred = self.target_transformer.inverse_transform(y_pred_transformed.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_selected, y_train_transformed, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'cv_r2_mean': cv_scores.mean(),
                'cv_r2_std': cv_scores.std(),
                'y_pred': y_pred
            }
            
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  MAPE: {mape:.2%}")
            print(f"  R²: {r2:.4f}")
            print(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['cv_r2_mean'])
        self.best_model_name = best_model_name
        self.model = results[best_model_name]['model']
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"CV R² Score: {results[best_model_name]['cv_r2_mean']:.4f}")
        print(f"MAPE: {results[best_model_name]['mape']:.2%}")
        
        # Visualizations
        self.plot_model_comparison(results, y_test)
        self.plot_residuals(results[best_model_name]['y_pred'], y_test, best_model_name)
        
        # Feature importance for tree-based models
        if best_model_name in ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting']:
            feature_names = selected_features
            importances = self.model.feature_importances_
            self.visualizer.plot_feature_importance(feature_names, importances, f"Investment Recommendation - {best_model_name}")
        
        # Learning curve
        self.visualizer.plot_learning_curve(self.model, X_train_selected, y_train_transformed, f"Investment Recommendation - {best_model_name}")
        
        metrics = {
            'best_model': best_model_name,
            'rmse': results[best_model_name]['rmse'],
            'mae': results[best_model_name]['mae'],
            'mape': results[best_model_name]['mape'],
            'r2': results[best_model_name]['r2'],
            'cv_r2_mean': results[best_model_name]['cv_r2_mean'],
            'cv_r2_std': results[best_model_name]['cv_r2_std'],
            'all_results': results
        }
        
        self.results['Improved Investment Recommendation'] = metrics
        return metrics
    
    def plot_target_distribution(self, y):
        """Plot target variable distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original distribution
        axes[0, 0].hist(y, bins=50, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Original Target Distribution')
        axes[0, 0].set_xlabel('Projected Pension Amount')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log transformation
        y_log = np.log1p(y)
        axes[0, 1].hist(y_log, bins=50, alpha=0.7, color='lightcoral')
        axes[0, 1].set_title('Log-Transformed Target Distribution')
        axes[0, 1].set_xlabel('Log(1 + Projected Pension Amount)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 0].boxplot(y)
        axes[1, 0].set_title('Target Distribution (Box Plot)')
        axes[1, 0].set_ylabel('Projected Pension Amount')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(y, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Target Variable')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/target_distribution_investment_recommendation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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
        
        # MAPE comparison
        mape_scores = [results[model]['mape'] for model in models]
        axes[1, 0].bar(models, mape_scores, alpha=0.7, color='lightgreen')
        axes[1, 0].set_title('MAPE Comparison')
        axes[1, 0].set_ylabel('MAPE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # RMSE comparison
        rmse_scores = [results[model]['rmse'] for model in models]
        axes[1, 1].bar(models, rmse_scores, alpha=0.7, color='gold')
        axes[1, 1].set_title('RMSE Comparison')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison_investment_recommendation.png', dpi=300, bbox_inches='tight')
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
        plt.savefig('visualizations/residual_analysis_investment_recommendation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        os.makedirs('models', exist_ok=True)
        
        joblib.dump(self.model, 'models/improved_investment_recommendation_model.pkl')
        joblib.dump(self.scaler, 'models/investment_recommendation_scaler.pkl')
        joblib.dump(self.target_transformer, 'models/investment_recommendation_transformer.pkl')
        joblib.dump(self.feature_selector, 'models/investment_recommendation_selector.pkl')
        joblib.dump(self.best_model_name, 'models/investment_recommendation_best_model.pkl')
        joblib.dump(self.stratified_models, 'models/investment_recommendation_stratified.pkl')
        
        print("Improved investment recommendation model and preprocessing objects saved successfully!")

if __name__ == "__main__":
    trainer = ImprovedInvestmentRecommendationTrainer()
    
    # Train stratified models
    df = trainer.load_and_preprocess_data()
    trainer.train_stratified_models(df)
    
    # Train main models
    metrics = trainer.train_multiple_models()
    trainer.save_model()
    
    print("\n🎉 Improved Investment Recommendation Model training complete!")
    print("Check the 'visualizations/' directory for plots.")
