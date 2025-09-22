#!/usr/bin/env python3
"""
Comprehensive ML Pipeline with Cross-Validation, Hyperparameter Optimization, and Unified Evaluation
Implements standardized evaluation framework across all models
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, KFold,
    GridSearchCV, RandomizedSearchCV, validation_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error,
    classification_report, confusion_matrix
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from ml_visualizer import MLVisualizer
from supabase_config import supabase, USER_PROFILES_TABLE
import warnings
warnings.filterwarnings('ignore')

class UnifiedMLPipeline:
    def __init__(self):
        self.visualizer = MLVisualizer()
        self.results = {}
        self.best_models = {}
        self.scalers = {}
        self.evaluation_metrics = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess data for all models"""
        print("Loading data for Unified ML Pipeline...")
        
        # Fetch data from Supabase
        response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
        if not response.data:
            raise ValueError("No data found in Supabase database")
        
        df = pd.DataFrame(response.data)
        
        # Handle missing values
        df = df.fillna({
            'Age': 30, 'Annual_Income': 0, 'Current_Savings': 0, 'Contribution_Amount': 0,
            'Years_Contributed': 0, 'Portfolio_Diversity_Score': 0.5, 'Savings_Rate': 0.1,
            'Debt_Level': 'Low', 'Investment_Experience_Level': 'Beginner', 'Contribution_Frequency': 'Monthly',
            'Employment_Status': 'Full-time', 'Risk_Tolerance': 'Medium', 'Investment_Type': 'ETF',
            'Annual_Return_Rate': 7.0, 'Volatility': 2.0, 'Fees_Percentage': 1.0,
            'Projected_Pension_Amount': 0
        })
        
        # Convert numeric columns
        numeric_columns = ['Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
                          'Years_Contributed', 'Portfolio_Diversity_Score', 'Savings_Rate',
                          'Annual_Return_Rate', 'Volatility', 'Fees_Percentage', 'Projected_Pension_Amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Encode categorical variables
        categorical_columns = ['Debt_Level', 'Investment_Experience_Level', 'Contribution_Frequency',
                              'Employment_Status', 'Risk_Tolerance', 'Investment_Type']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        
        # Create derived features
        self.create_derived_features(df)
        
        return df
    
    def create_derived_features(self, df):
        """Create derived features for all models"""
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
        
        df['Risk_Adjusted_Return'] = np.where(
            df['Volatility'] > 0,
            df['Annual_Return_Rate'] / df['Volatility'], 0
        )
        
        # Enhanced features
        df['Age_Income_Interaction'] = df['Age'] * df['Annual_Income'] / 1000000
        df['Savings_Experience_Interaction'] = df['Current_Savings'] * df['Investment_Experience_Level_encoded']
        df['Years_to_Retirement'] = np.maximum(0, 65 - df['Age'])
        
        print("Derived features created successfully")
    
    def hyperparameter_optimization(self, model, param_grid, X, y, cv=5, scoring='accuracy'):
        """Perform hyperparameter optimization using GridSearchCV"""
        print(f"Performing hyperparameter optimization for {type(model).__name__}...")
        
        # Use RandomizedSearchCV for large parameter spaces
        if len(param_grid) > 20:
            search = RandomizedSearchCV(
                model, param_grid, n_iter=50, cv=cv, scoring=scoring,
                random_state=42, n_jobs=-1
            )
        else:
            search = GridSearchCV(
                model, param_grid, cv=cv, scoring=scoring,
                n_jobs=-1
            )
        
        search.fit(X, y)
        
        print(f"Best parameters: {search.best_params_}")
        print(f"Best score: {search.best_score_:.4f}")
        
        return search.best_estimator_, search.best_params_, search.best_score_
    
    def train_risk_prediction_model(self, df):
        """Train optimized risk prediction model"""
        print("\n🎯 Training Optimized Risk Prediction Model...")
        
        # Features for risk prediction
        risk_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Years_Contributed', 'Investment_Experience_Level_encoded', 'Portfolio_Diversity_Score',
            'Savings_Rate', 'Debt_Level_encoded', 'DTI_Ratio', 'Savings_to_Income_Ratio',
            'Contribution_Percent_of_Income', 'Age_Income_Interaction', 'Savings_Experience_Interaction'
        ]
        
        # Prepare data
        risk_data = df[risk_features + ['Risk_Tolerance_encoded']].dropna()
        X = risk_data[risk_features]
        y = risk_data['Risk_Tolerance_encoded']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models and parameter grids
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'Neural Network': MLPClassifier(random_state=42)
        }
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'poly', 'sigmoid']
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000]
            }
        }
        
        best_model = None
        best_score = 0
        best_name = None
        
        for name, model in models.items():
            print(f"\nOptimizing {name}...")
            
            # Optimize hyperparameters
            optimized_model, best_params, best_score_cv = self.hyperparameter_optimization(
                model, param_grids[name], X_train_scaled, y_train, cv=5, scoring='f1_weighted'
            )
            
            # Evaluate on test set
            y_pred = optimized_model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"  CV Score: {best_score_cv:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test F1: {test_f1:.4f}")
            
            if test_f1 > best_score:
                best_score = test_f1
                best_model = optimized_model
                best_name = name
        
        # Store results
        self.best_models['risk_prediction'] = best_model
        self.scalers['risk_prediction'] = scaler
        self.evaluation_metrics['risk_prediction'] = {
            'model_name': best_name,
            'test_accuracy': accuracy_score(y_test, best_model.predict(X_test_scaled)),
            'test_f1': f1_score(y_test, best_model.predict(X_test_scaled), average='weighted'),
            'test_precision': precision_score(y_test, best_model.predict(X_test_scaled), average='weighted'),
            'test_recall': recall_score(y_test, best_model.predict(X_test_scaled), average='weighted')
        }
        
        print(f"\n🏆 Best Risk Prediction Model: {best_name}")
        print(f"Test F1 Score: {best_score:.4f}")
        
        return best_model, scaler
    
    def train_churn_risk_model(self, df):
        """Train optimized churn risk model"""
        print("\n🎯 Training Optimized Churn Risk Model...")
        
        # Create churn labels
        def create_churn_label(row):
            churn_score = 0
            if row['Contribution_Frequency_encoded'] == 0: churn_score += 1
            if row['Contribution_Percent_of_Income'] < 0.02: churn_score += 1
            if row['DTI_Ratio'] > 0.4: churn_score += 1
            if row['Employment_Status_encoded'] == 0: churn_score += 1
            if row['Savings_Rate'] < 0.05: churn_score += 1
            return 1 if churn_score >= 2 else 0
        
        df['Churn_Risk'] = df.apply(create_churn_label, axis=1)
        
        # Features for churn prediction
        churn_features = [
            'Age', 'Annual_Income', 'Employment_Status_encoded', 'Debt_Level_encoded',
            'Contribution_Frequency_encoded', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded',
            'Contribution_Percent_of_Income', 'DTI_Ratio', 'Savings_to_Income_Ratio'
        ]
        
        # Prepare data
        churn_data = df[churn_features + ['Churn_Risk']].dropna()
        X = churn_data[churn_features]
        y = churn_data['Churn_Risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models and parameter grids
        models = {
            'XGBoost': xgb.XGBClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'LightGBM': lgb.LGBMClassifier(random_state=42),
            'Neural Network': MLPClassifier(random_state=42)
        }
        
        param_grids = {
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'scale_pos_weight': [1, 2, 3]  # Handle class imbalance
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'class_weight': ['balanced', None]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'class_weight': ['balanced', None]
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        best_model = None
        best_score = 0
        best_name = None
        
        for name, model in models.items():
            print(f"\nOptimizing {name}...")
            
            # Optimize hyperparameters
            optimized_model, best_params, best_score_cv = self.hyperparameter_optimization(
                model, param_grids[name], X_train_scaled, y_train, cv=5, scoring='f1'
            )
            
            # Evaluate on test set
            y_pred = optimized_model.predict(X_test_scaled)
            test_accuracy = accuracy_score(y_test, y_pred)
            test_f1 = f1_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred)
            test_recall = recall_score(y_test, y_pred)
            
            print(f"  CV Score: {best_score_cv:.4f}")
            print(f"  Test Accuracy: {test_accuracy:.4f}")
            print(f"  Test F1: {test_f1:.4f}")
            print(f"  Test Precision: {test_precision:.4f}")
            print(f"  Test Recall: {test_recall:.4f}")
            
            if test_f1 > best_score:
                best_score = test_f1
                best_model = optimized_model
                best_name = name
        
        # Store results
        self.best_models['churn_risk'] = best_model
        self.scalers['churn_risk'] = scaler
        self.evaluation_metrics['churn_risk'] = {
            'model_name': best_name,
            'test_accuracy': accuracy_score(y_test, best_model.predict(X_test_scaled)),
            'test_f1': f1_score(y_test, best_model.predict(X_test_scaled)),
            'test_precision': precision_score(y_test, best_model.predict(X_test_scaled)),
            'test_recall': recall_score(y_test, best_model.predict(X_test_scaled))
        }
        
        print(f"\n🏆 Best Churn Risk Model: {best_name}")
        print(f"Test F1 Score: {best_score:.4f}")
        
        return best_model, scaler
    
    def train_financial_health_model(self, df):
        """Train optimized financial health model"""
        print("\n🎯 Training Optimized Financial Health Model...")
        
        # Calculate financial health score
        def calculate_health_score(row):
            score = 0
            if row['Annual_Income'] > 100000: score += 20
            elif row['Annual_Income'] > 75000: score += 15
            elif row['Annual_Income'] > 50000: score += 10
            else: score += 5
            
            savings_ratio = row['Savings_to_Income_Ratio']
            if savings_ratio > 2.0: score += 25
            elif savings_ratio > 1.0: score += 20
            elif savings_ratio > 0.5: score += 15
            elif savings_ratio > 0.2: score += 10
            else: score += 5
            
            contrib_ratio = row['Contribution_Percent_of_Income']
            if contrib_ratio > 0.15: score += 20
            elif contrib_ratio > 0.10: score += 15
            elif contrib_ratio > 0.05: score += 10
            else: score += 5
            
            dti = row['DTI_Ratio']
            if dti < 0.2: score += 15
            elif dti < 0.3: score += 12
            elif dti < 0.4: score += 8
            else: score += 3
            
            diversity = row['Portfolio_Diversity_Score']
            if diversity > 0.8: score += 10
            elif diversity > 0.6: score += 8
            elif diversity > 0.4: score += 5
            else: score += 2
            
            experience = row['Investment_Experience_Level_encoded']
            if experience >= 2: score += 10
            elif experience >= 1: score += 7
            else: score += 4
            
            return min(100, max(0, score))
        
        df['Financial_Health_Score'] = df.apply(calculate_health_score, axis=1)
        
        # Features for financial health
        health_features = [
            'Annual_Income', 'Current_Savings', 'Savings_Rate', 'Debt_Level_encoded',
            'Portfolio_Diversity_Score', 'Contribution_Amount', 'Contribution_Frequency_encoded',
            'Years_Contributed', 'DTI_Ratio', 'Savings_to_Income_Ratio',
            'Contribution_Percent_of_Income', 'Risk_Adjusted_Return', 'Age',
            'Investment_Experience_Level_encoded', 'Age_Income_Interaction', 'Savings_Experience_Interaction'
        ]
        
        # Prepare data
        health_data = df[health_features + ['Financial_Health_Score']].dropna()
        X = health_data[health_features]
        y = health_data['Financial_Health_Score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Define models and parameter grids
        models = {
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42),
            'Neural Network': MLPRegressor(random_state=42)
        }
        
        param_grids = {
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Neural Network': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000]
            }
        }
        
        best_model = None
        best_score = 0
        best_name = None
        
        for name, model in models.items():
            print(f"\nOptimizing {name}...")
            
            # Optimize hyperparameters
            optimized_model, best_params, best_score_cv = self.hyperparameter_optimization(
                model, param_grids[name], X_train_scaled, y_train, cv=5, scoring='r2'
            )
            
            # Evaluate on test set
            y_pred = optimized_model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mae = mean_absolute_error(y_test, y_pred)
            
            print(f"  CV Score: {best_score_cv:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.2f}")
            print(f"  Test MAE: {test_mae:.2f}")
            
            if test_r2 > best_score:
                best_score = test_r2
                best_model = optimized_model
                best_name = name
        
        # Store results
        self.best_models['financial_health'] = best_model
        self.scalers['financial_health'] = scaler
        self.evaluation_metrics['financial_health'] = {
            'model_name': best_name,
            'test_r2': r2_score(y_test, best_model.predict(X_test_scaled)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, best_model.predict(X_test_scaled))),
            'test_mae': mean_absolute_error(y_test, best_model.predict(X_test_scaled))
        }
        
        print(f"\n🏆 Best Financial Health Model: {best_name}")
        print(f"Test R² Score: {best_score:.4f}")
        
        return best_model, scaler
    
    def train_user_segmentation_model(self, df):
        """Train optimized user segmentation model"""
        print("\n🎯 Training Optimized User Segmentation Model...")
        
        # Features for clustering
        clustering_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Years_Contributed', 'Portfolio_Diversity_Score',
            'Savings_Rate', 'DTI_Ratio', 'Savings_to_Income_Ratio', 'Contribution_Percent_of_Income'
        ]
        
        # Prepare data
        clustering_data = df[clustering_features].dropna()
        X = clustering_data[clustering_features]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Define models and parameter grids
        models = {
            'KMeans': KMeans(random_state=42),
            'Gaussian Mixture': GaussianMixture(random_state=42),
            'DBSCAN': DBSCAN()
        }
        
        param_grids = {
            'KMeans': {
                'n_clusters': [3, 4, 5, 6, 7],
                'init': ['k-means++', 'random'],
                'n_init': [10, 20]
            },
            'Gaussian Mixture': {
                'n_components': [3, 4, 5, 6, 7],
                'covariance_type': ['full', 'tied', 'diag', 'spherical']
            },
            'DBSCAN': {
                'eps': [0.3, 0.5, 0.7, 1.0],
                'min_samples': [3, 5, 10, 15]
            }
        }
        
        best_model = None
        best_score = 0
        best_name = None
        
        for name, model in models.items():
            print(f"\nOptimizing {name}...")
            
            if name == 'DBSCAN':
                # DBSCAN doesn't have a single score, use silhouette score
                from sklearn.metrics import silhouette_score
                best_score_cv = 0
                best_params = {}
                
                for eps in param_grids[name]['eps']:
                    for min_samples in param_grids[name]['min_samples']:
                        model_temp = DBSCAN(eps=eps, min_samples=min_samples)
                        labels = model_temp.fit_predict(X_scaled)
                        
                        if len(set(labels)) > 1:  # More than one cluster
                            score = silhouette_score(X_scaled, labels)
                            if score > best_score_cv:
                                best_score_cv = score
                                best_params = {'eps': eps, 'min_samples': min_samples}
                
                optimized_model = DBSCAN(**best_params)
                optimized_model.fit(X_scaled)
            else:
                # Optimize hyperparameters
                optimized_model, best_params, best_score_cv = self.hyperparameter_optimization(
                    model, param_grids[name], X_scaled, None, cv=5, scoring='silhouette_score'
                )
            
            print(f"  Best Score: {best_score_cv:.4f}")
            
            if best_score_cv > best_score:
                best_score = best_score_cv
                best_model = optimized_model
                best_name = name
        
        # Store results
        self.best_models['user_segmentation'] = best_model
        self.scalers['user_segmentation'] = scaler
        self.evaluation_metrics['user_segmentation'] = {
            'model_name': best_name,
            'silhouette_score': best_score
        }
        
        print(f"\n🏆 Best User Segmentation Model: {best_name}")
        print(f"Silhouette Score: {best_score:.4f}")
        
        return best_model, scaler
    
    def generate_comprehensive_report(self):
        """Generate comprehensive evaluation report"""
        print("\n📊 COMPREHENSIVE MODEL EVALUATION REPORT")
        print("="*80)
        
        # Model performance summary
        print("\n🏆 BEST MODELS SUMMARY:")
        print("-" * 80)
        
        for model_name, metrics in self.evaluation_metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Best Algorithm: {metrics['model_name']}")
            
            if 'test_accuracy' in metrics:
                print(f"  Accuracy: {metrics['test_accuracy']:.4f}")
            if 'test_f1' in metrics:
                print(f"  F1 Score: {metrics['test_f1']:.4f}")
            if 'test_precision' in metrics:
                print(f"  Precision: {metrics['test_precision']:.4f}")
            if 'test_recall' in metrics:
                print(f"  Recall: {metrics['test_recall']:.4f}")
            if 'test_r2' in metrics:
                print(f"  R² Score: {metrics['test_r2']:.4f}")
            if 'test_rmse' in metrics:
                print(f"  RMSE: {metrics['test_rmse']:.2f}")
            if 'test_mae' in metrics:
                print(f"  MAE: {metrics['test_mae']:.2f}")
            if 'silhouette_score' in metrics:
                print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        
        # Create comparison plots
        self.plot_model_comparison()
        
        # Production readiness assessment
        print("\n🚀 PRODUCTION READINESS ASSESSMENT:")
        print("-" * 80)
        
        production_ready = []
        needs_improvement = []
        
        for model_name, metrics in self.evaluation_metrics.items():
            if model_name == 'risk_prediction':
                if metrics['test_f1'] > 0.8:
                    production_ready.append(f"✅ {model_name.upper()}: Ready (F1: {metrics['test_f1']:.3f})")
                else:
                    needs_improvement.append(f"⚠️ {model_name.upper()}: Needs improvement (F1: {metrics['test_f1']:.3f})")
            
            elif model_name == 'churn_risk':
                if metrics['test_f1'] > 0.7 and metrics['test_recall'] > 0.6:
                    production_ready.append(f"✅ {model_name.upper()}: Ready (F1: {metrics['test_f1']:.3f}, Recall: {metrics['test_recall']:.3f})")
                else:
                    needs_improvement.append(f"⚠️ {model_name.upper()}: Needs improvement (F1: {metrics['test_f1']:.3f}, Recall: {metrics['test_recall']:.3f})")
            
            elif model_name == 'financial_health':
                if metrics['test_r2'] > 0.7:
                    production_ready.append(f"✅ {model_name.upper()}: Ready (R²: {metrics['test_r2']:.3f})")
                else:
                    needs_improvement.append(f"⚠️ {model_name.upper()}: Needs improvement (R²: {metrics['test_r2']:.3f})")
            
            elif model_name == 'user_segmentation':
                if metrics['silhouette_score'] > 0.3:
                    production_ready.append(f"✅ {model_name.upper()}: Ready (Silhouette: {metrics['silhouette_score']:.3f})")
                else:
                    needs_improvement.append(f"⚠️ {model_name.upper()}: Needs improvement (Silhouette: {metrics['silhouette_score']:.3f})")
        
        print("\nREADY FOR PRODUCTION:")
        for item in production_ready:
            print(f"  {item}")
        
        print("\nNEEDS IMPROVEMENT:")
        for item in needs_improvement:
            print(f"  {item}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 80)
        
        recommendations = [
            "1. Deploy production-ready models immediately",
            "2. Implement continuous monitoring for all deployed models",
            "3. Set up automated retraining pipeline",
            "4. Create model performance dashboards",
            "5. Implement A/B testing framework",
            "6. Add ensemble methods for better accuracy",
            "7. Create model versioning and rollback capabilities"
        ]
        
        for rec in recommendations:
            print(f"  {rec}")
    
    def plot_model_comparison(self):
        """Plot comprehensive model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract metrics for plotting
        model_names = list(self.evaluation_metrics.keys())
        
        # Classification metrics
        classification_models = [name for name in model_names if 'test_f1' in self.evaluation_metrics[name]]
        if classification_models:
            f1_scores = [self.evaluation_metrics[name]['test_f1'] for name in classification_models]
            axes[0, 0].bar(classification_models, f1_scores, alpha=0.7, color='skyblue')
            axes[0, 0].set_title('F1 Scores - Classification Models')
            axes[0, 0].set_ylabel('F1 Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Regression metrics
        regression_models = [name for name in model_names if 'test_r2' in self.evaluation_metrics[name]]
        if regression_models:
            r2_scores = [self.evaluation_metrics[name]['test_r2'] for name in regression_models]
            axes[0, 1].bar(regression_models, r2_scores, alpha=0.7, color='lightcoral')
            axes[0, 1].set_title('R² Scores - Regression Models')
            axes[0, 1].set_ylabel('R² Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Clustering metrics
        clustering_models = [name for name in model_names if 'silhouette_score' in self.evaluation_metrics[name]]
        if clustering_models:
            silhouette_scores = [self.evaluation_metrics[name]['silhouette_score'] for name in clustering_models]
            axes[1, 0].bar(clustering_models, silhouette_scores, alpha=0.7, color='lightgreen')
            axes[1, 0].set_title('Silhouette Scores - Clustering Models')
            axes[1, 0].set_ylabel('Silhouette Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall performance summary
        all_scores = []
        all_names = []
        for name, metrics in self.evaluation_metrics.items():
            if 'test_f1' in metrics:
                all_scores.append(metrics['test_f1'])
                all_names.append(f"{name}\n(F1)")
            elif 'test_r2' in metrics:
                all_scores.append(metrics['test_r2'])
                all_names.append(f"{name}\n(R²)")
            elif 'silhouette_score' in metrics:
                all_scores.append(metrics['silhouette_score'])
                all_names.append(f"{name}\n(Silhouette)")
        
        axes[1, 1].bar(all_names, all_scores, alpha=0.7, color='gold')
        axes[1, 1].set_title('Overall Model Performance')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('visualizations/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_all_models(self):
        """Save all trained models and scalers"""
        os.makedirs('models', exist_ok=True)
        
        for model_name, model in self.best_models.items():
            joblib.dump(model, f'models/optimized_{model_name}_model.pkl')
        
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, f'models/optimized_{scaler_name}_scaler.pkl')
        
        joblib.dump(self.evaluation_metrics, 'models/optimized_evaluation_metrics.pkl')
        
        print("All optimized models and scalers saved successfully!")

def main():
    """Main pipeline execution"""
    print("🚀 Starting Unified ML Pipeline with Hyperparameter Optimization")
    print("="*80)
    
    # Initialize pipeline
    pipeline = UnifiedMLPipeline()
    
    # Load data
    df = pipeline.load_and_preprocess_data()
    
    # Train all models
    pipeline.train_risk_prediction_model(df)
    pipeline.train_churn_risk_model(df)
    pipeline.train_financial_health_model(df)
    pipeline.train_user_segmentation_model(df)
    
    # Generate comprehensive report
    pipeline.generate_comprehensive_report()
    
    # Save all models
    pipeline.save_all_models()
    
    print("\n🎉 Unified ML Pipeline Complete!")
    print("Check the 'visualizations/' directory for plots.")
    print("All optimized models saved in 'models/' directory.")

if __name__ == "__main__":
    main()
