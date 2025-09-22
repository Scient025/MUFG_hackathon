#!/usr/bin/env python3
"""
Comprehensive Model Testing Script
Tests all models using CSV data and compares predictions with actual values
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple

class ModelTester:
    def __init__(self, csv_path: str = "../case1.csv", models_dir: str = "models"):
        self.csv_path = csv_path
        self.models_dir = models_dir
        self.df = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.results = {}
        
        # Load data and models
        self.load_csv_data()
        self.load_models()
    
    def load_csv_data(self):
        """Load and preprocess CSV data"""
        print("📊 Loading CSV data...")
        
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ CSV data loaded: {len(self.df)} records, {len(self.df.columns)} features")
            
            # Handle missing values
            self.df = self.df.fillna({
                'Risk_Tolerance': 'Medium',
                'Investment_Type': 'ETF',
                'Fund_Name': 'Default Fund',
                'Marital_Status': 'Single',
                'Education_Level': "Bachelor's",
                'Health_Status': 'Average',
                'Annual_Income': 0,
                'Current_Savings': 0,
                'Contribution_Amount': 0,
                'Years_Contributed': 0,
                'Age': 30,
                'Portfolio_Diversity_Score': 0.5,
                'Savings_Rate': 0.1,
                'Annual_Return_Rate': 5.0,
                'Volatility': 2.0,
                'Fees_Percentage': 1.0,
                'Projected_Pension_Amount': 0,
                'Debt_Level': 'Low',
                'Employment_Status': 'Full-time',
                'Investment_Experience_Level': 'Beginner',
                'Contribution_Frequency': 'Monthly',
                'Transaction_Amount': 0,
                'Transaction_Pattern_Score': 0.5,
                'Anomaly_Score': 0.1,
                'Suspicious_Flag': 'No'
            })
            
            # Convert numeric columns
            numeric_columns = [
                'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount', 'Years_Contributed',
                'Portfolio_Diversity_Score', 'Savings_Rate', 'Annual_Return_Rate', 'Volatility', 'Fees_Percentage',
                'Projected_Pension_Amount', 'Transaction_Amount', 'Transaction_Pattern_Score', 'Anomaly_Score'
            ]
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)
            
            # Encode categorical variables
            categorical_columns = [
                'Gender', 'Country', 'Employment_Status', 'Risk_Tolerance', 'Investment_Type', 'Fund_Name',
                'Marital_Status', 'Education_Level', 'Health_Status', 'Home_Ownership_Status',
                'Investment_Experience_Level', 'Financial_Goals', 'Insurance_Coverage', 'Pension_Type',
                'Withdrawal_Strategy', 'Debt_Level', 'Contribution_Frequency', 'Suspicious_Flag'
            ]
            for col in categorical_columns:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('Unknown')
                    le = LabelEncoder()
                    self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                    self.label_encoders[col] = le
            
            # Create derived features
            self.create_derived_features()
            
        except Exception as e:
            print(f"❌ Error loading CSV data: {e}")
            raise
    
    def create_derived_features(self):
        """Create derived features for testing"""
        # DTI Ratio
        self.df['DTI_Ratio'] = np.where(
            self.df['Annual_Income'] > 0,
            self.df['Debt_Level'].map({'Low': 0.1, 'Medium': 0.3, 'High': 0.5, 'Unknown': 0.2}).fillna(0.2),
            0.2
        )
        
        # Savings to Income Ratio
        self.df['Savings_to_Income_Ratio'] = np.where(
            self.df['Annual_Income'] > 0,
            self.df['Current_Savings'] / self.df['Annual_Income'], 0
        )
        
        # Contribution Percent of Income
        self.df['Contribution_Percent_of_Income'] = np.where(
            self.df['Annual_Income'] > 0,
            (self.df['Contribution_Amount'] * 12) / self.df['Annual_Income'], 0
        )
        
        # Risk Adjusted Return
        self.df['Risk_Adjusted_Return'] = np.where(
            self.df['Volatility'] > 0,
            self.df['Annual_Return_Rate'] / self.df['Volatility'], 0
        )
        
        print("✅ Derived features created")
    
    def load_models(self):
        """Load all trained models"""
        print("🤖 Loading trained models...")
        
        model_files = {
            'improved_risk_prediction': 'improved_risk_prediction_model.pkl',
            'financial_health': 'financial_health_model.pkl',
            'robust_churn_risk': 'robust_churn_risk_model.pkl',
            'investment_recommendation': 'investment_recommendation_model.pkl',
            'kmeans': 'kmeans_model.pkl'
        }
        
        scaler_files = {
            'scaler': 'scaler.pkl',
            'risk_scaler': 'risk_scaler.pkl',
            'investment_scaler': 'investment_scaler.pkl'
        }
        
        # Load models
        for model_name, filename in model_files.items():
            try:
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    self.models[model_name] = joblib.load(filepath)
                    print(f"✅ Loaded {model_name}")
                else:
                    print(f"⚠️ Model file not found: {filename}")
            except Exception as e:
                print(f"❌ Error loading {model_name}: {e}")
        
        # Load scalers
        for scaler_name, filename in scaler_files.items():
            try:
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    self.scalers[scaler_name] = joblib.load(filepath)
                    print(f"✅ Loaded {scaler_name}")
                else:
                    print(f"⚠️ Scaler file not found: {filename}")
            except Exception as e:
                print(f"❌ Error loading {scaler_name}: {e}")
        
        print(f"✅ Loaded {len(self.models)} models and {len(self.scalers)} scalers")
    
    def test_improved_risk_prediction_model(self):
        """Test improved risk prediction model"""
        print("\n🔍 Testing Improved Risk Prediction Model...")
        
        if 'improved_risk_prediction' not in self.models:
            print("❌ Improved risk prediction model not available")
            return None
        
        # Features for improved risk prediction
        risk_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Years_Contributed', 'Investment_Experience_Level_encoded',
            'Portfolio_Diversity_Score', 'Savings_Rate', 'Debt_Level_encoded',
            'Age_Income_Interaction', 'Savings_Income_Ratio', 'Contribution_Income_Ratio',
            'Age_Group', 'Income_Bracket', 'Financial_Stability'
        ]
        
        # Create additional features if not present
        if 'Age_Income_Interaction' not in self.df.columns:
            self.df['Age_Income_Interaction'] = self.df['Age'] * self.df['Annual_Income']
        if 'Savings_Income_Ratio' not in self.df.columns:
            self.df['Savings_Income_Ratio'] = self.df['Current_Savings'] / (self.df['Annual_Income'] + 1)
        if 'Contribution_Income_Ratio' not in self.df.columns:
            self.df['Contribution_Income_Ratio'] = (self.df['Contribution_Amount'] * 12) / (self.df['Annual_Income'] + 1)
        if 'Age_Group' not in self.df.columns:
            self.df['Age_Group'] = pd.cut(self.df['Age'], bins=[0, 30, 40, 50, 100], labels=[0, 1, 2, 3]).astype(int)
        if 'Income_Bracket' not in self.df.columns:
            self.df['Income_Bracket'] = pd.cut(self.df['Annual_Income'], bins=[0, 50000, 75000, 100000, float('inf')], labels=[0, 1, 2, 3]).astype(int)
        if 'Financial_Stability' not in self.df.columns:
            self.df['Financial_Stability'] = (
                self.df['Savings_Income_Ratio'] * 0.4 +
                self.df['Contribution_Income_Ratio'] * 0.3 +
                self.df['Portfolio_Diversity_Score'] * 0.2 +
                (self.df['Years_Contributed'] / 10) * 0.1
            )
        
        # Prepare data
        risk_data = self.df[risk_features + ['Risk_Tolerance_encoded']].dropna()
        X = risk_data[risk_features]
        y_true = risk_data['Risk_Tolerance_encoded']
        
        # Predict (Random Forest doesn't need scaling)
        y_pred = self.models['improved_risk_prediction'].predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'User_ID': risk_data.index,
            'Actual_Risk': y_true,
            'Predicted_Risk': y_pred,
            'Correct': y_true == y_pred
        })
        
        # Map encoded values back to labels
        risk_labels = ['Low', 'Medium', 'High']
        comparison_df['Actual_Risk_Label'] = comparison_df['Actual_Risk'].map(lambda x: risk_labels[int(x)])
        comparison_df['Predicted_Risk_Label'] = comparison_df['Predicted_Risk'].map(lambda x: risk_labels[int(x)])
        
        results = {
            'model_name': 'Improved Risk Prediction',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': len(y_true),
            'correct_predictions': sum(y_true == y_pred),
            'comparison_df': comparison_df,
            'status': 'GOOD' if accuracy > 0.7 else 'NEEDS IMPROVEMENT' if accuracy > 0.5 else 'POOR'
        }
        
        print(f"📊 Improved Risk Prediction Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Status: {results['status']}")
        
        return results
    
    def test_financial_health_model(self):
        """Test financial health model"""
        print("\n🔍 Testing Financial Health Model...")
        
        if 'financial_health' not in self.models:
            print("❌ Financial health model not available")
            return None
        
        # Features for financial health
        health_features = [
            'Annual_Income','Current_Savings','Savings_Rate','Debt_Level_encoded','Portfolio_Diversity_Score',
            'Contribution_Amount','Contribution_Frequency_encoded','Years_Contributed','DTI_Ratio',
            'Savings_to_Income_Ratio','Contribution_Percent_of_Income','Risk_Adjusted_Return','Age','Investment_Experience_Level_encoded'
        ]
        
        # Calculate actual financial health scores using the same logic as training
        def calculate_actual_health_score(row):
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
        
        # Prepare data
        health_data = self.df[health_features].dropna()
        X = health_data[health_features]
        
        # Calculate actual scores
        health_data_with_scores = health_data.copy()
        health_data_with_scores['Actual_Health_Score'] = health_data_with_scores.apply(calculate_actual_health_score, axis=1)
        y_true = health_data_with_scores['Actual_Health_Score']
        
        # Predict
        y_pred = self.models['financial_health'].predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'User_ID': health_data.index,
            'Actual_Score': y_true,
            'Predicted_Score': y_pred,
            'Difference': y_pred - y_true,
            'Abs_Difference': np.abs(y_pred - y_true)
        })
        
        results = {
            'model_name': 'Financial Health',
            'rmse': rmse,
            'r2_score': r2,
            'mse': mse,
            'total_samples': len(y_true),
            'mean_absolute_error': np.mean(np.abs(y_pred - y_true)),
            'comparison_df': comparison_df,
            'status': 'EXCELLENT' if r2 > 0.8 else 'GOOD' if r2 > 0.6 else 'NEEDS IMPROVEMENT' if r2 > 0.4 else 'POOR'
        }
        
        print(f"📊 Financial Health Results:")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Mean Absolute Error: {results['mean_absolute_error']:.2f}")
        print(f"   Status: {results['status']}")
        
        return results
    
    def test_robust_churn_risk_model(self):
        """Test robust churn risk model"""
        print("\n🔍 Testing Robust Churn Risk Model...")
        
        if 'robust_churn_risk' not in self.models:
            print("❌ Robust churn risk model not available")
            return None
        
        # Features for robust churn risk (10 features as expected by the model)
        churn_features = [
            'Age', 'Annual_Income', 'Years_Contributed', 'Portfolio_Diversity_Score',
            'Debt_Level_encoded', 'Investment_Experience_Level_encoded', 'Marital_Status_encoded',
            'Number_of_Dependents', 'Education_Level_encoded', 'Health_Status_encoded'
        ]
        
        # Create realistic churn labels for robust model evaluation
        def create_actual_churn_label(row):
            churn_score = 0
            # Age and engagement factors
            if row['Age'] < 30 and row['Years_Contributed'] < 5: churn_score += 2
            # Marital status and education factors
            if row['Marital_Status_encoded'] == 0 and row['Education_Level_encoded'] == 0: churn_score += 2
            # Dependents factor
            if row['Number_of_Dependents'] > 2: churn_score += 2
            # Financial stress factors
            if row['Portfolio_Diversity_Score'] < 0.3: churn_score += 1
            return 1 if churn_score >= 2 else 0
        
        # Prepare data
        churn_data = self.df[churn_features].dropna()
        X = churn_data[churn_features]
        
        # Calculate actual churn labels
        churn_data_with_labels = churn_data.copy()
        churn_data_with_labels['Actual_Churn'] = churn_data_with_labels.apply(create_actual_churn_label, axis=1)
        y_true = churn_data_with_labels['Actual_Churn']
        
        # Predict
        y_pred = self.models['robust_churn_risk'].predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'User_ID': churn_data.index,
            'Actual_Churn': y_true,
            'Predicted_Churn': y_pred,
            'Correct': y_true == y_pred
        })
        
        results = {
            'model_name': 'Churn Risk',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': len(y_true),
            'correct_predictions': sum(y_true == y_pred),
            'comparison_df': comparison_df,
            'status': 'EXCELLENT' if accuracy > 0.9 else 'GOOD' if accuracy > 0.8 else 'NEEDS IMPROVEMENT' if accuracy > 0.6 else 'POOR'
        }
        
        print(f"📊 Churn Risk Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Status: {results['status']}")
        
        return results
    
    def test_investment_recommendation_model(self):
        """Test investment recommendation model"""
        print("\n🔍 Testing Investment Recommendation Model...")
        
        if 'investment_recommendation' not in self.models or 'investment_scaler' not in self.scalers:
            print("❌ Investment recommendation model/scaler not available")
            return None
        
        # Features for investment recommendation
        investment_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Investment_Type_encoded', 'Annual_Return_Rate',
            'Volatility', 'Fees_Percentage', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded'
        ]
        
        # Prepare data
        investment_data = self.df[investment_features + ['Projected_Pension_Amount']].dropna()
        X = investment_data[investment_features]
        y_true = investment_data['Projected_Pension_Amount']
        
        # Scale features
        X_scaled = self.scalers['investment_scaler'].transform(X)
        
        # Predict
        y_pred = self.models['investment_recommendation'].predict(X_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'User_ID': investment_data.index,
            'Actual_Pension': y_true,
            'Predicted_Pension': y_pred,
            'Difference': y_pred - y_true,
            'Abs_Difference': np.abs(y_pred - y_true),
            'Percentage_Error': np.abs((y_pred - y_true) / (y_true + 1)) * 100
        })
        
        results = {
            'model_name': 'Investment Recommendation',
            'rmse': rmse,
            'r2_score': r2,
            'mse': mse,
            'total_samples': len(y_true),
            'mean_absolute_error': np.mean(np.abs(y_pred - y_true)),
            'mean_percentage_error': np.mean(comparison_df['Percentage_Error']),
            'comparison_df': comparison_df,
            'status': 'EXCELLENT' if r2 > 0.8 else 'GOOD' if r2 > 0.6 else 'NEEDS IMPROVEMENT' if r2 > 0.4 else 'POOR'
        }
        
        print(f"📊 Investment Recommendation Results:")
        print(f"   RMSE: {rmse:.2f}")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Mean Absolute Error: {results['mean_absolute_error']:.2f}")
        print(f"   Mean Percentage Error: {results['mean_percentage_error']:.2f}%")
        print(f"   Status: {results['status']}")
        
        return results
    
    def test_user_segmentation_model(self):
        """Test user segmentation model"""
        print("\n🔍 Testing User Segmentation Model...")
        
        if 'kmeans' not in self.models or 'scaler' not in self.scalers:
            print("❌ KMeans model/scaler not available")
            return None
        
        # Features for clustering
        clustering_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Years_Contributed', 'Portfolio_Diversity_Score'
        ]
        
        # Prepare data
        clustering_data = self.df[clustering_features].dropna()
        X = clustering_data[clustering_features]
        
        # Scale features
        X_scaled = self.scalers['scaler'].transform(X)
        
        # Predict clusters
        clusters = self.models['kmeans'].predict(X_scaled)
        
        # Analyze cluster characteristics
        clustering_data_with_clusters = clustering_data.copy()
        clustering_data_with_clusters['Cluster'] = clusters
        
        cluster_analysis = {}
        for cluster_id in range(self.models['kmeans'].n_clusters):
            cluster_data = clustering_data_with_clusters[clustering_data_with_clusters['Cluster'] == cluster_id]
            cluster_analysis[cluster_id] = {
                'count': len(cluster_data),
                'avg_age': cluster_data['Age'].mean(),
                'avg_income': cluster_data['Annual_Income'].mean(),
                'avg_savings': cluster_data['Current_Savings'].mean(),
                'avg_contribution': cluster_data['Contribution_Amount'].mean(),
                'avg_risk_tolerance': cluster_data['Risk_Tolerance_encoded'].mean()
            }
        
        results = {
            'model_name': 'User Segmentation',
            'total_samples': len(clusters),
            'n_clusters': self.models['kmeans'].n_clusters,
            'cluster_distribution': np.bincount(clusters),
            'cluster_analysis': cluster_analysis,
            'status': 'GOOD' if len(set(clusters)) > 2 else 'NEEDS IMPROVEMENT'
        }
        
        print(f"📊 User Segmentation Results:")
        print(f"   Total Samples: {results['total_samples']}")
        print(f"   Number of Clusters: {results['n_clusters']}")
        print(f"   Cluster Distribution: {results['cluster_distribution']}")
        print(f"   Status: {results['status']}")
        
        return results
    
    def run_all_tests(self):
        """Run all model tests"""
        print("🚀 Starting Comprehensive Model Testing...")
        print("="*80)
        
        # Test all models
        test_methods = [
            self.test_improved_risk_prediction_model,
            self.test_financial_health_model,
            self.test_robust_churn_risk_model,
            self.test_investment_recommendation_model,
            self.test_user_segmentation_model
        ]
        
        for test_method in test_methods:
            try:
                result = test_method()
                if result:
                    self.results[result['model_name']] = result
            except Exception as e:
                print(f"❌ Error in {test_method.__name__}: {e}")
        
        return self.results
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE MODEL TESTING SUMMARY REPORT")
        print("="*80)
        
        if not self.results:
            print("❌ No test results available")
            return
        
        # Categorize models by performance
        excellent_models = []
        good_models = []
        needs_improvement_models = []
        poor_models = []
        
        for model_name, result in self.results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'EXCELLENT':
                excellent_models.append((model_name, result))
            elif status == 'GOOD':
                good_models.append((model_name, result))
            elif status == 'NEEDS IMPROVEMENT':
                needs_improvement_models.append((model_name, result))
            else:
                poor_models.append((model_name, result))
        
        print(f"\n✅ EXCELLENT PERFORMANCE ({len(excellent_models)} models):")
        for model_name, result in excellent_models:
            print(f"   🎯 {model_name}")
            if 'accuracy' in result:
                print(f"      Accuracy: {result['accuracy']:.4f}")
            if 'r2_score' in result:
                print(f"      R² Score: {result['r2_score']:.4f}")
        
        print(f"\n✅ GOOD PERFORMANCE ({len(good_models)} models):")
        for model_name, result in good_models:
            print(f"   👍 {model_name}")
            if 'accuracy' in result:
                print(f"      Accuracy: {result['accuracy']:.4f}")
            if 'r2_score' in result:
                print(f"      R² Score: {result['r2_score']:.4f}")
        
        print(f"\n⚠️ NEEDS IMPROVEMENT ({len(needs_improvement_models)} models):")
        for model_name, result in needs_improvement_models:
            print(f"   🔧 {model_name}")
            if 'accuracy' in result:
                print(f"      Accuracy: {result['accuracy']:.4f}")
            if 'r2_score' in result:
                print(f"      R² Score: {result['r2_score']:.4f}")
        
        print(f"\n❌ POOR PERFORMANCE ({len(poor_models)} models):")
        for model_name, result in poor_models:
            print(f"   🚨 {model_name}")
            if 'accuracy' in result:
                print(f"      Accuracy: {result['accuracy']:.4f}")
            if 'r2_score' in result:
                print(f"      R² Score: {result['r2_score']:.4f}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   🎯 USE IN PRODUCTION: {', '.join([m[0] for m in excellent_models + good_models])}")
        print(f"   🔧 IMPROVE BEFORE USE: {', '.join([m[0] for m in needs_improvement_models])}")
        print(f"   🚨 DO NOT USE: {', '.join([m[0] for m in poor_models])}")
        
        # Save detailed results
        self.save_detailed_results()
    
    def save_detailed_results(self):
        """Save detailed results to CSV files"""
        print(f"\n💾 Saving detailed results...")
        
        os.makedirs('test_results', exist_ok=True)
        
        for model_name, result in self.results.items():
            if 'comparison_df' in result:
                filename = f"test_results/{model_name.lower().replace(' ', '_')}_comparison.csv"
                result['comparison_df'].to_csv(filename, index=False)
                print(f"   ✅ Saved {filename}")
        
        print(f"   📁 All detailed results saved in 'test_results/' directory")

if __name__ == "__main__":
    # Initialize tester
    tester = ModelTester()
    
    # Run all tests
    results = tester.run_all_tests()
    
    # Generate summary report
    tester.generate_summary_report()
    
    print(f"\n🎉 Model testing complete!")
    print(f"📊 Tested {len(results)} models")
    print(f"📁 Check 'test_results/' directory for detailed comparison files")
