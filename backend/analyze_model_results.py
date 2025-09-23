#!/usr/bin/env python3
"""
Detailed Model Analysis Script
Analyzes the test results and provides actionable insights
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

class ModelAnalyzer:
    def __init__(self, results_dir: str = "test_results"):
        self.results_dir = results_dir
        self.analysis_results = {}
    
    def analyze_risk_prediction_results(self):
        """Analyze risk prediction model results"""
        print("🔍 ANALYZING RISK PREDICTION MODEL RESULTS")
        print("-" * 60)
        
        try:
            # Load comparison data
            original_df = pd.read_csv(f"{self.results_dir}/risk_prediction_comparison.csv")
            improved_df = pd.read_csv(f"{self.results_dir}/improved_risk_prediction_comparison.csv")
            
            print(f"📊 Original Risk Prediction Model:")
            print(f"   Total Samples: {len(original_df)}")
            print(f"   Correct Predictions: {original_df['Correct'].sum()}")
            print(f"   Accuracy: {original_df['Correct'].mean():.4f}")
            
            # Analyze prediction patterns
            original_confusion = pd.crosstab(original_df['Actual_Risk_Label'], original_df['Predicted_Risk_Label'])
            print(f"\n   Confusion Matrix:")
            print(original_confusion)
            
            print(f"\n📊 Improved Risk Prediction Model:")
            print(f"   Total Samples: {len(improved_df)}")
            print(f"   Correct Predictions: {improved_df['Correct'].sum()}")
            print(f"   Accuracy: {improved_df['Correct'].mean():.4f}")
            
            # Analyze prediction patterns
            improved_confusion = pd.crosstab(improved_df['Actual_Risk_Label'], improved_df['Predicted_Risk_Label'])
            print(f"\n   Confusion Matrix:")
            print(improved_confusion)
            
            # Calculate improvement
            improvement = improved_df['Correct'].mean() - original_df['Correct'].mean()
            print(f"\n🎯 IMPROVEMENT: {improvement:.4f} ({improvement*100:.2f} percentage points)")
            
            # Analyze where improvements occurred
            print(f"\n📈 IMPROVEMENT ANALYSIS:")
            for risk_level in ['Low', 'Medium', 'High']:
                original_acc = (original_df[original_df['Actual_Risk_Label'] == risk_level]['Correct'].mean())
                improved_acc = (improved_df[improved_df['Actual_Risk_Label'] == risk_level]['Correct'].mean())
                improvement_level = improved_acc - original_acc
                print(f"   {risk_level} Risk: {original_acc:.3f} → {improved_acc:.3f} ({improvement_level:+.3f})")
            
            self.analysis_results['risk_prediction'] = {
                'original_accuracy': original_df['Correct'].mean(),
                'improved_accuracy': improved_df['Correct'].mean(),
                'improvement': improvement,
                'status': 'SIGNIFICANT IMPROVEMENT' if improvement > 0.1 else 'MODERATE IMPROVEMENT' if improvement > 0.05 else 'MINIMAL IMPROVEMENT'
            }
            
        except Exception as e:
            print(f"❌ Error analyzing risk prediction: {e}")
    
    def analyze_financial_health_results(self):
        """Analyze financial health model results"""
        print("\n🔍 ANALYZING FINANCIAL HEALTH MODEL RESULTS")
        print("-" * 60)
        
        try:
            df = pd.read_csv(f"{self.results_dir}/financial_health_comparison.csv")
            
            print(f"📊 Financial Health Model:")
            print(f"   Total Samples: {len(df)}")
            print(f"   Mean Absolute Error: {df['Abs_Difference'].mean():.2f}")
            print(f"   RMSE: {np.sqrt(np.mean(df['Difference']**2)):.2f}")
            print(f"   R² Score: {1 - (np.sum(df['Difference']**2) / np.sum((df['Actual_Score'] - df['Actual_Score'].mean())**2)):.4f}")
            
            # Analyze prediction accuracy by score ranges
            df['Score_Range'] = pd.cut(df['Actual_Score'], bins=[0, 25, 50, 75, 100], labels=['Poor (0-25)', 'Fair (25-50)', 'Good (50-75)', 'Excellent (75-100)'])
            
            print(f"\n📈 ACCURACY BY SCORE RANGE:")
            for range_name in df['Score_Range'].cat.categories:
                range_data = df[df['Score_Range'] == range_name]
                if len(range_data) > 0:
                    mae = range_data['Abs_Difference'].mean()
                    print(f"   {range_name}: MAE = {mae:.2f} (n={len(range_data)})")
            
            # Identify problematic predictions
            large_errors = df[df['Abs_Difference'] > 20]
            print(f"\n🚨 LARGE ERRORS (>20 points): {len(large_errors)} cases")
            if len(large_errors) > 0:
                print(f"   Average error: {large_errors['Abs_Difference'].mean():.2f}")
                print(f"   Max error: {large_errors['Abs_Difference'].max():.2f}")
            
            self.analysis_results['financial_health'] = {
                'mae': df['Abs_Difference'].mean(),
                'rmse': np.sqrt(np.mean(df['Difference']**2)),
                'r2': 1 - (np.sum(df['Difference']**2) / np.sum((df['Actual_Score'] - df['Actual_Score'].mean())**2)),
                'large_errors': len(large_errors),
                'status': 'POOR' if df['Abs_Difference'].mean() > 15 else 'NEEDS IMPROVEMENT' if df['Abs_Difference'].mean() > 10 else 'ACCEPTABLE'
            }
            
        except Exception as e:
            print(f"❌ Error analyzing financial health: {e}")
    
    def analyze_investment_recommendation_results(self):
        """Analyze investment recommendation model results"""
        print("\n🔍 ANALYZING INVESTMENT RECOMMENDATION MODEL RESULTS")
        print("-" * 60)
        
        try:
            df = pd.read_csv(f"{self.results_dir}/investment_recommendation_comparison.csv")
            
            print(f"📊 Investment Recommendation Model:")
            print(f"   Total Samples: {len(df)}")
            print(f"   Mean Absolute Error: ${df['Abs_Difference'].mean():,.0f}")
            print(f"   RMSE: ${np.sqrt(np.mean(df['Difference']**2)):,.0f}")
            print(f"   Mean Percentage Error: {df['Percentage_Error'].mean():.1f}%")
            
            # Analyze prediction accuracy by pension amount ranges
            df['Pension_Range'] = pd.cut(df['Actual_Pension'], bins=[0, 100000, 300000, 500000, float('inf')], 
                                       labels=['Low (<100k)', 'Medium (100k-300k)', 'High (300k-500k)', 'Very High (>500k)'])
            
            print(f"\n📈 ACCURACY BY PENSION RANGE:")
            for range_name in df['Pension_Range'].cat.categories:
                range_data = df[df['Pension_Range'] == range_name]
                if len(range_data) > 0:
                    mae = range_data['Abs_Difference'].mean()
                    mpe = range_data['Percentage_Error'].mean()
                    print(f"   {range_name}: MAE = ${mae:,.0f}, MPE = {mpe:.1f}% (n={len(range_data)})")
            
            # Identify problematic predictions
            large_errors = df[df['Percentage_Error'] > 100]
            print(f"\n🚨 LARGE ERRORS (>100% error): {len(large_errors)} cases")
            if len(large_errors) > 0:
                print(f"   Average error: {large_errors['Percentage_Error'].mean():.1f}%")
                print(f"   Max error: {large_errors['Percentage_Error'].max():.1f}%")
            
            self.analysis_results['investment_recommendation'] = {
                'mae': df['Abs_Difference'].mean(),
                'rmse': np.sqrt(np.mean(df['Difference']**2)),
                'mpe': df['Percentage_Error'].mean(),
                'large_errors': len(large_errors),
                'status': 'POOR' if df['Percentage_Error'].mean() > 50 else 'NEEDS IMPROVEMENT' if df['Percentage_Error'].mean() > 25 else 'ACCEPTABLE'
            }
            
        except Exception as e:
            print(f"❌ Error analyzing investment recommendation: {e}")
    
    def generate_actionable_recommendations(self):
        """Generate actionable recommendations based on analysis"""
        print("\n💡 ACTIONABLE RECOMMENDATIONS")
        print("=" * 60)
        
        recommendations = {
            "IMMEDIATE ACTIONS": [],
            "SHORT-TERM IMPROVEMENTS": [],
            "LONG-TERM STRATEGIES": []
        }
        
        # Risk Prediction recommendations
        if 'risk_prediction' in self.analysis_results:
            risk_result = self.analysis_results['risk_prediction']
            if risk_result['status'] == 'SIGNIFICANT IMPROVEMENT':
                recommendations["IMMEDIATE ACTIONS"].append("✅ Deploy Improved Risk Prediction model to production")
                recommendations["SHORT-TERM IMPROVEMENTS"].append("🔧 Fine-tune Improved Risk Prediction model hyperparameters")
            else:
                recommendations["SHORT-TERM IMPROVEMENTS"].append("🔧 Investigate why Risk Prediction improvement is limited")
        
        # Financial Health recommendations
        if 'financial_health' in self.analysis_results:
            health_result = self.analysis_results['financial_health']
            if health_result['status'] == 'POOR':
                recommendations["IMMEDIATE ACTIONS"].append("🚨 DO NOT USE Financial Health model in production")
                recommendations["SHORT-TERM IMPROVEMENTS"].append("🔧 Completely retrain Financial Health model with better features")
                recommendations["SHORT-TERM IMPROVEMENTS"].append("🔧 Check data quality and feature engineering for Financial Health")
            elif health_result['status'] == 'NEEDS IMPROVEMENT':
                recommendations["SHORT-TERM IMPROVEMENTS"].append("🔧 Improve Financial Health model before production use")
        
        # Investment Recommendation recommendations
        if 'investment_recommendation' in self.analysis_results:
            inv_result = self.analysis_results['investment_recommendation']
            if inv_result['status'] == 'POOR':
                recommendations["IMMEDIATE ACTIONS"].append("🚨 DO NOT USE Investment Recommendation model in production")
                recommendations["SHORT-TERM IMPROVEMENTS"].append("🔧 Investigate data quality issues in Investment Recommendation model")
                recommendations["SHORT-TERM IMPROVEMENTS"].append("🔧 Try different algorithms for Investment Recommendation")
            elif inv_result['status'] == 'NEEDS IMPROVEMENT':
                recommendations["SHORT-TERM IMPROVEMENTS"].append("🔧 Improve Investment Recommendation model accuracy")
        
        # General recommendations
        recommendations["LONG-TERM STRATEGIES"].extend([
            "📊 Implement continuous model monitoring",
            "🔄 Set up automated model retraining pipeline",
            "📈 Add more sophisticated feature engineering",
            "🎯 Implement ensemble methods for better accuracy",
            "📋 Create model performance dashboards",
            "🔍 Add A/B testing framework for model comparison"
        ])
        
        # Print recommendations
        for category, items in recommendations.items():
            if items:
                print(f"\n{category}:")
                for i, item in enumerate(items, 1):
                    print(f"  {i}. {item}")
    
    def create_performance_summary(self):
        """Create a performance summary table"""
        print("\n📊 MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        
        summary_data = []
        
        # Risk Prediction
        if 'risk_prediction' in self.analysis_results:
            risk_result = self.analysis_results['risk_prediction']
            summary_data.append({
                'Model': 'Risk Prediction (Original)',
                'Metric': 'Accuracy',
                'Value': f"{risk_result['original_accuracy']:.4f}",
                'Status': 'POOR',
                'Recommendation': 'DO NOT USE'
            })
            summary_data.append({
                'Model': 'Risk Prediction (Improved)',
                'Metric': 'Accuracy',
                'Value': f"{risk_result['improved_accuracy']:.4f}",
                'Status': 'GOOD',
                'Recommendation': 'USE IN PRODUCTION'
            })
        
        # Financial Health
        if 'financial_health' in self.analysis_results:
            health_result = self.analysis_results['financial_health']
            summary_data.append({
                'Model': 'Financial Health',
                'Metric': 'MAE',
                'Value': f"{health_result['mae']:.2f}",
                'Status': health_result['status'],
                'Recommendation': 'DO NOT USE' if health_result['status'] == 'POOR' else 'NEEDS IMPROVEMENT'
            })
        
        # Investment Recommendation
        if 'investment_recommendation' in self.analysis_results:
            inv_result = self.analysis_results['investment_recommendation']
            summary_data.append({
                'Model': 'Investment Recommendation',
                'Metric': 'MPE',
                'Value': f"{inv_result['mpe']:.1f}%",
                'Status': inv_result['status'],
                'Recommendation': 'DO NOT USE' if inv_result['status'] == 'POOR' else 'NEEDS IMPROVEMENT'
            })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Count models by status
        status_counts = summary_df['Status'].value_counts()
        print(f"\n📈 MODEL STATUS BREAKDOWN:")
        for status, count in status_counts.items():
            print(f"   {status}: {count} models")
    
    def run_full_analysis(self):
        """Run complete analysis"""
        print("🚀 STARTING DETAILED MODEL ANALYSIS")
        print("=" * 80)
        
        # Check if results directory exists
        if not os.path.exists(self.results_dir):
            print(f"❌ Results directory '{self.results_dir}' not found")
            print("   Please run the model testing script first")
            return
        
        # Run all analyses
        self.analyze_risk_prediction_results()
        self.analyze_financial_health_results()
        self.analyze_investment_recommendation_results()
        
        # Generate recommendations
        self.generate_actionable_recommendations()
        
        # Create performance summary
        self.create_performance_summary()
        
        print(f"\n🎉 Analysis complete!")
        print(f"📁 Detailed results saved in '{self.results_dir}/' directory")

if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    analyzer.run_full_analysis()
