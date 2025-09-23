#!/usr/bin/env python3
"""
Main Training Script - Orchestrates all model training with visualizations
"""

import os
import sys
from train_improved_risk_prediction import ImprovedRiskPredictionTrainer
from train_financial_health import FinancialHealthTrainer
from train_churn_risk import ChurnRiskTrainer
from train_investment_recommendation import InvestmentRecommendationTrainer
from train_user_segmentation import UserSegmentationTrainer
from model_usage_analysis import analyze_model_usage

def main():
    """Main training pipeline"""
    print("🚀 Starting Enhanced ML Model Training Pipeline")
    print("="*60)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    results = {}
    
    try:
        # 1. Train Improved Risk Prediction Model
        print("\n📊 Training Improved Risk Prediction Model...")
        risk_trainer = ImprovedRiskPredictionTrainer()
        risk_metrics = risk_trainer.train_model()
        risk_trainer.save_model()
        results.update(risk_trainer.results)
        
        # 2. Train Financial Health Model
        print("\n📊 Training Financial Health Model...")
        health_trainer = FinancialHealthTrainer()
        health_metrics = health_trainer.train_model()
        health_trainer.save_model()
        results.update(health_trainer.results)
        
        # 3. Train Churn Risk Model
        print("\n📊 Training Churn Risk Model...")
        churn_trainer = ChurnRiskTrainer()
        churn_metrics = churn_trainer.train_model()
        churn_trainer.save_model()
        results.update(churn_trainer.results)
        
        # 4. Train Investment Recommendation Model
        print("\n📊 Training Investment Recommendation Model...")
        investment_trainer = InvestmentRecommendationTrainer()
        investment_metrics = investment_trainer.train_model()
        investment_trainer.save_model()
        results.update(investment_trainer.results)
        
        # 5. Train User Segmentation Model
        print("\n📊 Training User Segmentation Model...")
        segmentation_trainer = UserSegmentationTrainer()
        segmentation_metrics = segmentation_trainer.train_model()
        segmentation_trainer.save_model()
        results.update(segmentation_trainer.results)
        
        # 6. Analyze Model Usage
        print("\n📊 Analyzing Model Usage...")
        analyze_model_usage()
        
        # 7. Summary Report
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE - SUMMARY REPORT")
        print("="*60)
        
        for model_name, metrics in results.items():
            print(f"\n✅ {model_name}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {metric.capitalize()}: {value:.4f}")
                else:
                    print(f"   {metric.capitalize()}: {value}")
        
        print(f"\n📁 Files created:")
        print(f"   - Models: models/ directory")
        print(f"   - Visualizations: visualizations/ directory")
        print(f"   - Confusion matrices, ROC curves, Learning curves")
        print(f"   - Feature importance plots")
        
        print(f"\n🔧 Next steps:")
        print(f"   1. Review visualizations in visualizations/ directory")
        print(f"   2. Test models with: python -m pytest tests/")
        print(f"   3. Deploy models to production")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
