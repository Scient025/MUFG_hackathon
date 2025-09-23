#!/usr/bin/env python3
"""
Model Usage Analysis - Shows where each model is used in the codebase
"""

import os
import re
from typing import Dict, List, Tuple

def analyze_model_usage():
    """Analyze where each ML model is used in the codebase"""
    
    # Define model files and their expected usage
    models_info = {
        'improved_risk_prediction_model.pkl': {
            'description': 'Random Forest for improved risk tolerance prediction',
            'used_in': ['inference.py', 'chat_router.py'],
            'methods': ['predict_risk_tolerance']
        },
        'financial_health_model.pkl': {
            'description': 'Random Forest for financial health scoring',
            'used_in': ['advanced_ml_models.py', 'integrated_ml_pipeline.py'],
            'methods': ['predict_financial_health']
        },
        'churn_risk_model.pkl': {
            'description': 'XGBoost for churn risk prediction',
            'used_in': ['advanced_ml_models.py', 'integrated_ml_pipeline.py'],
            'methods': ['predict_churn_risk']
        },
        'investment_recommendation_model.pkl': {
            'description': 'XGBoost for investment recommendations',
            'used_in': ['inference.py', 'chat_router.py'],
            'methods': ['predict_pension_projection']
        },
        'kmeans_model.pkl': {
            'description': 'KMeans clustering for user segmentation',
            'used_in': ['inference.py', 'chat_router.py'],
            'methods': ['get_user_segment']
        },
        'anomaly_detection_model.pkl': {
            'description': 'Isolation Forest for anomaly detection',
            'used_in': ['advanced_ml_models.py', 'integrated_ml_pipeline.py'],
            'methods': ['detect_anomalies']
        },
        'fund_recommendation_model.pkl': {
            'description': 'Nearest Neighbors for fund recommendations',
            'used_in': ['advanced_ml_models.py', 'integrated_ml_pipeline.py'],
            'methods': ['recommend_funds']
        },
        'peer_matching_model.pkl': {
            'description': 'Nearest Neighbors for peer matching',
            'used_in': ['advanced_ml_models.py', 'integrated_ml_pipeline.py'],
            'methods': ['find_similar_peers']
        },
        'portfolio_optimization_model.pkl': {
            'description': 'Portfolio optimization algorithms',
            'used_in': ['advanced_ml_models.py', 'integrated_ml_pipeline.py'],
            'methods': ['optimize_portfolio']
        }
    }
    
    print("="*80)
    print("ML MODELS USAGE ANALYSIS")
    print("="*80)
    
    for model_name, info in models_info.items():
        print(f"\n📊 {model_name}")
        print(f"   Description: {info['description']}")
        print(f"   Used in files: {', '.join(info['used_in'])}")
        print(f"   Methods: {', '.join(info['methods'])}")
    
    print("\n" + "="*80)
    print("MODEL DEPENDENCIES")
    print("="*80)
    
    dependencies = {
        'Basic ML Pipeline (train.py)': [
            'kmeans_model.pkl',
            'risk_prediction_model.pkl', 
            'investment_recommendation_model.pkl',
            'scaler.pkl',
            'risk_scaler.pkl',
            'investment_scaler.pkl',
            'label_encoders.pkl'
        ],
        'Advanced ML Models (advanced_ml_models.py)': [
            'financial_health_model.pkl',
            'churn_risk_model.pkl',
            'anomaly_detection_model.pkl',
            'fund_recommendation_model.pkl',
            'monte_carlo_config.pkl',
            'peer_matching_model.pkl',
            'portfolio_optimization_model.pkl'
        ]
    }
    
    for pipeline, models in dependencies.items():
        print(f"\n🔧 {pipeline}:")
        for model in models:
            print(f"   - {model}")
    
    print("\n" + "="*80)
    print("API ENDPOINTS USING MODELS")
    print("="*80)
    
    api_endpoints = {
        '/api/user/{user_id}': ['get_user_profile'],
        '/api/summary/{user_id}': ['get_summary_stats'],
        '/api/peer_stats/{user_id}': ['get_user_segment', 'get_peer_statistics'],
        '/api/simulate': ['predict_pension_projection'],
        '/api/risk/{user_id}': ['predict_risk_tolerance'],
        '/api/chat': ['get_user_context', 'comprehensive analysis']
    }
    
    for endpoint, methods in api_endpoints.items():
        print(f"\n🌐 {endpoint}:")
        for method in methods:
            print(f"   - {method}")

if __name__ == "__main__":
    analyze_model_usage()
