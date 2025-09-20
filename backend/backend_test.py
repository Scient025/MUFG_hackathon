from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import traceback
import os
from datetime import datetime
import json
import os
os.environ["FLASK_SKIP_DOTENV"] = "1"

# Import your ML models class
from advanced_ml_models import AdvancedMLModels

app = Flask(__name__)

# Initialize ML models (this will load pre-trained models)
try:
    ml_models = AdvancedMLModels()
    print("✅ ML Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading ML models: {e}")
    ml_models = None

# Helper function to handle errors
def handle_error(error_msg: str, status_code: int = 500):
    return jsonify({
        'success': False,
        'error': error_msg,
        'timestamp': datetime.now().isoformat()
    }), status_code

# Helper function for successful responses
def success_response(data: Dict[str, Any], message: str = "Success"):
    return jsonify({
        'success': True,
        'message': message,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'api_status': 'healthy',
        'ml_models_loaded': ml_models is not None,
        'available_models': []
    }
    
    if ml_models:
        status['available_models'] = list(ml_models.models.keys())
    
    return success_response(status, "API is healthy")

@app.route('/models/info', methods=['GET'])
def models_info():
    """Get information about available models"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    model_info = {
        'total_models': len(ml_models.models),
        'models': {
            'financial_health': {
                'description': 'Predicts financial health score (0-100)',
                'type': 'Random Forest Regressor',
                'features': ['Income', 'Savings', 'Debt', 'Contributions', 'Portfolio diversity']
            },
            'churn_risk': {
                'description': 'Predicts probability of user churning',
                'type': 'XGBoost Classifier',
                'features': ['Age', 'Income', 'Employment', 'Contributions', 'Experience']
            },
            'anomaly_detection': {
                'description': 'Detects unusual transaction patterns',
                'type': 'Isolation Forest',
                'features': ['Transaction amount', 'Pattern score', 'Income', 'Savings']
            },
            'fund_recommendation': {
                'description': 'Recommends investment funds',
                'type': 'Collaborative Filtering (KNN)',
                'features': ['User preferences', 'Risk tolerance', 'Historical performance']
            },
            'monte_carlo': {
                'description': 'Retirement planning stress testing',
                'type': 'Monte Carlo Simulation',
                'features': ['Age', 'Contributions', 'Returns', 'Volatility']
            },
            'peer_matching': {
                'description': 'Finds similar users',
                'type': 'K-Nearest Neighbors',
                'features': ['Demographics', 'Financial profile', 'Investment behavior']
            },
            'portfolio_optimization': {
                'description': 'Optimizes portfolio allocation',
                'type': 'Mean-Variance Optimization',
                'features': ['Expected returns', 'Risk tolerance', 'Correlations']
            }
        },
        'data_summary': {
            'total_users': len(ml_models.df) if ml_models and ml_models.df is not None else 0,
            'features_count': len(ml_models.df.columns) if ml_models and ml_models.df is not None else 0
        }
    }
    
    return success_response(model_info, "Model information retrieved")

@app.route('/predict/financial-health', methods=['POST'])
def predict_financial_health():
    """Predict financial health score for a user"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return handle_error("user_id is required", 400)
        
        result = ml_models.predict_financial_health(user_id)
        
        if 'error' in result:
            return handle_error(result['error'], 404)
        
        return success_response(result, "Financial health prediction completed")
    
    except Exception as e:
        return handle_error(f"Prediction error: {str(e)}")

@app.route('/predict/churn-risk', methods=['POST'])
def predict_churn_risk():
    """Predict churn risk for a user"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return handle_error("user_id is required", 400)
        
        result = ml_models.predict_churn_risk(user_id)
        
        if 'error' in result:
            return handle_error(result['error'], 404)
        
        return success_response(result, "Churn risk prediction completed")
    
    except Exception as e:
        return handle_error(f"Prediction error: {str(e)}")

@app.route('/predict/anomaly', methods=['POST'])
def detect_anomalies():
    """Detect anomalies for a user"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return handle_error("user_id is required", 400)
        
        result = ml_models.detect_anomalies(user_id)
        
        if 'error' in result:
            return handle_error(result['error'], 404)
        
        return success_response(result, "Anomaly detection completed")
    
    except Exception as e:
        return handle_error(f"Detection error: {str(e)}")

@app.route('/recommend/funds', methods=['POST'])
def recommend_funds():
    """Recommend funds for a user"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        n_recommendations = data.get('n_recommendations', 5)
        
        if not user_id:
            return handle_error("user_id is required", 400)
        
        result = ml_models.recommend_funds(user_id, n_recommendations)
        
        if 'error' in result:
            return handle_error(result['error'], 404)
        
        return success_response(result, "Fund recommendations completed")
    
    except Exception as e:
        return handle_error(f"Recommendation error: {str(e)}")

@app.route('/simulate/monte-carlo', methods=['POST'])
def monte_carlo_simulation():
    """Run Monte Carlo simulation for retirement planning"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        n_simulations = data.get('n_simulations', 10000)
        
        if not user_id:
            return handle_error("user_id is required", 400)
        
        result = ml_models.run_monte_carlo_simulation(user_id, n_simulations)
        
        if 'error' in result:
            return handle_error(result['error'], 404)
        
        return success_response(result, "Monte Carlo simulation completed")
    
    except Exception as e:
        return handle_error(f"Simulation error: {str(e)}")

@app.route('/match/peers', methods=['POST'])
def find_peers():
    """Find similar peers for a user"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        n_peers = data.get('n_peers', 5)
        
        if not user_id:
            return handle_error("user_id is required", 400)
        
        result = ml_models.find_similar_peers(user_id, n_peers)
        
        if 'error' in result:
            return handle_error(result['error'], 404)
        
        return success_response(result, "Peer matching completed")
    
    except Exception as e:
        return handle_error(f"Matching error: {str(e)}")

@app.route('/optimize/portfolio', methods=['POST'])
def optimize_portfolio():
    """Optimize portfolio allocation for a user"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        
        if not user_id:
            return handle_error("user_id is required", 400)
        
        result = ml_models.optimize_portfolio(user_id)
        
        if 'error' in result:
            return handle_error(result['error'], 404)
        
        return success_response(result, "Portfolio optimization completed")
    
    except Exception as e:
        return handle_error(f"Optimization error: {str(e)}")

@app.route('/batch/predict', methods=['POST'])
def batch_predictions():
    """Run batch predictions for multiple users"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        prediction_types = data.get('prediction_types', ['financial_health', 'churn_risk'])
        
        if not user_ids:
            return handle_error("user_ids list is required", 400)
        
        if len(user_ids) > 100:  # Limit batch size
            return handle_error("Maximum 100 users per batch request", 400)
        
        results = {}
        
        for user_id in user_ids:
            user_results = {'user_id': user_id}
            
            try:
                if 'financial_health' in prediction_types:
                    user_results['financial_health'] = ml_models.predict_financial_health(user_id)
                
                if 'churn_risk' in prediction_types:
                    user_results['churn_risk'] = ml_models.predict_churn_risk(user_id)
                
                if 'anomaly' in prediction_types:
                    user_results['anomaly'] = ml_models.detect_anomalies(user_id)
                
                if 'fund_recommendations' in prediction_types:
                    user_results['fund_recommendations'] = ml_models.recommend_funds(user_id)
                
                results[user_id] = user_results
                
            except Exception as e:
                results[user_id] = {'error': str(e)}
        
        return success_response({
            'batch_size': len(user_ids),
            'completed': len(results),
            'results': results
        }, "Batch predictions completed")
    
    except Exception as e:
        return handle_error(f"Batch prediction error: {str(e)}")

@app.route('/analytics/summary', methods=['GET'])
def analytics_summary():
    """Get analytics summary across all users"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        df = ml_models.df
        
        if df is None or df.empty:
            return handle_error("No data available for analytics", 404)
        
        summary = {
            'user_statistics': {
                'total_users': len(df),
                'average_age': float(df['Age'].mean()),
                'average_income': float(df['Annual_Income'].mean()),
                'average_savings': float(df['Current_Savings'].mean()),
                'average_contribution': float(df['Contribution_Amount'].mean())
            },
            'financial_health_distribution': {
                'excellent_health': int((df['Financial_Health_Score'] >= 80).sum()),
                'good_health': int(((df['Financial_Health_Score'] >= 60) & (df['Financial_Health_Score'] < 80)).sum()),
                'moderate_health': int(((df['Financial_Health_Score'] >= 40) & (df['Financial_Health_Score'] < 60)).sum()),
                'poor_health': int((df['Financial_Health_Score'] < 40).sum())
            },
            'risk_tolerance_distribution': df['Risk_Tolerance'].value_counts().to_dict(),
            'investment_type_distribution': df['Investment_Type'].value_counts().to_dict(),
            'churn_risk_distribution': {
                'high_risk': int((df['Churn_Risk'] == 1).sum()),
                'low_risk': int((df['Churn_Risk'] == 0).sum())
            }
        }
        
        return success_response(summary, "Analytics summary generated")
    
    except Exception as e:
        return handle_error(f"Analytics error: {str(e)}")

@app.route('/test/sample-predictions', methods=['GET'])
def test_sample_predictions():
    """Test all models with sample users (for testing purposes)"""
    if not ml_models:
        return handle_error("ML models not loaded", 503)
    
    try:
        df = ml_models.df
        
        if df is None or df.empty:
            return handle_error("No data available for testing", 404)
        
        # Get first 3 users for testing
        sample_users = df['User_ID'].head(3).tolist()
        test_results = {}
        
        for user_id in sample_users:
            user_tests = {'user_id': user_id}
            
            try:
                # Test all model predictions
                user_tests['financial_health'] = ml_models.predict_financial_health(user_id)
                user_tests['churn_risk'] = ml_models.predict_churn_risk(user_id)
                user_tests['anomaly_detection'] = ml_models.detect_anomalies(user_id)
                user_tests['fund_recommendations'] = ml_models.recommend_funds(user_id)
                user_tests['monte_carlo'] = ml_models.run_monte_carlo_simulation(user_id, 1000)  # Smaller simulation for testing
                user_tests['peer_matching'] = ml_models.find_similar_peers(user_id)
                user_tests['portfolio_optimization'] = ml_models.optimize_portfolio(user_id)
                
                test_results[user_id] = user_tests
                
            except Exception as e:
                test_results[user_id] = {'error': str(e)}
        
        return success_response({
            'test_completed': True,
            'users_tested': len(sample_users),
            'results': test_results
        }, "Sample predictions test completed")
    
    except Exception as e:
        return handle_error(f"Test error: {str(e)}")

if __name__ == '__main__':
    print("🚀 Starting ML Models API Server...")
    print("📊 Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /models/info - Model information")
    print("  POST /predict/financial-health - Financial health prediction")
    print("  POST /predict/churn-risk - Churn risk prediction")
    print("  POST /predict/anomaly - Anomaly detection")
    print("  POST /recommend/funds - Fund recommendations")
    print("  POST /simulate/monte-carlo - Monte Carlo simulation")
    print("  POST /match/peers - Peer matching")
    print("  POST /optimize/portfolio - Portfolio optimization")
    print("  POST /batch/predict - Batch predictions")
    print("  GET  /analytics/summary - Analytics summary")
    print("  GET  /test/sample-predictions - Test all models")
    print("\n🔧 Starting server on http://localhost:5000")
    
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
