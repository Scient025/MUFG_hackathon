import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Optional
import os
from advanced_ml_models import AdvancedMLModels
from supabase_config import supabase, USER_PROFILES_TABLE

class IntegratedMLPipeline:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.advanced_ml = AdvancedMLModels(models_dir)
        self.df = self.advanced_ml.df
        
    def get_comprehensive_user_analysis(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive analysis for a user using all ML models"""
        try:
            # Get all predictions
            financial_health = self.advanced_ml.predict_financial_health(user_id)
            churn_risk = self.advanced_ml.predict_churn_risk(user_id)
            anomaly_detection = self.advanced_ml.detect_anomalies(user_id)
            fund_recommendations = self.advanced_ml.recommend_funds(user_id)
            monte_carlo = self.advanced_ml.run_monte_carlo_simulation(user_id)
            peer_matching = self.advanced_ml.find_similar_peers(user_id)
            portfolio_optimization = self.advanced_ml.optimize_portfolio(user_id)
            
            # Get user profile
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            return {
                'user_id': user_id,
                'user_profile': {
                    'name': user.get('Name', 'Unknown'),
                    'age': int(user['Age']) if user['Age'] is not None else 0,
                    'annual_income': float(user['Annual_Income']) if user['Annual_Income'] is not None else 0.0,
                    'current_savings': float(user['Current_Savings']) if user['Current_Savings'] is not None else 0.0,
                    'risk_tolerance': user['Risk_Tolerance'],
                    'fund_name': user['Fund_Name'],
                    'contribution_amount': float(user['Contribution_Amount']) if user['Contribution_Amount'] is not None else 0.0
                },
                'financial_health': financial_health,
                'churn_risk': churn_risk,
                'anomaly_detection': anomaly_detection,
                'fund_recommendations': fund_recommendations,
                'monte_carlo_simulation': monte_carlo,
                'peer_matching': peer_matching,
                'portfolio_optimization': portfolio_optimization,
                'dashboard_metrics': self.calculate_dashboard_metrics(user_id)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_dashboard_metrics(self, user_id: str) -> Dict[str, Any]:
        """Calculate metrics for dashboard display"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Calculate metrics
            current_balance = user['Current_Savings']
            projected_pension = user.get('Projected_Pension_Amount', 0)
            percent_to_goal = (current_balance / projected_pension * 100) if projected_pension > 0 else 0
            monthly_income_at_retirement = projected_pension / 12 if projected_pension > 0 else 0
            employer_contribution = user['Employer_Contribution']
            total_annual_contribution = (user['Contribution_Amount'] + user['Employer_Contribution']) * 12
            
            # Get financial health score
            financial_health = self.advanced_ml.predict_financial_health(user_id)
            health_score = financial_health.get('financial_health_score', 0)
            
            # Get churn risk
            churn_risk = self.advanced_ml.predict_churn_risk(user_id)
            churn_probability = churn_risk.get('churn_probability', 0)
            
            # Get anomaly detection
            anomaly_detection = self.advanced_ml.detect_anomalies(user_id)
            anomaly_percentage = anomaly_detection.get('anomaly_percentage', 0)
            
            return {
                'current_balance': float(current_balance),
                'percent_to_goal': float(percent_to_goal),
                'monthly_income_at_retirement': float(monthly_income_at_retirement),
                'employer_contribution': float(employer_contribution),
                'total_annual_contribution': float(total_annual_contribution),
                'financial_health_score': float(health_score),
                'churn_risk_percentage': float(churn_probability * 100),
                'anomaly_score': float(anomaly_percentage),
                'goal_progress': {
                    'percentage': float(percent_to_goal),
                    'status': 'On track' if percent_to_goal >= 75 else 'Needs attention' if percent_to_goal >= 50 else 'At risk'
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_risk_analysis(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive risk analysis for a user"""
        try:
            # Get Monte Carlo simulation
            monte_carlo = self.advanced_ml.run_monte_carlo_simulation(user_id)
            
            # Get churn risk
            churn_risk = self.advanced_ml.predict_churn_risk(user_id)
            
            # Get anomaly detection
            anomaly_detection = self.advanced_ml.detect_anomalies(user_id)
            
            # Get portfolio optimization
            portfolio_optimization = self.advanced_ml.optimize_portfolio(user_id)
            
            return {
                'user_id': user_id,
                'monte_carlo_analysis': monte_carlo,
                'churn_risk_analysis': churn_risk,
                'anomaly_analysis': anomaly_detection,
                'portfolio_risk': portfolio_optimization,
                'overall_risk_score': self.calculate_overall_risk_score(churn_risk, anomaly_detection, monte_carlo)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_investment_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get investment recommendations for a user"""
        try:
            # Get fund recommendations
            fund_recommendations = self.advanced_ml.recommend_funds(user_id)
            
            # Get portfolio optimization
            portfolio_optimization = self.advanced_ml.optimize_portfolio(user_id)
            
            # Get peer matching
            peer_matching = self.advanced_ml.find_similar_peers(user_id)
            
            return {
                'user_id': user_id,
                'fund_recommendations': fund_recommendations,
                'portfolio_optimization': portfolio_optimization,
                'peer_insights': peer_matching,
                'action_items': self.generate_action_items(fund_recommendations, portfolio_optimization)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_overall_risk_score(self, churn_risk: Dict, anomaly_detection: Dict, monte_carlo: Dict) -> float:
        """Calculate overall risk score (0-100)"""
        try:
            churn_score = churn_risk.get('churn_probability', 0) * 100
            anomaly_score = anomaly_detection.get('anomaly_percentage', 0)
            
            # Monte Carlo risk based on probability of meeting target
            monte_carlo_prob = monte_carlo.get('probability_above_target', 0.5)
            monte_carlo_risk = (1 - monte_carlo_prob) * 100
            
            # Weighted average
            overall_risk = (churn_score * 0.4 + anomaly_score * 0.3 + monte_carlo_risk * 0.3)
            
            return float(min(100, max(0, overall_risk)))
        except:
            return 50.0  # Default moderate risk
    
    def generate_action_items(self, fund_recommendations: Dict, portfolio_optimization: Dict) -> List[str]:
        """Generate actionable recommendations"""
        action_items = []
        
        # Fund recommendations
        recommendations = fund_recommendations.get('recommendations', [])
        if recommendations:
            action_items.append(f"Consider switching to recommended funds: {', '.join(recommendations[:3])}")
        
        # Portfolio optimization
        allocation = portfolio_optimization.get('optimized_allocation', [])
        if allocation:
            action_items.append("Review your portfolio allocation for better diversification")
        
        # General recommendations
        action_items.append("Review your contribution amount quarterly")
        action_items.append("Consider increasing contributions with salary increases")
        
        return action_items

if __name__ == "__main__":
    # Test the integrated pipeline
    pipeline = IntegratedMLPipeline()
    
    # Test with a sample user
    test_user = pipeline.df['User_ID'].iloc[0]
    print(f"Testing with user: {test_user}")
    
    # Get comprehensive analysis
    analysis = pipeline.get_comprehensive_user_analysis(test_user)
    
    if 'error' not in analysis:
        print("\n=== COMPREHENSIVE USER ANALYSIS ===")
        print(f"User: {analysis['user_profile']['name']}")
        print(f"Financial Health Score: {analysis['financial_health']['financial_health_score']}/100")
        print(f"Churn Risk: {analysis['churn_risk']['churn_probability']:.1%}")
        print(f"Anomaly Score: {analysis['anomaly_detection']['anomaly_percentage']:.1f}%")
        print(f"Fund Recommendations: {len(analysis['fund_recommendations']['recommendations'])} funds")
        print(f"Monte Carlo Simulations: {analysis['monte_carlo_simulation']['simulations']}")
        print(f"Similar Peers Found: {analysis['peer_matching']['total_peers_found']}")
        
        print("\n=== DASHBOARD METRICS ===")
        metrics = analysis['dashboard_metrics']
        print(f"Current Balance: ${metrics['current_balance']:,.0f}")
        print(f"Percent to Goal: {metrics['percent_to_goal']:.1f}%")
        print(f"Financial Health: {metrics['financial_health_score']}/100")
        print(f"Goal Status: {metrics['goal_progress']['status']}")
    else:
        print(f"Error: {analysis['error']}")
