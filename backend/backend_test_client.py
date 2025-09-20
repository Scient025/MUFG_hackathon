import requests
import json
import time
from typing import Dict, Any, List
import pandas as pd

class MLModelsAPIClient:
    """
    Test client for ML Models API
    """
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f"Request failed: {str(e)}",
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        print("🔍 Checking API health...")
        result = self._make_request('GET', '/health')
        self._print_result("Health Check", result)
        return result
    
    def get_models_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        print("📊 Getting models information...")
        result = self._make_request('GET', '/models/info')
        self._print_result("Models Info", result)
        return result
    
    def predict_financial_health(self, user_id: str) -> Dict[str, Any]:
        """Predict financial health for a user"""
        print(f"💰 Predicting financial health for user: {user_id}")
        result = self._make_request('POST', '/predict/financial-health', {'user_id': user_id})
        self._print_result("Financial Health Prediction", result)
        return result
    
    def predict_churn_risk(self, user_id: str) -> Dict[str, Any]:
        """Predict churn risk for a user"""
        print(f"⚠️ Predicting churn risk for user: {user_id}")
        result = self._make_request('POST', '/predict/churn-risk', {'user_id': user_id})
        self._print_result("Churn Risk Prediction", result)
        return result
    
    def detect_anomalies(self, user_id: str) -> Dict[str, Any]:
        """Detect anomalies for a user"""
        print(f"🚨 Detecting anomalies for user: {user_id}")
        result = self._make_request('POST', '/predict/anomaly', {'user_id': user_id})
        self._print_result("Anomaly Detection", result)
        return result
    
    def recommend_funds(self, user_id: str, n_recommendations: int = 5) -> Dict[str, Any]:
        """Get fund recommendations for a user"""
        print(f"🎯 Getting fund recommendations for user: {user_id}")
        result = self._make_request('POST', '/recommend/funds', {
            'user_id': user_id,
            'n_recommendations': n_recommendations
        })
        self._print_result("Fund Recommendations", result)
        return result
    
    def monte_carlo_simulation(self, user_id: str, n_simulations: int = 10000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for a user"""
        print(f"🎲 Running Monte Carlo simulation for user: {user_id}")
        result = self._make_request('POST', '/simulate/monte-carlo', {
            'user_id': user_id,
            'n_simulations': n_simulations
        })
        self._print_result("Monte Carlo Simulation", result)
        return result
    
    def find_peers(self, user_id: str, n_peers: int = 5) -> Dict[str, Any]:
        """Find similar peers for a user"""
        print(f"👥 Finding peers for user: {user_id}")
        result = self._make_request('POST', '/match/peers', {
            'user_id': user_id,
            'n_peers': n_peers
        })
        self._print_result("Peer Matching", result)
        return result
    
    def optimize_portfolio(self, user_id: str) -> Dict[str, Any]:
        """Optimize portfolio for a user"""
        print(f"📈 Optimizing portfolio for user: {user_id}")
        result = self._make_request('POST', '/optimize/portfolio', {'user_id': user_id})
        self._print_result("Portfolio Optimization", result)
        return result
    
    def batch_predictions(self, user_ids: List[str], prediction_types: List[str] = None) -> Dict[str, Any]:
        """Run batch predictions for multiple users"""
        if prediction_types is None:
            prediction_types = ['financial_health', 'churn_risk']
        
        print(f"📦 Running batch predictions for {len(user_ids)} users")
        result = self._make_request('POST', '/batch/predict', {
            'user_ids': user_ids,
            'prediction_types': prediction_types
        })
        self._print_result("Batch Predictions", result)
        return result
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary"""
        print("📊 Getting analytics summary...")
        result = self._make_request('GET', '/analytics/summary')
        self._print_result("Analytics Summary", result)
        return result
    
    def test_sample_predictions(self) -> Dict[str, Any]:
        """Test all models with sample users"""
        print("🧪 Testing all models with sample users...")
        result = self._make_request('GET', '/test/sample-predictions')
        self._print_result("Sample Predictions Test", result)
        return result
    
    def _print_result(self, operation: str, result: Dict[str, Any]):
        """Pretty print API results"""
        print(f"\n{'='*50}")
        print(f"🔸 {operation}")
        print(f"{'='*50}")
        
        if result.get('success'):
            print("✅ Status: SUCCESS")
            if 'message' in result:
                print(f"📝 Message: {result['message']}")
            
            # Print key data points
            if 'data' in result:
                data = result['data']
                self._print_data_summary(data)
        else:
            print("❌ Status: FAILED")
            if 'error' in result:
                print(f"🚫 Error: {result['error']}")
        
        print("\n")
    
    def _print_data_summary(self, data: Dict[str, Any]):
        """Print summary of response data"""
        if not data:
            return
        
        # Financial Health
        if 'financial_health_score' in data:
            print(f"💰 Financial Health Score: {data['financial_health_score']:.1f}/100")
            print(f"📊 Peer Percentile: {data.get('peer_percentile', 0):.1f}%")
        
        # Churn Risk
        if 'churn_probability' in data:
            prob = data['churn_probability']
            print(f"⚠️ Churn Probability: {prob:.2%}")
            print(f"📈 Risk Level: {data.get('risk_level', 'Unknown')}")
        
        # Anomaly Detection
        if 'is_anomaly' in data:
            print(f"🚨 Anomaly Detected: {'Yes' if data['is_anomaly'] else 'No'}")
            print(f"📊 Anomaly Score: {data.get('anomaly_score', 0):.3f}")
        
        # Fund Recommendations
        if 'recommendations' in data and isinstance(data['recommendations'], list):
            print(f"🎯 Recommended Funds: {', '.join(data['recommendations'][:3])}")
        
        # Monte Carlo
        if 'percentiles' in data:
            percentiles = data['percentiles']
            print(f"🎲 Retirement Projections:")
            print(f"   50th percentile: ${percentiles.get('p50', 0):,.0f}")
            print(f"   90th percentile: ${percentiles.get('p90', 0):,.0f}")
        
        # Portfolio Optimization
        if 'optimized_allocation' in data:
            allocation = data['optimized_allocation'][:3]  # Top 3
            print("📈 Top Portfolio Allocations:")
            for fund in allocation:
                print(f"   {fund['fund_name']}: {fund['allocation_percent']:.1f}%")
        
        # Peer Matching
        if 'peers' in data:
            peers = data['peers'][:3]  # Top 3
            print("👥 Similar Peers:")
            for peer in peers:
                print(f"   User {peer['user_id']}: {peer['similarity_score']:.2f} similarity")

def run_comprehensive_test():
    """Run comprehensive test suite"""
    client = MLModelsAPIClient()
    
    print("🚀 Starting Comprehensive ML Models API Test")
    print("="*60)
    
    # 1. Health Check
    health_result = client.health_check()
    if not health_result.get('success'):
        print("❌ API is not healthy. Stopping tests.")
        return
    
    time.sleep(1)
    
    # 2. Get Models Info
    models_info = client.get_models_info()
    
    time.sleep(1)
    
    # 3. Test Sample Predictions (this will test all models)
    sample_test = client.test_sample_predictions()
    
    if sample_test.get('success') and 'data' in sample_test:
        # Extract a user ID for individual tests
        results = sample_test['data'].get('results', {})
        test_user_id = list(results.keys())[0] if results else None
        
        if test_user_id:
            print(f"\n🎯 Running individual tests with user: {test_user_id}")
            
            time.sleep(1)
            
            # Individual model tests
            client.predict_financial_health(test_user_id)
            time.sleep(0.5)
            
            client.predict_churn_risk(test_user_id)
            time.sleep(0.5)
            
            client.detect_anomalies(test_user_id)
            time.sleep(0.5)
            
            client.recommend_funds(test_user_id, 3)
            time.sleep(0.5)
            
            client.monte_carlo_simulation(test_user_id, 1000)  # Smaller for testing
            time.sleep(0.5)
            
            client.find_peers(test_user_id, 3)
            time.sleep(0.5)
            
            client.optimize_portfolio(test_user_id)
            time.sleep(1)
            
            # Batch test
            user_ids = list(results.keys())[:3]
            client.batch_predictions(user_ids, ['financial_health', 'churn_risk'])
            time.sleep(1)
    
    # 4. Analytics Summary
    client.get_analytics_summary()
    
    print("\n🎉 Comprehensive test completed!")

def run_performance_test():
    """Run performance test"""
    client = MLModelsAPIClient()
    
    print("⚡ Starting Performance Test")
    print("="*40)
    
    # Test response times
    endpoints_to_test = [
        ('GET', '/health', None),
        ('GET', '/models/info', None),
        ('GET', '/analytics/summary', None),
    ]
    
    for method, endpoint, data in endpoints_to_test:
        start_time = time.time()
        result = client._make_request(method, endpoint, data)
        end_time = time.time()
        
        response_time = (end_time - start_time) * 1000  # Convert to ms
        status = "✅" if result.get('success') else "❌"
        
        print(f"{status} {method} {endpoint}: {response_time:.2f}ms")

if __name__ == "__main__":
    print("🔧 ML Models API Test Client")
    print("Choose test type:")
    print("1. Comprehensive Test (all endpoints)")
    print("2. Performance Test (response times)")
    print("3. Custom Test (interactive)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        run_comprehensive_test()
    elif choice == "2":
        run_performance_test()
    elif choice == "3":
        # Interactive testing
        client = MLModelsAPIClient()
        
        print("\n🎯 Interactive Testing Mode")
        print("Available commands:")
        print("  health - Check API health")
        print("  info - Get models information")
        print("  predict <user_id> - Run all predictions for a user")
        print("  financial <user_id> - Financial health prediction")
        print("  churn <user_id> - Churn risk prediction")
        print("  anomaly <user_id> - Anomaly detection")
        print("  funds <user_id> - Fund recommendations")
        print("  monte <user_id> - Monte Carlo simulation")
        print("  peers <user_id> - Find similar peers")
        print("  portfolio <user_id> - Portfolio optimization")
        print("  batch <user1,user2,user3> - Batch predictions")
        print("  analytics - Analytics summary")
        print("  test - Test with sample users")
        print("  quit - Exit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit" or command == "exit":
                    break
                elif command == "health":
                    client.health_check()
                elif command == "info":
                    client.get_models_info()
                elif command == "analytics":
                    client.get_analytics_summary()
                elif command == "test":
                    client.test_sample_predictions()
                elif command.startswith("predict "):
                    user_id = command.split(" ", 1)[1]
                    print(f"🔄 Running all predictions for user: {user_id}")
                    client.predict_financial_health(user_id)
                    client.predict_churn_risk(user_id)
                    client.detect_anomalies(user_id)
                    client.recommend_funds(user_id)
                    client.find_peers(user_id)
                elif command.startswith("financial "):
                    user_id = command.split(" ", 1)[1]
                    client.predict_financial_health(user_id)
                elif command.startswith("churn "):
                    user_id = command.split(" ", 1)[1]
                    client.predict_churn_risk(user_id)
                elif command.startswith("anomaly "):
                    user_id = command.split(" ", 1)[1]
                    client.detect_anomalies(user_id)
                elif command.startswith("funds "):
                    user_id = command.split(" ", 1)[1]
                    client.recommend_funds(user_id)
                elif command.startswith("monte "):
                    user_id = command.split(" ", 1)[1]
                    client.monte_carlo_simulation(user_id, 1000)
                elif command.startswith("peers "):
                    user_id = command.split(" ", 1)[1]
                    client.find_peers(user_id)
                elif command.startswith("portfolio "):
                    user_id = command.split(" ", 1)[1]
                    client.optimize_portfolio(user_id)
                elif command.startswith("batch "):
                    user_ids_str = command.split(" ", 1)[1]
                    user_ids = [uid.strip() for uid in user_ids_str.split(",")]
                    client.batch_predictions(user_ids)
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    else:
        print("❌ Invalid choice. Please run again and select 1, 2, or 3.")