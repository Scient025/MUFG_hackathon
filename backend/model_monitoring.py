#!/usr/bin/env python3
"""
Model Monitoring and Drift Detection Framework
Implements continuous monitoring, drift detection, and automated alerting for deployed models
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    def __init__(self, models_dir: str = "models", monitoring_dir: str = "monitoring"):
        self.models_dir = models_dir
        self.monitoring_dir = monitoring_dir
        self.models = {}
        self.scalers = {}
        self.baseline_stats = {}
        self.performance_history = {}
        self.drift_alerts = []
        
        # Create monitoring directory
        os.makedirs(monitoring_dir, exist_ok=True)
        os.makedirs(f"{monitoring_dir}/reports", exist_ok=True)
        os.makedirs(f"{monitoring_dir}/alerts", exist_ok=True)
        os.makedirs(f"{monitoring_dir}/drift_detection", exist_ok=True)
        
        # Load models and establish baselines
        self.load_models()
        self.establish_baselines()
    
    def load_models(self):
        """Load all deployed models and scalers"""
        print("Loading deployed models...")
        
        model_files = {
            'risk_prediction': 'optimized_risk_prediction_model.pkl',
            'churn_risk': 'optimized_churn_risk_model.pkl',
            'financial_health': 'optimized_financial_health_model.pkl',
            'user_segmentation': 'optimized_user_segmentation_model.pkl'
        }
        
        scaler_files = {
            'risk_prediction': 'optimized_risk_prediction_scaler.pkl',
            'churn_risk': 'optimized_churn_risk_scaler.pkl',
            'financial_health': 'optimized_financial_health_scaler.pkl',
            'user_segmentation': 'optimized_user_segmentation_scaler.pkl'
        }
        
        for model_name, model_file in model_files.items():
            model_path = os.path.join(self.models_dir, model_file)
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"✅ Loaded {model_name} model")
            else:
                print(f"⚠️ Model file not found: {model_path}")
        
        for scaler_name, scaler_file in scaler_files.items():
            scaler_path = os.path.join(self.models_dir, scaler_file)
            if os.path.exists(scaler_path):
                self.scalers[scaler_name] = joblib.load(scaler_path)
                print(f"✅ Loaded {scaler_name} scaler")
            else:
                print(f"⚠️ Scaler file not found: {scaler_path}")
    
    def establish_baselines(self):
        """Establish baseline statistics for drift detection"""
        print("Establishing baseline statistics...")
        
        # Load evaluation metrics if available
        metrics_path = os.path.join(self.models_dir, 'optimized_evaluation_metrics.pkl')
        if os.path.exists(metrics_path):
            self.baseline_stats = joblib.load(metrics_path)
            print("✅ Loaded baseline performance metrics")
        else:
            # Set default baselines
            self.baseline_stats = {
                'risk_prediction': {'test_f1': 0.8, 'test_accuracy': 0.8},
                'churn_risk': {'test_f1': 0.7, 'test_recall': 0.6},
                'financial_health': {'test_r2': 0.7, 'test_rmse': 10.0},
                'user_segmentation': {'silhouette_score': 0.3}
            }
            print("⚠️ Using default baseline metrics")
        
        # Initialize performance history
        for model_name in self.models.keys():
            self.performance_history[model_name] = {
                'timestamps': [],
                'metrics': [],
                'predictions': [],
                'features': []
            }
    
    def monitor_model_performance(self, model_name: str, X_test: np.ndarray, y_test: np.ndarray, 
                                predictions: np.ndarray, features: np.ndarray = None):
        """Monitor model performance and detect degradation"""
        timestamp = datetime.now()
        
        # Calculate current metrics
        current_metrics = self.calculate_metrics(model_name, y_test, predictions)
        
        # Store in history
        self.performance_history[model_name]['timestamps'].append(timestamp)
        self.performance_history[model_name]['metrics'].append(current_metrics)
        self.performance_history[model_name]['predictions'].append(predictions.tolist())
        if features is not None:
            self.performance_history[model_name]['features'].append(features.tolist())
        
        # Check for performance degradation
        alerts = self.check_performance_degradation(model_name, current_metrics)
        
        # Save performance data
        self.save_performance_data(model_name)
        
        return alerts
    
    def calculate_metrics(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate appropriate metrics based on model type"""
        metrics = {}
        
        if model_name in ['risk_prediction', 'churn_risk']:
            # Classification metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['precision_binary'] = precision_score(y_true, y_pred, zero_division=0)
                metrics['recall_binary'] = recall_score(y_true, y_pred, zero_division=0)
                metrics['f1_binary'] = f1_score(y_true, y_pred, zero_division=0)
        
        elif model_name == 'financial_health':
            # Regression metrics
            metrics['r2'] = r2_score(y_true, y_pred)
            metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics['mae'] = np.mean(np.abs(y_true - y_pred))
            metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        elif model_name == 'user_segmentation':
            # Clustering metrics
            from sklearn.metrics import silhouette_score
            if len(np.unique(y_pred)) > 1:
                metrics['silhouette_score'] = silhouette_score(y_true, y_pred)
            else:
                metrics['silhouette_score'] = 0.0
        
        return metrics
    
    def check_performance_degradation(self, model_name: str, current_metrics: Dict[str, float]) -> List[Dict]:
        """Check for performance degradation and generate alerts"""
        alerts = []
        
        if model_name not in self.baseline_stats:
            return alerts
        
        baseline = self.baseline_stats[model_name]
        
        # Define degradation thresholds
        degradation_thresholds = {
            'accuracy': 0.05,  # 5% drop
            'f1': 0.05,
            'f1_binary': 0.05,
            'precision': 0.05,
            'recall': 0.05,
            'r2': 0.1,  # 10% drop
            'silhouette_score': 0.1
        }
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in baseline and metric_name in degradation_thresholds:
                baseline_value = baseline[metric_name]
                threshold = degradation_thresholds[metric_name]
                
                # Check for degradation
                if current_value < baseline_value - threshold:
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'model_name': model_name,
                        'metric': metric_name,
                        'baseline_value': baseline_value,
                        'current_value': current_value,
                        'degradation': baseline_value - current_value,
                        'severity': 'HIGH' if baseline_value - current_value > threshold * 2 else 'MEDIUM',
                        'message': f"Performance degradation detected: {metric_name} dropped from {baseline_value:.4f} to {current_value:.4f}"
                    }
                    alerts.append(alert)
                    self.drift_alerts.append(alert)
        
        return alerts
    
    def detect_data_drift(self, model_name: str, new_features: np.ndarray) -> Dict[str, Any]:
        """Detect data drift using statistical tests"""
        drift_results = {}
        
        if model_name not in self.performance_history:
            return drift_results
        
        # Get historical features
        historical_features = self.performance_history[model_name]['features']
        if len(historical_features) < 10:  # Need sufficient history
            return drift_results
        
        # Convert to numpy array
        historical_data = np.array(historical_features[-100:])  # Last 100 samples
        new_data = new_features.reshape(1, -1)
        
        drift_results['model_name'] = model_name
        drift_results['timestamp'] = datetime.now().isoformat()
        drift_results['drift_detected'] = False
        drift_results['drift_details'] = []
        
        # Statistical tests for each feature
        for feature_idx in range(new_data.shape[1]):
            historical_feature = historical_data[:, feature_idx]
            new_feature = new_data[:, feature_idx]
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(historical_feature, new_feature)
            
            # Jensen-Shannon divergence
            js_div = jensenshannon(historical_feature, new_feature)
            
            # Feature drift detection
            drift_detected = ks_pvalue < 0.05 or js_div > 0.1
            
            if drift_detected:
                drift_results['drift_detected'] = True
                drift_results['drift_details'].append({
                    'feature_index': feature_idx,
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'js_divergence': js_div,
                    'drift_severity': 'HIGH' if js_div > 0.2 else 'MEDIUM'
                })
        
        return drift_results
    
    def detect_concept_drift(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Detect concept drift by monitoring prediction accuracy over time"""
        concept_drift_results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'concept_drift_detected': False,
            'drift_score': 0.0
        }
        
        if model_name not in self.performance_history:
            return concept_drift_results
        
        # Get recent performance history
        recent_metrics = self.performance_history[model_name]['metrics'][-20:]  # Last 20 samples
        
        if len(recent_metrics) < 10:
            return concept_drift_results
        
        # Calculate performance trend
        if model_name in ['risk_prediction', 'churn_risk']:
            metric_values = [m.get('f1', m.get('accuracy', 0)) for m in recent_metrics]
        elif model_name == 'financial_health':
            metric_values = [m.get('r2', 0) for m in recent_metrics]
        else:
            metric_values = [m.get('silhouette_score', 0) for m in recent_metrics]
        
        # Linear regression to detect trend
        x = np.arange(len(metric_values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, metric_values)
        
        # Concept drift if significant negative trend
        if slope < -0.01 and p_value < 0.05:
            concept_drift_results['concept_drift_detected'] = True
            concept_drift_results['drift_score'] = abs(slope)
            concept_drift_results['trend_pvalue'] = p_value
        
        return concept_drift_results
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_monitored': list(self.models.keys()),
            'performance_summary': {},
            'drift_alerts': self.drift_alerts[-50:],  # Last 50 alerts
            'recommendations': []
        }
        
        # Performance summary for each model
        for model_name in self.models.keys():
            if model_name in self.performance_history:
                recent_metrics = self.performance_history[model_name]['metrics'][-10:]  # Last 10 samples
                
                if recent_metrics:
                    # Calculate average performance
                    avg_metrics = {}
                    for metric_name in recent_metrics[0].keys():
                        values = [m[metric_name] for m in recent_metrics if metric_name in m]
                        avg_metrics[metric_name] = np.mean(values)
                    
                    report['performance_summary'][model_name] = avg_metrics
        
        # Generate recommendations
        recommendations = []
        
        # Check for models with poor performance
        for model_name, metrics in report['performance_summary'].items():
            if model_name in ['risk_prediction', 'churn_risk']:
                f1_score = metrics.get('f1', 0)
                if f1_score < 0.7:
                    recommendations.append(f"Consider retraining {model_name} - F1 score is {f1_score:.3f}")
            
            elif model_name == 'financial_health':
                r2_score = metrics.get('r2', 0)
                if r2_score < 0.6:
                    recommendations.append(f"Consider retraining {model_name} - R² score is {r2_score:.3f}")
            
            elif model_name == 'user_segmentation':
                silhouette_score = metrics.get('silhouette_score', 0)
                if silhouette_score < 0.2:
                    recommendations.append(f"Consider retraining {model_name} - Silhouette score is {silhouette_score:.3f}")
        
        # Check for frequent alerts
        if len(self.drift_alerts) > 10:
            recommendations.append("High number of drift alerts detected - investigate data quality")
        
        report['recommendations'] = recommendations
        
        return report
    
    def save_performance_data(self, model_name: str):
        """Save performance data to file"""
        data = {
            'model_name': model_name,
            'performance_history': self.performance_history[model_name],
            'baseline_stats': self.baseline_stats.get(model_name, {}),
            'last_updated': datetime.now().isoformat()
        }
        
        file_path = os.path.join(self.monitoring_dir, f"{model_name}_performance.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def save_monitoring_report(self, report: Dict[str, Any]):
        """Save monitoring report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.monitoring_dir, 'reports', f"monitoring_report_{timestamp}.json")
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Monitoring report saved: {file_path}")
    
    def create_monitoring_dashboard(self):
        """Create visual monitoring dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance trends
        for i, model_name in enumerate(self.models.keys()):
            if model_name in self.performance_history:
                row = i // 2
                col = i % 2
                
                timestamps = self.performance_history[model_name]['timestamps']
                metrics = self.performance_history[model_name]['metrics']
                
                if timestamps and metrics:
                    # Plot primary metric
                    if model_name in ['risk_prediction', 'churn_risk']:
                        metric_values = [m.get('f1', 0) for m in metrics]
                        metric_name = 'F1 Score'
                    elif model_name == 'financial_health':
                        metric_values = [m.get('r2', 0) for m in metrics]
                        metric_name = 'R² Score'
                    else:
                        metric_values = [m.get('silhouette_score', 0) for m in metrics]
                        metric_name = 'Silhouette Score'
                    
                    axes[row, col].plot(timestamps, metric_values, marker='o', linewidth=2)
                    axes[row, col].set_title(f'{model_name.upper()} - {metric_name}')
                    axes[row, col].set_ylabel(metric_name)
                    axes[row, col].tick_params(axis='x', rotation=45)
                    axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.monitoring_dir, 'monitoring_dashboard.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Monitoring dashboard created")
    
    def send_alert(self, alert: Dict[str, Any]):
        """Send alert notification (placeholder for actual implementation)"""
        print(f"ALERT: {alert['message']}")
        print(f"   Model: {alert['model_name']}")
        print(f"   Severity: {alert['severity']}")
        print(f"   Timestamp: {alert['timestamp']}")
        
        # In production, this would send emails, Slack messages, etc.
        # For now, save to file
        alert_file = os.path.join(self.monitoring_dir, 'alerts', 
                                 f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(alert_file, 'w') as f:
            json.dump(alert, f, indent=2)
    
    def run_monitoring_cycle(self, test_data: Dict[str, Any]):
        """Run complete monitoring cycle"""
        print(f"\n🔍 Running monitoring cycle at {datetime.now()}")
        
        all_alerts = []
        
        for model_name, model in self.models.items():
            if model_name in test_data:
                data = test_data[model_name]
                X_test = data['X_test']
                y_test = data['y_test']
                
                # Make predictions
                if model_name in self.scalers:
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                else:
                    X_test_scaled = X_test
                
                predictions = model.predict(X_test_scaled)
                
                # Monitor performance
                alerts = self.monitor_model_performance(model_name, X_test, y_test, predictions, X_test)
                all_alerts.extend(alerts)
                
                # Detect data drift
                drift_results = self.detect_data_drift(model_name, X_test)
                if drift_results.get('drift_detected', False):
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'model_name': model_name,
                        'type': 'data_drift',
                        'severity': 'MEDIUM',
                        'message': f"Data drift detected in {model_name}",
                        'details': drift_results
                    }
                    all_alerts.append(alert)
                
                # Detect concept drift
                concept_drift = self.detect_concept_drift(model_name, y_test, predictions)
                if concept_drift.get('concept_drift_detected', False):
                    alert = {
                        'timestamp': datetime.now().isoformat(),
                        'model_name': model_name,
                        'type': 'concept_drift',
                        'severity': 'HIGH',
                        'message': f"Concept drift detected in {model_name}",
                        'details': concept_drift
                    }
                    all_alerts.append(alert)
        
        # Send alerts
        for alert in all_alerts:
            self.send_alert(alert)
        
        # Generate and save report
        report = self.generate_monitoring_report()
        self.save_monitoring_report(report)
        
        # Create dashboard
        self.create_monitoring_dashboard()
        
        print(f"Monitoring cycle complete. Generated {len(all_alerts)} alerts.")
        
        return report, all_alerts

def create_test_data():
    """Create sample test data for monitoring"""
    np.random.seed(42)
    
    test_data = {}
    
    # Risk prediction test data
    X_risk = np.random.randn(100, 13)  # 13 features
    y_risk = np.random.randint(0, 3, 100)  # 3 classes
    test_data['risk_prediction'] = {'X_test': X_risk, 'y_test': y_risk}
    
    # Churn risk test data
    X_churn = np.random.randn(100, 12)  # 12 features
    y_churn = np.random.randint(0, 2, 100)  # Binary
    test_data['churn_risk'] = {'X_test': X_churn, 'y_test': y_churn}
    
    # Financial health test data
    X_health = np.random.randn(100, 16)  # 16 features
    y_health = np.random.uniform(0, 100, 100)  # Continuous
    test_data['financial_health'] = {'X_test': X_health, 'y_test': y_health}
    
    # User segmentation test data
    X_seg = np.random.randn(100, 11)  # 11 features
    y_seg = np.random.randint(0, 5, 100)  # 5 clusters
    test_data['user_segmentation'] = {'X_test': X_seg, 'y_test': y_seg}
    
    return test_data

def main():
    """Main monitoring execution"""
    print("Starting Model Monitoring and Drift Detection Framework")
    print("="*80)
    
    # Initialize monitor
    monitor = ModelMonitor()
    
    # Create test data
    test_data = create_test_data()
    
    # Run monitoring cycle
    report, alerts = monitor.run_monitoring_cycle(test_data)
    
    print("\n📊 MONITORING SUMMARY:")
    print("-" * 80)
    print(f"Models monitored: {len(monitor.models)}")
    print(f"Alerts generated: {len(alerts)}")
    print(f"Performance history: {sum(len(h['timestamps']) for h in monitor.performance_history.values())} samples")
    
    print("\n🎉 Monitoring framework ready!")
    print("Check the 'monitoring/' directory for reports and dashboards.")

if __name__ == "__main__":
    main()
