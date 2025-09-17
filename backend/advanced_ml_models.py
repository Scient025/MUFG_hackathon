import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Optional, Tuple
import os
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from supabase_config import supabase, USER_PROFILES_TABLE

class AdvancedMLModels:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.df = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        
        # Load data and initialize models
        self.load_data()
        self.train_all_models()
    
    def load_data(self):
        """Load data from Supabase"""
        try:
            # Fetch all user data from Supabase
            response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
            
            if not response.data:
                raise ValueError("No data found in Supabase database")
            
            # Convert to DataFrame
            self.df = pd.DataFrame(response.data)
            print(f"Advanced ML data loaded: {len(self.df)} users, {len(self.df.columns)} features")
            
            # Handle missing values
            self.df = self.df.fillna({
                'Risk_Tolerance': 'Medium',
                'Investment_Type': 'ETF',
                'Fund_Name': 'Default Fund',
                'Marital_Status': 'Single',
                'Education_Level': 'Bachelor\'s',
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
                'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
                'Years_Contributed', 'Portfolio_Diversity_Score', 'Savings_Rate',
                'Annual_Return_Rate', 'Volatility', 'Fees_Percentage', 'Projected_Pension_Amount',
                'Transaction_Amount', 'Transaction_Pattern_Score', 'Anomaly_Score'
            ]
            
            for col in numeric_columns:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Encode categorical variables
            categorical_columns = [
                'Gender', 'Country', 'Employment_Status', 'Risk_Tolerance',
                'Investment_Type', 'Fund_Name', 'Marital_Status', 'Education_Level',
                'Health_Status', 'Home_Ownership_Status', 'Investment_Experience_Level',
                'Financial_Goals', 'Insurance_Coverage', 'Pension_Type', 'Withdrawal_Strategy',
                'Debt_Level', 'Contribution_Frequency', 'Suspicious_Flag'
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
            print(f"Error loading data from Supabase: {e}")
            raise
    
    def create_derived_features(self):
        """Create derived features for ML models"""
        # Debt-to-Income Ratio
        self.df['DTI_Ratio'] = np.where(
            self.df['Annual_Income'] > 0,
            self.df['Debt_Level'].map({'Low': 0.1, 'Medium': 0.3, 'High': 0.5, 'Unknown': 0.2}) * self.df['Annual_Income'] / self.df['Annual_Income'],
            np.nan
        )
        
        # Savings-to-Income Ratio
        self.df['Savings_to_Income_Ratio'] = np.where(
            self.df['Annual_Income'] > 0,
            self.df['Current_Savings'] / self.df['Annual_Income'],
            0
        )
        
        # Contribution % of Income
        self.df['Contribution_Percent_of_Income'] = np.where(
            self.df['Annual_Income'] > 0,
            (self.df['Contribution_Amount'] * 12) / self.df['Annual_Income'],
            0
        )
        
        # Risk-Adjusted Return
        self.df['Risk_Adjusted_Return'] = np.where(
            self.df['Volatility'] > 0,
            self.df['Annual_Return_Rate'] / self.df['Volatility'],
            0
        )
        
        print("Derived features created successfully")
    
    def train_financial_health_model(self):
        """Train Financial Health Score model (Random Forest)"""
        print("Training Financial Health Score model...")
        
        # Features for financial health
        health_features = [
            'Annual_Income', 'Current_Savings', 'Savings_Rate', 'Debt_Level_encoded',
            'Portfolio_Diversity_Score', 'Contribution_Amount', 'Contribution_Frequency_encoded',
            'Years_Contributed', 'DTI_Ratio', 'Savings_to_Income_Ratio', 'Contribution_Percent_of_Income',
            'Risk_Adjusted_Return', 'Age', 'Investment_Experience_Level_encoded'
        ]
        
        # Create financial health score based on expert rules
        def calculate_health_score(row):
            score = 0
            
            # Income component (20 points)
            if row['Annual_Income'] > 100000:
                score += 20
            elif row['Annual_Income'] > 75000:
                score += 15
            elif row['Annual_Income'] > 50000:
                score += 10
            else:
                score += 5
            
            # Savings component (25 points)
            savings_ratio = row['Savings_to_Income_Ratio']
            if savings_ratio > 2.0:
                score += 25
            elif savings_ratio > 1.0:
                score += 20
            elif savings_ratio > 0.5:
                score += 15
            elif savings_ratio > 0.2:
                score += 10
            else:
                score += 5
            
            # Contribution component (20 points)
            contrib_ratio = row['Contribution_Percent_of_Income']
            if contrib_ratio > 0.15:
                score += 20
            elif contrib_ratio > 0.10:
                score += 15
            elif contrib_ratio > 0.05:
                score += 10
            else:
                score += 5
            
            # Debt component (15 points)
            dti = row['DTI_Ratio']
            if dti < 0.2:
                score += 15
            elif dti < 0.3:
                score += 12
            elif dti < 0.4:
                score += 8
            else:
                score += 3
            
            # Portfolio diversity (10 points)
            diversity = row['Portfolio_Diversity_Score']
            if diversity > 0.8:
                score += 10
            elif diversity > 0.6:
                score += 8
            elif diversity > 0.4:
                score += 5
            else:
                score += 2
            
            # Experience component (10 points)
            experience = row['Investment_Experience_Level_encoded']
            if experience >= 2:  # Expert/Advanced
                score += 10
            elif experience >= 1:  # Intermediate
                score += 7
            else:  # Beginner
                score += 4
            
            return min(100, max(0, score))
        
        # Calculate health scores
        self.df['Financial_Health_Score'] = self.df.apply(calculate_health_score, axis=1)
        
        # Prepare data for Random Forest
        health_data = self.df[health_features + ['Financial_Health_Score']].dropna()
        X = health_data[health_features]
        y = health_data['Financial_Health_Score']
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        
        # Save model
        self.models['financial_health'] = rf_model
        joblib.dump(rf_model, f'{self.models_dir}/financial_health_model.pkl')
        
        print(f"Financial Health model trained on {len(health_data)} samples")
        return rf_model
    
    def train_churn_risk_model(self):
        """Train Churn Risk model (XGBoost Classifier)"""
        print("Training Churn Risk model...")
        
        # Features for churn prediction
        churn_features = [
            'Age', 'Annual_Income', 'Employment_Status_encoded', 'Debt_Level_encoded',
            'Contribution_Frequency_encoded', 'Years_Contributed', 'Savings_Rate',
            'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded',
            'Contribution_Percent_of_Income', 'DTI_Ratio'
        ]
        
        # Create churn labels based on business rules
        def create_churn_label(row):
            churn_score = 0
            
            # 1. Very low contributions
            if row['Contribution_Frequency_encoded'] == 0:  # Low frequency
                churn_score += 1
            # 2. Suspicious activity
            if row.get('Suspicious_Flag', 'No') == 'Yes':
                churn_score += 1
            # 3. Very low contribution percentage
            if row['Contribution_Percent_of_Income'] < 0.02:  # Less than 2%
                churn_score += 1
            # 4. High debt ratio
            if row['DTI_Ratio'] > 0.4:
                churn_score += 1
            # 5. Unemployed
            if row['Employment_Status_encoded'] == 0:  # Assuming 0 = Unemployed
                churn_score += 1
            # 6. Low savings rate
            if row['Savings_Rate'] < 0.05:  # Less than 5%
                churn_score += 1
            # 7. Young age with low contributions
            if row['Age'] < 30 and row['Contribution_Amount'] < 500:
                churn_score += 1
            
            # Return 1 if churn score >= 2, otherwise 0
            return 1 if churn_score >= 2 else 0
        
        # Create churn labels
        self.df['Churn_Risk'] = self.df.apply(create_churn_label, axis=1)
        
        # Prepare data
        churn_data = self.df[churn_features + ['Churn_Risk']].dropna()
        X = churn_data[churn_features]
        y = churn_data['Churn_Risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Churn Risk model accuracy: {accuracy:.3f}")
        
        # Save model
        self.models['churn_risk'] = xgb_model
        joblib.dump(xgb_model, f'{self.models_dir}/churn_risk_model.pkl')
        
        return xgb_model
    
    def train_anomaly_detection_model(self):
        """Train Anomaly Detection model (Isolation Forest)"""
        print("Training Anomaly Detection model...")
        
        # Features for anomaly detection
        anomaly_features = [
            'Transaction_Amount', 'Transaction_Pattern_Score', 'Anomaly_Score',
            'Annual_Income', 'Contribution_Amount', 'Current_Savings',
            'Portfolio_Diversity_Score', 'Savings_Rate'
        ]
        
        # Prepare data
        anomaly_data = self.df[anomaly_features].dropna()
        
        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        iso_forest.fit(anomaly_data)
        
        # Save model
        self.models['anomaly_detection'] = iso_forest
        joblib.dump(iso_forest, f'{self.models_dir}/anomaly_detection_model.pkl')
        
        print(f"Anomaly Detection model trained on {len(anomaly_data)} samples")
        return iso_forest
    
    def train_fund_recommendation_model(self):
        """Train Fund Recommendation model (Collaborative Filtering)"""
        print("Training Fund Recommendation model...")
        
        # Create user-item matrix for collaborative filtering
        fund_data = self.df[['User_ID', 'Fund_Name', 'Investment_Type', 'Risk_Tolerance_encoded', 
                           'Annual_Return_Rate', 'Volatility', 'Fees_Percentage']].dropna()
        
        # Create fund embeddings
        fund_features = ['Annual_Return_Rate', 'Volatility', 'Fees_Percentage']
        fund_embeddings = fund_data.groupby('Fund_Name')[fund_features].mean()
        
        # Create user embeddings
        user_features = ['Age', 'Annual_Income', 'Risk_Tolerance_encoded', 'Investment_Experience_Level_encoded']
        user_embeddings = self.df.groupby('User_ID')[user_features].mean()
        
        # Simple collaborative filtering using KNN
        from sklearn.neighbors import NearestNeighbors
        
        # Create user-fund interaction matrix
        user_fund_matrix = fund_data.pivot_table(
            index='User_ID', 
            columns='Fund_Name', 
            values='Annual_Return_Rate', 
            fill_value=0
        )
        
        # Train KNN for user-based collaborative filtering
        knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_model.fit(user_fund_matrix.fillna(0))
        
        # Save models and data
        self.models['fund_recommendation'] = {
            'knn_model': knn_model,
            'user_fund_matrix': user_fund_matrix,
            'fund_embeddings': fund_embeddings,
            'user_embeddings': user_embeddings
        }
        
        joblib.dump(self.models['fund_recommendation'], f'{self.models_dir}/fund_recommendation_model.pkl')
        
        print(f"Fund Recommendation model trained on {len(fund_data)} interactions")
        return self.models['fund_recommendation']
    
    def train_monte_carlo_model(self):
        """Train Monte Carlo Retirement Stress Test"""
        print("Training Monte Carlo model...")
        
        # This is more of a simulation framework than a traditional ML model
        monte_carlo_config = {
            'n_simulations': 10000,
            'default_return': 7.0,
            'default_volatility': 15.0,
            'inflation_rate': 3.0,
            'fees_rate': 1.0
        }
        
        self.models['monte_carlo'] = monte_carlo_config
        joblib.dump(monte_carlo_config, f'{self.models_dir}/monte_carlo_config.pkl')
        
        print("Monte Carlo configuration saved")
        return monte_carlo_config
    
    def train_peer_matching_model(self):
        """Train Peer Matching model (KNN Similarity Search)"""
        print("Training Peer Matching model...")
        
        # Features for peer matching
        peer_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance_encoded', 'Investment_Experience_Level_encoded', 
            'Portfolio_Diversity_Score', 'Savings_Rate', 'Years_Contributed'
        ]
        
        # Prepare data
        peer_data = self.df[peer_features + ['User_ID']].dropna()
        X = peer_data[peer_features]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train KNN
        knn_model = NearestNeighbors(n_neighbors=10, metric='euclidean')
        knn_model.fit(X_scaled)
        
        # Save model
        self.models['peer_matching'] = {
            'knn_model': knn_model,
            'scaler': scaler,
            'user_ids': peer_data['User_ID'].values,
            'features': peer_features
        }
        
        joblib.dump(self.models['peer_matching'], f'{self.models_dir}/peer_matching_model.pkl')
        
        print(f"Peer Matching model trained on {len(peer_data)} users")
        return self.models['peer_matching']
    
    def train_portfolio_optimization_model(self):
        """Train Portfolio Optimization model (Simplified Markowitz)"""
        print("Training Portfolio Optimization model...")
        
        # Get fund data for optimization
        fund_data = self.df[['Fund_Name', 'Annual_Return_Rate', 'Volatility', 'Fees_Percentage']].dropna()
        fund_stats = fund_data.groupby('Fund_Name').agg({
            'Annual_Return_Rate': 'mean',
            'Volatility': 'mean',
            'Fees_Percentage': 'mean'
        }).reset_index()
        
        # Create covariance matrix (simplified)
        n_funds = len(fund_stats)
        returns = fund_stats['Annual_Return_Rate'].values
        volatilities = fund_stats['Volatility'].values
        
        # Simple covariance matrix based on volatilities
        cov_matrix = np.outer(volatilities, volatilities) * 0.3  # Correlation of 0.3
        np.fill_diagonal(cov_matrix, volatilities ** 2)
        
        # Portfolio optimization configuration
        portfolio_config = {
            'fund_names': fund_stats['Fund_Name'].tolist(),
            'expected_returns': returns,
            'covariance_matrix': cov_matrix,
            'fees': fund_stats['Fees_Percentage'].values,
            'risk_free_rate': 2.0
        }
        
        self.models['portfolio_optimization'] = portfolio_config
        joblib.dump(portfolio_config, f'{self.models_dir}/portfolio_optimization_model.pkl')
        
        print(f"Portfolio Optimization model trained on {n_funds} funds")
        return portfolio_config
    
    def train_all_models(self):
        """Train all advanced ML models"""
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Train all models
        self.train_financial_health_model()
        self.train_churn_risk_model()
        self.train_anomaly_detection_model()
        self.train_fund_recommendation_model()
        self.train_monte_carlo_model()
        self.train_peer_matching_model()
        self.train_portfolio_optimization_model()
        
        # Save label encoders
        joblib.dump(self.label_encoders, f'{self.models_dir}/advanced_label_encoders.pkl')
        
        print("All advanced ML models trained and saved successfully!")
        return self.models
    
    # Prediction methods
    def predict_financial_health(self, user_id: str) -> Dict[str, Any]:
        """Predict financial health score for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Get actual health score
            actual_score = float(user['Financial_Health_Score'])
            
            # Calculate peer percentile
            peer_percentile = float((self.df['Financial_Health_Score'] <= actual_score).mean() * 100)
            
            return {
                'user_id': user_id,
                'financial_health_score': actual_score,
                'peer_percentile': peer_percentile,
                'status': 'Above peers' if peer_percentile > 50 else 'Below peers',
                'recommendations': self.get_health_recommendations(actual_score)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def predict_churn_risk(self, user_id: str) -> Dict[str, Any]:
        """Predict churn risk for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Get churn probability
            churn_features = [
                'Age', 'Annual_Income', 'Employment_Status_encoded', 'Debt_Level_encoded',
                'Contribution_Frequency_encoded', 'Years_Contributed', 'Savings_Rate',
                'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded',
                'Contribution_Percent_of_Income', 'DTI_Ratio'
            ]
            
            X = user[churn_features].values.reshape(1, -1)
            churn_probability = self.models['churn_risk'].predict_proba(X)[0][1]
            
            return {
                'user_id': user_id,
                'churn_probability': float(churn_probability),
                'risk_level': 'High' if churn_probability > 0.5 else 'Medium' if churn_probability > 0.3 else 'Low',
                'recommendations': self.get_churn_recommendations(churn_probability)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_anomalies(self, user_id: str) -> Dict[str, Any]:
        """Detect anomalies for a user using only Supabase-available fields"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Features available in Supabase for anomaly detection
            anomaly_features = [
                'Annual_Income', 'Contribution_Amount', 'Current_Savings',
                'Portfolio_Diversity_Score', 'Savings_Rate', 'Debt_Level',
                'Monthly_Expenses', 'Age'
            ]
            
            # Check if all required features are available
            available_features = []
            feature_values = []
            for feature in anomaly_features:
                if feature in user and pd.notna(user[feature]):
                    available_features.append(feature)
                    feature_values.append(user[feature])
            
            if len(available_features) < 3:
                # Not enough data for anomaly detection
                return {
                    'user_id': user_id,
                    'anomaly_score': 0.0,
                    'is_anomaly': False,
                    'anomaly_percentage': 0.0,
                    'recommendations': ['Insufficient data for anomaly detection']
                }
            
            # Use available features for anomaly detection
            X = np.array(feature_values).reshape(1, -1)
            
            # Simple anomaly detection based on statistical thresholds
            anomaly_score = self.calculate_simple_anomaly_score(user)
            is_anomaly = anomaly_score > 0.7  # Threshold for anomaly
            
            return {
                'user_id': user_id,
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'anomaly_percentage': float(anomaly_score * 100),
                'recommendations': self.get_anomaly_recommendations(is_anomaly)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_simple_anomaly_score(self, user) -> float:
        """Calculate a simple anomaly score based on available Supabase fields"""
        try:
            score = 0.0
            
            # Convert all values to proper numeric types
            income = float(user.get('Annual_Income', 0))
            savings = float(user.get('Current_Savings', 0))
            contribution = float(user.get('Contribution_Amount', 0))
            age = float(user.get('Age', 30))
            diversity = float(user.get('Portfolio_Diversity_Score', 0.5))
            
            # Income vs Savings ratio anomaly
            if income > 0:
                savings_ratio = savings / income
                if savings_ratio > 3.0:  # Very high savings ratio
                    score += 0.3
                elif savings_ratio < 0.1:  # Very low savings ratio
                    score += 0.4
            
            # Contribution vs Income ratio anomaly
            if income > 0:
                contrib_ratio = contribution / income
                if contrib_ratio > 0.2:  # Very high contribution ratio
                    score += 0.2
                elif contrib_ratio < 0.02:  # Very low contribution ratio
                    score += 0.3
            
            # Debt level anomaly - handle categorical debt levels
            debt_level = user.get('Debt_Level', 'Low')
            if isinstance(debt_level, str):
                # Convert categorical debt level to numeric multiplier
                debt_multipliers = {'Low': 0.1, 'Medium': 0.3, 'High': 0.5, 'Unknown': 0.2}
                debt_multiplier = debt_multipliers.get(debt_level, 0.2)
                debt_amount = income * debt_multiplier
            else:
                # If it's already numeric, use it directly
                debt_amount = float(debt_level)
            
            if debt_amount > income * 2:  # Debt more than 2x income
                score += 0.4
            elif debt_amount > income:  # Debt more than income
                score += 0.2
            
            # Age vs Savings anomaly
            if age > 0:
                expected_savings = income * (age - 20) * 0.1  # Rough expectation
                if savings < expected_savings * 0.1:  # Very low savings for age
                    score += 0.3
            
            # Portfolio diversity anomaly
            if diversity < 0.2:  # Very low diversity
                score += 0.2
            elif diversity > 0.95:  # Very high diversity (might be suspicious)
                score += 0.1
            
            return min(1.0, score)  # Cap at 1.0
        except Exception as e:
            print(f"Error calculating anomaly score: {e}")
            return 0.0
    
    def recommend_funds(self, user_id: str, n_recommendations: int = 5) -> Dict[str, Any]:
        """Recommend funds for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Get user's current fund
            current_fund = user['Fund_Name']
            
            # Find similar users
            fund_model = self.models['fund_recommendation']
            user_fund_matrix = fund_model['user_fund_matrix']
            
            if user_id in user_fund_matrix.index:
                user_vector = user_fund_matrix.loc[user_id].values.reshape(1, -1)
                distances, indices = fund_model['knn_model'].kneighbors(user_vector)
                
                # Get recommendations from similar users
                similar_users = user_fund_matrix.index[indices[0]]
                recommendations = []
                
                for similar_user in similar_users:
                    user_funds = user_fund_matrix.loc[similar_user]
                    top_funds = user_funds.nlargest(3).index.tolist()
                    recommendations.extend(top_funds)
                
                # Remove duplicates and current fund
                recommendations = list(set(recommendations))
                if current_fund in recommendations:
                    recommendations.remove(current_fund)
                
                recommendations = recommendations[:n_recommendations]
            else:
                # Fallback: recommend based on risk tolerance
                risk_tolerance = user['Risk_Tolerance']
                fund_embeddings = fund_model['fund_embeddings']
                
                if risk_tolerance == 'High':
                    recommendations = fund_embeddings.nlargest(n_recommendations, 'Annual_Return_Rate').index.tolist()
                elif risk_tolerance == 'Low':
                    recommendations = fund_embeddings.nsmallest(n_recommendations, 'Volatility').index.tolist()
                else:  # Medium
                    recommendations = fund_embeddings.nlargest(n_recommendations, 'Risk_Adjusted_Return').index.tolist()
            
            return {
                'user_id': user_id,
                'current_fund': current_fund,
                'recommendations': recommendations,
                'reasoning': f"Based on users with similar risk profile ({user['Risk_Tolerance']})"
            }
        except Exception as e:
            return {'error': str(e)}
    
    def run_monte_carlo_simulation(self, user_id: str, n_simulations: int = 10000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for retirement planning"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Parameters
            current_age = user['Age']
            retirement_age = user['Retirement_Age_Goal']
            current_savings = user['Current_Savings']
            monthly_contribution = user['Contribution_Amount']
            annual_return = user['Annual_Return_Rate']
            volatility = user['Volatility']
            years_to_retirement = retirement_age - current_age
            
            # Monte Carlo simulation
            np.random.seed(42)
            final_balances = []
            
            for _ in range(n_simulations):
                balance = current_savings
                for year in range(years_to_retirement):
                    # Random return for this year
                    annual_return_sim = np.random.normal(annual_return, volatility)
                    
                    # Add contributions
                    balance += monthly_contribution * 12
                    
                    # Apply return
                    balance *= (1 + annual_return_sim / 100)
                
                final_balances.append(balance)
            
            final_balances = np.array(final_balances)
            
            # Calculate percentiles
            percentiles = {
                'p10': float(np.percentile(final_balances, 10)),
                'p25': float(np.percentile(final_balances, 25)),
                'p50': float(np.percentile(final_balances, 50)),
                'p75': float(np.percentile(final_balances, 75)),
                'p90': float(np.percentile(final_balances, 90))
            }
            
            return {
                'user_id': user_id,
                'simulations': n_simulations,
                'years_to_retirement': years_to_retirement,
                'percentiles': percentiles,
                'mean_balance': float(np.mean(final_balances)),
                'std_balance': float(np.std(final_balances)),
                'probability_above_target': float(np.mean(final_balances >= user.get('Projected_Pension_Amount', 0)))
            }
        except Exception as e:
            return {'error': str(e)}
    
    def find_similar_peers(self, user_id: str, n_peers: int = 5) -> Dict[str, Any]:
        """Find similar peers for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Get user features
            peer_model = self.models['peer_matching']
            user_features = user[peer_model['features']].values.reshape(1, -1)
            
            # Scale features
            user_scaled = peer_model['scaler'].transform(user_features)
            
            # Find similar users
            distances, indices = peer_model['knn_model'].kneighbors(user_scaled)
            
            # Get peer information
            similar_user_ids = peer_model['user_ids'][indices[0]]
            peer_distances = distances[0]
            
            peers = []
            for i, peer_id in enumerate(similar_user_ids):
                if peer_id != user_id:  # Exclude self
                    peer_data = self.df[self.df['User_ID'] == peer_id].iloc[0]
                    peers.append({
                        'user_id': peer_id,
                        'similarity_score': float(1 - peer_distances[i]),  # Convert distance to similarity
                        'age': int(peer_data['Age']),
                        'annual_income': float(peer_data['Annual_Income']),
                        'risk_tolerance': peer_data['Risk_Tolerance'],
                        'contribution_amount': float(peer_data['Contribution_Amount']),
                        'fund_name': peer_data['Fund_Name']
                    })
            
            return {
                'user_id': user_id,
                'peers': peers[:n_peers],
                'total_peers_found': len(peers)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def optimize_portfolio(self, user_id: str) -> Dict[str, Any]:
        """Optimize portfolio allocation for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Get portfolio configuration
            portfolio_config = self.models['portfolio_optimization']
            fund_names = portfolio_config['fund_names']
            expected_returns = portfolio_config['expected_returns']
            cov_matrix = portfolio_config['covariance_matrix']
            risk_free_rate = portfolio_config['risk_free_rate']
            
            # Portfolio optimization function
            def portfolio_performance(weights):
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                return -sharpe_ratio  # Minimize negative Sharpe ratio
            
            # Constraints
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            bounds = tuple((0, 1) for _ in range(len(fund_names)))  # Weights between 0 and 1
            
            # Initial guess
            initial_weights = np.array([1.0 / len(fund_names)] * len(fund_names))
            
            # Optimize
            result = minimize(portfolio_performance, initial_weights, method='SLSQP', 
                           bounds=bounds, constraints=constraints)
            
            # Format results
            optimized_weights = result.x
            portfolio_allocation = []
            
            for i, fund_name in enumerate(fund_names):
                if optimized_weights[i] > 0.01:  # Only include funds with >1% allocation
                    portfolio_allocation.append({
                        'fund_name': fund_name,
                        'allocation_percent': float(optimized_weights[i] * 100),
                        'expected_return': float(expected_returns[i]),
                        'volatility': float(np.sqrt(cov_matrix[i, i]))
                    })
            
            # Calculate portfolio metrics
            portfolio_return = np.sum(optimized_weights * expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimized_weights.T, np.dot(cov_matrix, optimized_weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            return {
                'user_id': user_id,
                'current_fund': user['Fund_Name'],
                'optimized_allocation': portfolio_allocation,
                'portfolio_metrics': {
                    'expected_return': float(portfolio_return),
                    'volatility': float(portfolio_volatility),
                    'sharpe_ratio': float(sharpe_ratio)
                },
                'recommendations': self.get_portfolio_recommendations(portfolio_allocation, user['Fund_Name'])
            }
        except Exception as e:
            return {'error': str(e)}
    
    # Helper methods for recommendations
    def get_health_recommendations(self, score: float) -> List[str]:
        if score >= 80:
            return ["Excellent financial health! Keep up the great work.", "Consider tax optimization strategies."]
        elif score >= 60:
            return ["Good financial health. Consider increasing contributions.", "Review your investment allocation."]
        elif score >= 40:
            return ["Moderate financial health. Increase savings rate.", "Consider reducing debt."]
        else:
            return ["Financial health needs improvement.", "Increase contributions and reduce expenses.", "Consider financial counseling."]
    
    def get_churn_recommendations(self, probability: float) -> List[str]:
        if probability > 0.7:
            return ["High churn risk detected.", "Consider increasing engagement.", "Review contribution frequency."]
        elif probability > 0.4:
            return ["Moderate churn risk.", "Consider automatic contributions.", "Review investment strategy."]
        else:
            return ["Low churn risk.", "Continue current strategy.", "Consider increasing contributions."]
    
    def get_anomaly_recommendations(self, is_anomaly: bool) -> List[str]:
        if is_anomaly:
            return ["Unusual activity detected.", "Review recent transactions.", "Consider contacting support."]
        else:
            return ["No anomalies detected.", "Account activity appears normal."]
    
    def get_portfolio_recommendations(self, allocation: List[Dict], current_fund: str) -> List[str]:
        recommendations = []
        
        # Check if current fund is in optimized allocation
        current_fund_allocation = next((item for item in allocation if item['fund_name'] == current_fund), None)
        
        if current_fund_allocation:
            recommendations.append(f"Your current fund ({current_fund}) is part of the optimized portfolio.")
        else:
            recommendations.append(f"Consider diversifying beyond your current fund ({current_fund}).")
        
        # Add diversification recommendations
        if len(allocation) < 3:
            recommendations.append("Consider diversifying across more funds.")
        
        return recommendations

if __name__ == "__main__":
    # Train all advanced models
    advanced_ml = AdvancedMLModels()
    
    print("\nAdvanced ML Models Training Complete!")
    print("Models saved to 'models/' directory:")
    print("- financial_health_model.pkl: Financial Health Score")
    print("- churn_risk_model.pkl: Churn Risk Prediction")
    print("- anomaly_detection_model.pkl: Anomaly Detection")
    print("- fund_recommendation_model.pkl: Fund Recommendations")
    print("- monte_carlo_config.pkl: Monte Carlo Configuration")
    print("- peer_matching_model.pkl: Peer Matching")
    print("- portfolio_optimization_model.pkl: Portfolio Optimization")
    print("- advanced_label_encoders.pkl: Label Encoders")
