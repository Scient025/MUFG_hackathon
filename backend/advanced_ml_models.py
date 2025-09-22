import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any
import os
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
import xgboost as xgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from supabase_config import supabase, USER_PROFILES_TABLE


def safe_num(x):
    """Convert numpy and pandas types into plain Python types for JSON serialization."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if hasattr(x, 'item'):  # NumPy scalar
        return x.item()
    return x


class AdvancedMLModels:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.df = None
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}

        self.load_data()
        self.train_all_models()

    # ----------------------
    # Data Loading & Features
    # ----------------------
    def load_data(self):
        response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
        if not response.data:
            raise ValueError("No data found in Supabase database")

        self.df = pd.DataFrame(response.data)
        print(f"Advanced ML data loaded: {len(self.df)} users, {len(self.df.columns)} features")

        # Fill defaults
        self.df = self.df.fillna({
            'Risk_Tolerance': 'Medium', 'Investment_Type': 'ETF', 'Fund_Name': 'Default Fund',
            'Marital_Status': 'Single', 'Education_Level': "Bachelor's", 'Health_Status': 'Average',
            'Annual_Income': 0, 'Current_Savings': 0, 'Contribution_Amount': 0, 'Years_Contributed': 0,
            'Age': 30, 'Portfolio_Diversity_Score': 0.5, 'Savings_Rate': 0.1, 'Annual_Return_Rate': 5.0,
            'Volatility': 2.0, 'Fees_Percentage': 1.0, 'Projected_Pension_Amount': 0,
            'Debt_Level': 'Low', 'Employment_Status': 'Full-time', 'Investment_Experience_Level': 'Beginner',
            'Contribution_Frequency': 'Monthly', 'Transaction_Amount': 0, 'Transaction_Pattern_Score': 0.5,
            'Anomaly_Score': 0.1, 'Suspicious_Flag': 'No'
        })

        # Convert numeric columns
        numeric_columns = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount', 'Years_Contributed',
            'Portfolio_Diversity_Score', 'Savings_Rate', 'Annual_Return_Rate', 'Volatility', 'Fees_Percentage',
            'Projected_Pension_Amount', 'Transaction_Amount', 'Transaction_Pattern_Score', 'Anomaly_Score'
        ]
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Encode categoricals
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

        self.create_derived_features()

    def create_derived_features(self):
        self.df['DTI_Ratio'] = np.where(
            self.df['Annual_Income'] > 0,
            self.df['Debt_Level'].map({'Low': 0.1, 'Medium': 0.3, 'High': 0.5, 'Unknown': 0.2}) * self.df['Annual_Income'] / self.df['Annual_Income'],
            np.nan
        )
        self.df['Savings_to_Income_Ratio'] = np.where(
            self.df['Annual_Income'] > 0,
            self.df['Current_Savings'] / self.df['Annual_Income'], 0
        )
        self.df['Contribution_Percent_of_Income'] = np.where(
            self.df['Annual_Income'] > 0,
            (self.df['Contribution_Amount'] * 12) / self.df['Annual_Income'], 0
        )
        self.df['Risk_Adjusted_Return'] = np.where(
            self.df['Volatility'] > 0,
            self.df['Annual_Return_Rate'] / self.df['Volatility'], 0
        )
        print("Derived features created successfully")

    # ----------------------
    # Recommendation Helpers
    # ----------------------
    def get_health_recommendations(self, score: float) -> list:
        if score >= 80:
            return ["Maintain current savings habits", "Consider diversifying investments further"]
        elif score >= 60:
            return ["Increase savings rate", "Review debt levels and reduce high-interest debt"]
        elif score >= 40:
            return ["Set a monthly budget", "Start an emergency fund", "Increase retirement contributions"]
        else:
            return ["Seek financial advice", "Focus on paying off debt", "Automate savings to build discipline"]

    def get_churn_recommendations(self, probability: float) -> list:
        if probability > 0.5:
            return ["Reach out with personalized offers", "Provide financial planning assistance"]
        elif probability > 0.3:
            return ["Send reminders about contributions", "Offer small incentives to stay engaged"]
        else:
            return ["Keep engagement high with educational content"]

    def get_anomaly_recommendations(self, is_anomaly: bool) -> list:
        if is_anomaly:
            return ["Investigate unusual account activity", "Verify recent transactions", "Alert compliance team if needed"]
        else:
            return ["No unusual activity detected", "Continue regular monitoring"]

    def get_portfolio_recommendations(self, allocation: list, current_fund: str) -> list:
        recs = []
        for fund in allocation:
            if fund["allocation_percent"] > 30:
                recs.append(f"Consider balancing {fund['fund_name']} as it exceeds 30% allocation")
        if current_fund not in [f["fund_name"] for f in allocation]:
            recs.append(f"Your current fund {current_fund} is not in the optimized portfolio — consider switching")
        return recs or ["Your portfolio looks balanced"]

    # ----------------------
    # Model Training
    # ----------------------
    def train_financial_health_model(self):
        print("Training Financial Health Score model...")
        health_features = [
            'Annual_Income','Current_Savings','Savings_Rate','Debt_Level_encoded','Portfolio_Diversity_Score',
            'Contribution_Amount','Contribution_Frequency_encoded','Years_Contributed','DTI_Ratio',
            'Savings_to_Income_Ratio','Contribution_Percent_of_Income','Risk_Adjusted_Return','Age','Investment_Experience_Level_encoded']

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

        self.df['Financial_Health_Score'] = self.df.apply(calculate_health_score, axis=1)
        health_data = self.df[health_features + ['Financial_Health_Score']].dropna()
        X = health_data[health_features]
        y = health_data['Financial_Health_Score']
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        self.models['financial_health'] = rf_model
        joblib.dump(rf_model, f'{self.models_dir}/financial_health_model.pkl')
        print(f"Financial Health model trained on {len(health_data)} samples")
        return rf_model

    def train_churn_risk_model(self):
        print("Training Churn Risk model...")
        churn_features = [
            'Age','Annual_Income','Employment_Status_encoded','Debt_Level_encoded','Contribution_Frequency_encoded',
            'Years_Contributed','Savings_Rate','Portfolio_Diversity_Score','Investment_Experience_Level_encoded',
            'Contribution_Percent_of_Income','DTI_Ratio']

        def create_churn_label(row):
            churn_score = 0
            if row['Contribution_Frequency_encoded'] == 0: churn_score += 1
            if row.get('Suspicious_Flag', 'No') == 'Yes': churn_score += 1
            if row['Contribution_Percent_of_Income'] < 0.02: churn_score += 1
            if row['DTI_Ratio'] > 0.4: churn_score += 1
            if row['Employment_Status_encoded'] == 0: churn_score += 1
            if row['Savings_Rate'] < 0.05: churn_score += 1
            if row['Age'] < 30 and row['Contribution_Amount'] < 500: churn_score += 1
            return 1 if churn_score >= 2 else 0

        self.df['Churn_Risk'] = self.df.apply(create_churn_label, axis=1)
        churn_data = self.df[churn_features + ['Churn_Risk']].dropna()
        X = churn_data[churn_features]
        y = churn_data['Churn_Risk']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Churn Risk model accuracy: {accuracy:.3f}")
        self.models['churn_risk'] = xgb_model
        joblib.dump(xgb_model, f'{self.models_dir}/churn_risk_model.pkl')
        return xgb_model

    def train_anomaly_detection_model(self):
        print("Training Anomaly Detection model...")
        anomaly_features = ['Transaction_Amount','Transaction_Pattern_Score','Anomaly_Score','Annual_Income','Contribution_Amount','Current_Savings','Portfolio_Diversity_Score','Savings_Rate']
        anomaly_data = self.df[anomaly_features].dropna()

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(anomaly_data)

        # Simple parameter sweep targeting a desired contamination rate ~8%
        param_grid = {
            'contamination': [0.05, 0.08, 0.1],
            'n_estimators': [100, 200],
            'max_samples': ['auto', 0.8],
            'max_features': [0.8, 1.0]
        }

        best_model = None
        best_score = -np.inf
        for params in ParameterGrid(param_grid):
            model = IsolationForest(random_state=42, **params)
            preds = model.fit_predict(X_scaled)
            anomaly_rate = (preds == -1).mean()
            score = -abs(anomaly_rate - 0.08)  # closer to 8% anomalies is better
            if score > best_score:
                best_score = score
                best_model = model

        self.models['anomaly_detection'] = {'model': best_model, 'scaler': scaler, 'features': anomaly_features}
        joblib.dump(self.models['anomaly_detection'], f'{self.models_dir}/anomaly_detection_model.pkl')
        print(f"Anomaly Detection model trained on {len(anomaly_data)} samples with best score {best_score:.4f}")
        return self.models['anomaly_detection']

    def train_fund_recommendation_model(self):
        print("Training Fund Recommendation model...")
        fund_data = self.df[['User_ID','Fund_Name','Investment_Type','Risk_Tolerance_encoded','Annual_Return_Rate','Volatility','Fees_Percentage']].dropna()
        fund_features = ['Annual_Return_Rate','Volatility','Fees_Percentage']
        fund_embeddings = fund_data.groupby('Fund_Name')[fund_features].mean()
        user_features = ['Age','Annual_Income','Risk_Tolerance_encoded','Investment_Experience_Level_encoded']
        user_embeddings = self.df.groupby('User_ID')[user_features].mean()
        user_fund_matrix = fund_data.pivot_table(index='User_ID', columns='Fund_Name', values='Annual_Return_Rate', fill_value=0)
        knn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn_model.fit(user_fund_matrix.fillna(0))
        self.models['fund_recommendation'] = {'knn_model': knn_model,'user_fund_matrix': user_fund_matrix,'fund_embeddings': fund_embeddings,'user_embeddings': user_embeddings}
        joblib.dump(self.models['fund_recommendation'], f'{self.models_dir}/fund_recommendation_model.pkl')
        print(f"Fund Recommendation model trained on {len(fund_data)} interactions")
        return self.models['fund_recommendation']

    def train_monte_carlo_model(self):
        print("Training Monte Carlo model...")
        monte_carlo_config = {'n_simulations': 10000,'default_return': 7.0,'default_volatility': 15.0,'inflation_rate': 3.0,'fees_rate': 1.0}
        self.models['monte_carlo'] = monte_carlo_config
        joblib.dump(monte_carlo_config, f'{self.models_dir}/monte_carlo_config.pkl')
        print("Monte Carlo configuration saved")
        return monte_carlo_config

    def train_peer_matching_model(self):
        print("Training Peer Matching model...")
        peer_features = ['Age','Annual_Income','Current_Savings','Contribution_Amount','Risk_Tolerance_encoded','Investment_Experience_Level_encoded','Portfolio_Diversity_Score','Savings_Rate','Years_Contributed']
        peer_data = self.df[peer_features + ['User_ID']].dropna()
        X = peer_data[peer_features].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        metrics = ['euclidean', 'manhattan', 'cosine']
        k_values = [3, 5, 7, 10, 15]
        best_cfg = None
        best_score = np.inf
        best_knn = None

        for metric in metrics:
            for k in k_values:
                knn = NearestNeighbors(n_neighbors=k, metric=metric)
                knn.fit(X_scaled)
                distances, _ = knn.kneighbors(X_scaled)
                avg_dist = distances.mean()
                if avg_dist < best_score:
                    best_score = avg_dist
                    best_cfg = {'metric': metric, 'k': k}
                    best_knn = knn

        self.models['peer_matching'] = {
            'knn_model': best_knn,
            'scaler': scaler,
            'user_ids': peer_data['User_ID'].values,
            'features': peer_features,
            'config': best_cfg,
            'avg_distance': float(best_score)
        }
        joblib.dump(self.models['peer_matching'], f'{self.models_dir}/peer_matching_model.pkl')
        print(f"Peer Matching model trained on {len(peer_data)} users with best cfg {best_cfg} (avg_dist={best_score:.4f})")
        return self.models['peer_matching']

    def train_investment_recommendation_model(self):
        """Train an investment recommendation/propensity regressor with engineered features."""
        print("Training Investment Recommendation model (Random Forest)...")

        df = self.df.copy()
        # Basic engineered features (safe divisions)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = df['Current_Savings'] / df['Annual_Income']
        df['Savings_Income_Ratio'] = np.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)
        df['Age_Income_Interaction'] = (df['Age'] or 0) * (df['Annual_Income'] or 0) if isinstance(df, pd.Series) else df['Age'] * df['Annual_Income']
        df['Financial_Stability'] = (df['Savings_Rate'] or 0) * (df['Portfolio_Diversity_Score'] or 0) if isinstance(df, pd.Series) else df['Savings_Rate'] * df['Portfolio_Diversity_Score']

        target = 'Current_Savings'
        # Use numeric features only
        feature_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
        X = df[feature_cols].fillna(0)
        y = df[target].fillna(0)
        y_log = np.log1p(y)

        if len(X) < 10:
            print("Not enough data to train investment recommendation model")
            return None

        X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        r2 = r2_score(y_test, model.predict(X_test))

        self.models['investment_recommendation'] = {'model': model, 'feature_cols': feature_cols, 'r2': float(r2)}
        joblib.dump(self.models['investment_recommendation'], f'{self.models_dir}/investment_recommendation_model.pkl')
        print(f"Investment Recommendation model trained (R^2={r2:.3f}) with best params: {grid.best_params_}")
        return self.models['investment_recommendation']

    def train_portfolio_optimization_model(self):
        print("Training Portfolio Optimization model...")
        fund_data = self.df[['Fund_Name','Annual_Return_Rate','Volatility','Fees_Percentage']].dropna()
        fund_stats = fund_data.groupby('Fund_Name').agg({'Annual_Return_Rate':'mean','Volatility':'mean','Fees_Percentage':'mean'}).reset_index()
        n_funds = len(fund_stats)
        returns = fund_stats['Annual_Return_Rate'].values
        volatilities = fund_stats['Volatility'].values
        cov_matrix = np.outer(volatilities, volatilities) * 0.3
        np.fill_diagonal(cov_matrix, volatilities ** 2)
        portfolio_config = {'fund_names': fund_stats['Fund_Name'].tolist(),'expected_returns': returns,'covariance_matrix': cov_matrix,'fees': fund_stats['Fees_Percentage'].values,'risk_free_rate': 2.0}
        self.models['portfolio_optimization'] = portfolio_config
        joblib.dump(portfolio_config, f'{self.models_dir}/portfolio_optimization_model.pkl')
        print(f"Portfolio Optimization model trained on {n_funds} funds")
        return portfolio_config

    def train_all_models(self):
        os.makedirs(self.models_dir, exist_ok=True)
        self.train_financial_health_model()
        self.train_churn_risk_model()
        self.train_anomaly_detection_model()
        self.train_fund_recommendation_model()
        self.train_monte_carlo_model()
        self.train_peer_matching_model()
        self.train_portfolio_optimization_model()
        # Optional: train investment recommendation regressor
        try:
            self.train_investment_recommendation_model()
        except Exception as e:
            print(f"Investment recommendation training skipped: {e}")

    # ----------------------
    # Prediction Methods
    # ----------------------
    def predict_financial_health(self, user_id: str) -> Dict[str, Any]:
        """Predict financial health score for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Use the same features as training
            health_features = [
                'Annual_Income','Current_Savings','Savings_Rate','Debt_Level_encoded','Portfolio_Diversity_Score',
                'Contribution_Amount','Contribution_Frequency_encoded','Years_Contributed','DTI_Ratio',
                'Savings_to_Income_Ratio','Contribution_Percent_of_Income','Risk_Adjusted_Return','Age','Investment_Experience_Level_encoded'
            ]
            
            # Prepare features for prediction
            X = user[health_features].values.reshape(1, -1)
            
            # Predict using trained model
            if 'financial_health' in self.models:
                score = self.models['financial_health'].predict(X)[0]
            else:
                # Fallback calculation
                score = self.calculate_financial_health_score(user)
            
            recommendations = self.get_health_recommendations(score)
            
            return {
                'financial_health_score': safe_num(score),
                'recommendations': recommendations,
                'status': 'Excellent' if score >= 80 else 'Good' if score >= 60 else 'Fair' if score >= 40 else 'Poor'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_financial_health_score(self, user: pd.Series) -> float:
        """Calculate financial health score using the same logic as training"""
        score = 0
        
        # Income component (20 points)
        income = user['Annual_Income'] or 0
        if income > 100000: score += 20
        elif income > 75000: score += 15
        elif income > 50000: score += 10
        else: score += 5
        
        # Savings component (25 points)
        savings_ratio = user['Savings_to_Income_Ratio'] or 0
        if savings_ratio > 2.0: score += 25
        elif savings_ratio > 1.0: score += 20
        elif savings_ratio > 0.5: score += 15
        elif savings_ratio > 0.2: score += 10
        else: score += 5
        
        # Contribution component (20 points)
        contrib_ratio = user['Contribution_Percent_of_Income'] or 0
        if contrib_ratio > 0.15: score += 20
        elif contrib_ratio > 0.10: score += 15
        elif contrib_ratio > 0.05: score += 10
        else: score += 5
        
        # Debt component (15 points)
        dti = user['DTI_Ratio'] or 0
        if dti < 0.2: score += 15
        elif dti < 0.3: score += 12
        elif dti < 0.4: score += 8
        else: score += 3
        
        # Diversity component (10 points)
        diversity = user['Portfolio_Diversity_Score'] or 0
        if diversity > 0.8: score += 10
        elif diversity > 0.6: score += 8
        elif diversity > 0.4: score += 5
        else: score += 2
        
        # Experience component (10 points)
        experience = user['Investment_Experience_Level_encoded'] or 0
        if experience >= 2: score += 10
        elif experience >= 1: score += 7
        else: score += 4
        
        return min(100, max(0, score))

    def predict_churn_risk(self, user_id: str) -> Dict[str, Any]:
        """Predict churn risk for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Use the same features as training
            churn_features = [
                'Age','Annual_Income','Employment_Status_encoded','Debt_Level_encoded','Contribution_Frequency_encoded',
                'Years_Contributed','Savings_Rate','Portfolio_Diversity_Score','Investment_Experience_Level_encoded',
                'Contribution_Percent_of_Income','DTI_Ratio'
            ]
            
            # Prepare features for prediction
            X = user[churn_features].values.reshape(1, -1)
            
            # Predict using trained model
            if 'churn_risk' in self.models:
                probability = self.models['churn_risk'].predict_proba(X)[0][1]
            else:
                # Fallback calculation
                probability = self.calculate_churn_probability(user)
            
            recommendations = self.get_churn_recommendations(probability)
            
            return {
                'churn_probability': safe_num(probability),
                'risk_level': 'High' if probability > 0.5 else 'Medium' if probability > 0.3 else 'Low',
                'recommendations': recommendations
            }
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_churn_probability(self, user: pd.Series) -> float:
        """Calculate churn probability using the same logic as training"""
        churn_score = 0
        
        # Contribution frequency
        if user['Contribution_Frequency_encoded'] == 0: churn_score += 1
        
        # Suspicious flag
        if user.get('Suspicious_Flag', 'No') == 'Yes': churn_score += 1
        
        # Contribution percentage
        if user['Contribution_Percent_of_Income'] < 0.02: churn_score += 1
        
        # DTI ratio
        if user['DTI_Ratio'] > 0.4: churn_score += 1
        
        # Employment status
        if user['Employment_Status_encoded'] == 0: churn_score += 1
        
        # Savings rate
        if user['Savings_Rate'] < 0.05: churn_score += 1
        
        # Age and contribution amount
        if user['Age'] < 30 and user['Contribution_Amount'] < 500: churn_score += 1
        
        # Convert score to probability
        return min(1.0, churn_score / 7.0)

    def detect_anomalies(self, user_id: str) -> Dict[str, Any]:
        """Detect anomalies for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            # Use the same features as training
            anomaly_features = ['Transaction_Amount','Transaction_Pattern_Score','Anomaly_Score','Annual_Income','Contribution_Amount','Current_Savings','Portfolio_Diversity_Score','Savings_Rate']
            
            # Prepare features for prediction
            X = user[anomaly_features].values.reshape(1, -1)
            
            # Predict using trained model
            if 'anomaly_detection' in self.models:
                anomaly_score = self.models['anomaly_detection'].decision_function(X)[0]
                is_anomaly = self.models['anomaly_detection'].predict(X)[0] == -1
            else:
                # Fallback calculation
                anomaly_score = user.get('Anomaly_Score', 0.1)
                is_anomaly = anomaly_score > 0.5
            
            recommendations = self.get_anomaly_recommendations(is_anomaly)
            
            return {
                'anomaly_score': safe_num(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'anomaly_percentage': safe_num(abs(anomaly_score) * 100),
                'recommendations': recommendations
            }
        except Exception as e:
            return {'error': str(e)}

    def recommend_funds(self, user_id: str) -> Dict[str, Any]:
        """Recommend funds for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            if 'fund_recommendation' not in self.models:
                return {'error': 'Fund recommendation model not trained'}
            
            model_data = self.models['fund_recommendation']
            user_fund_matrix = model_data['user_fund_matrix']
            
            if user_id not in user_fund_matrix.index:
                # Return default recommendations
                return {
                    'recommendations': ['Default Fund', 'Conservative Fund', 'Growth Fund'],
                    'reasoning': 'No historical data available, using default recommendations'
                }
            
            # Find similar users
            user_vector = user_fund_matrix.loc[user_id].values.reshape(1, -1)
            distances, indices = model_data['knn_model'].kneighbors(user_vector)
            
            # Get fund recommendations from similar users
            similar_users = user_fund_matrix.index[indices[0]]
            recommendations = []
            
            for similar_user in similar_users:
                user_funds = user_fund_matrix.loc[similar_user]
                top_funds = user_funds.nlargest(3)
                recommendations.extend(top_funds.index.tolist())
            
            # Remove duplicates and limit to 5
            unique_recommendations = list(dict.fromkeys(recommendations))[:5]
            
            return {
                'recommendations': unique_recommendations,
                'reasoning': f'Based on {len(similar_users)} similar users',
                'similar_users_count': len(similar_users)
            }
        except Exception as e:
            return {'error': str(e)}

    def run_monte_carlo_simulation(self, user_id: str) -> Dict[str, Any]:
        """Run Monte Carlo simulation for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            if 'monte_carlo' not in self.models:
                return {'error': 'Monte Carlo model not configured'}
            
            config = self.models['monte_carlo']
            n_simulations = config['n_simulations']
            annual_return = config['default_return']
            volatility = config['default_volatility']
            
            # User-specific parameters
            current_savings = user['Current_Savings'] or 0
            annual_contribution = (user['Contribution_Amount'] or 0) * 12
            years_to_retirement = (user['Retirement_Age_Goal'] or 65) - (user['Age'] or 30)
            
            # Run simulation
            results = []
            for _ in range(n_simulations):
                balance = current_savings
                for year in range(years_to_retirement):
                    # Add contribution
                    balance += annual_contribution
                    # Apply return with volatility
                    return_rate = np.random.normal(annual_return, volatility) / 100
                    balance *= (1 + return_rate)
                results.append(balance)
            
            results = np.array(results)
            
            # Calculate statistics
            mean_result = np.mean(results)
            median_result = np.median(results)
            percentile_10 = np.percentile(results, 10)
            percentile_90 = np.percentile(results, 90)
            
            # Probability of meeting target
            target = user.get('Projected_Pension_Amount', mean_result)
            probability_above_target = np.mean(results >= target)
            
            return {
                'simulations': safe_num(n_simulations),
                'mean_result': safe_num(mean_result),
                'median_result': safe_num(median_result),
                'percentile_10': safe_num(percentile_10),
                'percentile_90': safe_num(percentile_90),
                'probability_above_target': safe_num(probability_above_target),
                'years_simulated': safe_num(years_to_retirement)
            }
        except Exception as e:
            return {'error': str(e)}

    def find_similar_peers(self, user_id: str) -> Dict[str, Any]:
        """Find similar peers for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            if 'peer_matching' not in self.models:
                return {'error': 'Peer matching model not trained'}
            
            model_data = self.models['peer_matching']
            peer_features = model_data['features']
            
            # Prepare user features
            user_features = user[peer_features].values.reshape(1, -1)
            
            # Scale features
            user_features_scaled = model_data['scaler'].transform(user_features)
            
            # Find similar users
            distances, indices = model_data['knn_model'].kneighbors(user_features_scaled)
            
            # Get peer information
            peer_ids = model_data['user_ids'][indices[0]]
            peer_distances = distances[0]
            
            # Calculate peer statistics
            peer_data = self.df[self.df['User_ID'].isin(peer_ids)]
            
            return {
                'total_peers_found': safe_num(len(peer_ids)),
                'similar_users': peer_ids.tolist(),
                'distances': peer_distances.tolist(),
                'peer_stats': {
                    'avg_age': safe_num(peer_data['Age'].mean()),
                    'avg_income': safe_num(peer_data['Annual_Income'].mean()),
                    'avg_savings': safe_num(peer_data['Current_Savings'].mean()),
                    'avg_contribution': safe_num(peer_data['Contribution_Amount'].mean())
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def optimize_portfolio(self, user_id: str) -> Dict[str, Any]:
        """Optimize portfolio for a user"""
        try:
            user_data = self.df[self.df['User_ID'] == user_id]
            if user_data.empty:
                return {'error': 'User not found'}
            
            user = user_data.iloc[0]
            
            if 'portfolio_optimization' not in self.models:
                return {'error': 'Portfolio optimization model not configured'}
            
            config = self.models['portfolio_optimization']
            fund_names = config['fund_names']
            expected_returns = config['expected_returns']
            cov_matrix = config['covariance_matrix']
            
            # User risk tolerance
            risk_tolerance = user.get('Risk_Tolerance', 'Medium')
            risk_multiplier = {'Low': 0.5, 'Medium': 1.0, 'High': 1.5}.get(risk_tolerance, 1.0)
            
            # Simple optimization (equal weight for now)
            n_funds = len(fund_names)
            equal_weights = np.ones(n_funds) / n_funds
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(equal_weights, expected_returns)
            portfolio_variance = np.dot(equal_weights, np.dot(cov_matrix, equal_weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Create allocation
            allocation = []
            for i, fund_name in enumerate(fund_names):
                allocation.append({
                    'fund_name': fund_name,
                    'allocation_percent': safe_num(equal_weights[i] * 100),
                    'expected_return': safe_num(expected_returns[i]),
                    'volatility': safe_num(np.sqrt(cov_matrix[i, i]))
                })
            
            return {
                'optimized_allocation': allocation,
                'expected_return': safe_num(portfolio_return),
                'expected_volatility': safe_num(portfolio_volatility),
                'risk_tolerance': risk_tolerance,
                'recommendations': self.get_portfolio_recommendations(allocation, user.get('Fund_Name', 'Unknown'))
            }
        except Exception as e:
            return {'error': str(e)}