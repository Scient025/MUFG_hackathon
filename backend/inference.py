import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Optional
import os
from sklearn.preprocessing import LabelEncoder
from supabase_config import supabase, USER_PROFILES_TABLE

class SuperannuationInference:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.df = None
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        
        # Load models and data
        self.load_models()
        self.load_data()
    
    def load_models(self):
        """Load trained ML models"""
        try:
            self.models['kmeans'] = joblib.load(f'{self.models_dir}/kmeans_model.pkl')
            self.models['risk_prediction'] = joblib.load(f'{self.models_dir}/risk_prediction_model.pkl')
            self.models['investment_recommendation'] = joblib.load(f'{self.models_dir}/investment_recommendation_model.pkl')
            self.scaler = joblib.load(f'{self.models_dir}/scaler.pkl')  # For clustering
            self.risk_scaler = joblib.load(f'{self.models_dir}/risk_scaler.pkl')  # For risk prediction
            self.investment_scaler = joblib.load(f'{self.models_dir}/investment_scaler.pkl')  # For investment model
            self.label_encoders = joblib.load(f'{self.models_dir}/label_encoders.pkl')
            print("All models loaded successfully!")
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def load_data(self):
        """Load data from Supabase"""
        try:
            # Fetch all user data from Supabase
            response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
            
            if not response.data:
                print("No data found in Supabase. Using empty DataFrame.")
                self.df = pd.DataFrame()
                return
            
            # Convert to DataFrame
            self.df = pd.DataFrame(response.data)
            
            # Create encoded columns for categorical features (same as training)
            categorical_features = ['Risk_Tolerance', 'Investment_Type', 'Investment_Experience_Level']
            for col in categorical_features:
                if col in self.df.columns:
                    # Create encoded version
                    le = LabelEncoder()
                    self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                    # Store encoder if not already stored
                    if col not in self.label_encoders:
                        self.label_encoders[col] = le
            
            print(f"Data loaded from Supabase: {len(self.df)} users")
            
        except Exception as e:
            print(f"Error loading data from Supabase: {e}")
            # Fallback to empty DataFrame
            self.df = pd.DataFrame()
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile by User_ID from Supabase"""
        try:
            # First try to get from loaded DataFrame
            if not self.df.empty:
                user_data = self.df[self.df['User_ID'] == user_id]
                if not user_data.empty:
                    user = user_data.iloc[0].to_dict()
                    # Add a generated name if not present
                    if 'Name' not in user or not user['Name']:
                        user['Name'] = self.generate_name(user_id)
                    return user
            
            # If not found in DataFrame, fetch directly from Supabase
            response = supabase.table(USER_PROFILES_TABLE).select("*").eq("User_ID", user_id).execute()
            
            if not response.data:
                raise ValueError(f"User {user_id} not found in Supabase")
            
            user = response.data[0]
            
            # Add a generated name if not present
            if 'Name' not in user or not user['Name']:
                user['Name'] = self.generate_name(user_id)
            
            return user
            
        except Exception as e:
            print(f"Error fetching user profile for {user_id}: {e}")
            raise ValueError(f"User {user_id} not found")
    
    def generate_name(self, user_id: str) -> str:
        """Generate a realistic name based on User_ID"""
        # Extract number from User_ID (e.g., U1000 -> 1000)
        user_num = int(user_id[1:]) if user_id.startswith('U') else 0
        
        # Simple name generation based on user number
        first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
            "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
            "Thomas", "Sarah", "Christopher", "Karen", "Charles", "Nancy", "Daniel", "Lisa",
            "Matthew", "Betty", "Anthony", "Helen", "Mark", "Sandra", "Donald", "Donna",
            "Steven", "Carol", "Paul", "Ruth", "Andrew", "Sharon", "Joshua", "Michelle",
            "Kenneth", "Laura", "Kevin", "Sarah", "Brian", "Kimberly", "George", "Deborah",
            "Timothy", "Dorothy", "Ronald", "Lisa", "Jason", "Nancy", "Edward", "Karen",
            "Jeffrey", "Betty", "Ryan", "Helen", "Jacob", "Sandra", "Gary", "Donna",
            "Nicholas", "Carol", "Eric", "Ruth", "Jonathan", "Sharon", "Stephen", "Michelle",
            "Larry", "Laura", "Justin", "Sarah", "Scott", "Kimberly", "Brandon", "Deborah",
            "Benjamin", "Dorothy", "Samuel", "Amy", "Gregory", "Angela", "Alexander", "Brenda",
            "Patrick", "Emma", "Jack", "Olivia", "Dennis", "Cynthia", "Jerry", "Marie",
            "Tyler", "Janet", "Aaron", "Catherine", "Jose", "Frances", "Henry", "Christine",
            "Adam", "Samantha", "Douglas", "Debra", "Nathan", "Rachel", "Peter", "Carolyn",
            "Zachary", "Janet", "Kyle", "Virginia", "Noah", "Maria", "Alan", "Heather",
            "Ethan", "Diane", "Jeremy", "Julie", "Mason", "Joyce", "Christian", "Victoria",
            "Keith", "Kelly", "Roger", "Christina", "Gerald", "Joan", "Carl", "Evelyn",
            "Harold", "Judith", "Sean", "Megan", "Austin", "Cheryl", "Arthur", "Andrea",
            "Lawrence", "Hannah", "Joe", "Jacqueline", "Noah", "Martha", "Wayne", "Gloria",
            "Roy", "Teresa", "Eugene", "Sara", "Louis", "Janice", "Philip", "Julia",
            "Bobby", "Marie", "Johnny", "Madison", "Willie", "Grace", "Ralph", "Judy"
        ]
        
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
            "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas",
            "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White",
            "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker", "Young",
            "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
            "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
            "Carter", "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker",
            "Cruz", "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy",
            "Cook", "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey",
            "Reed", "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson",
            "Watson", "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza",
            "Ruiz", "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers",
            "Long", "Ross", "Foster", "Jimenez", "Powell", "Jenkins", "Perry", "Russell",
            "Sullivan", "Bell", "Coleman", "Butler", "Henderson", "Barnes", "Gonzales", "Fisher",
            "Vasquez", "Simmons", "Romero", "Jordan", "Patterson", "Alexander", "Hamilton", "Graham",
            "Reynolds", "Griffin", "Wallace", "Moreno", "West", "Cole", "Hayes", "Bryant",
            "Herrera", "Gibson", "Ellis", "Tran", "Medina", "Aguilar", "Stevens", "Murray",
            "Ford", "Castro", "Marshall", "Owens", "Harrison", "Fernandez", "McDonald", "Woods",
            "Washington", "Kennedy", "Wells", "Vargas", "Henry", "Chen", "Freeman", "Webb",
            "Tucker", "Guzman", "Burns", "Crawford", "Olson", "Simpson", "Porter", "Hunter",
            "Gordon", "Mendez", "Silva", "Shaw", "Snyder", "Mason", "Dixon", "Munoz",
            "Hunt", "Hicks", "Holmes", "Palmer", "Wagner", "Robertson", "Black", "Holmes",
            "Stone", "Meyer", "Boyd", "Mills", "Warren", "Fox", "Rose", "Rice",
            "Moreno", "Schmidt", "Patel", "Ferguson", "Nichols", "Herrera", "Medina", "Ryan",
            "Fernandez", "Weaver", "Daniels", "Stephens", "Gardner", "Payne", "Kelley", "Dunn",
            "Pierce", "Arnold", "Tran", "Spencer", "Peters", "Hawkins", "Grant", "Hansen",
            "Castro", "Hoffman", "Hart", "Elliott", "Cunningham", "Knight", "Bradley", "Carroll",
            "Hudson", "Duncan", "Armstrong", "Berry", "Andrews", "Johnston", "Ray", "Lane",
            "Riley", "Carpenter", "Perkins", "Aguilar", "Silva", "Richards", "Willis", "Matthews",
            "Chapman", "Lawrence", "Garza", "Vargas", "Watkins", "Wheeler", "Burton", "Harper",
            "Lynch", "Fuller", "Owens", "Mcdonald", "Cruz", "Marshall", "Owen", "Gomez",
            "Keith", "Lawrence", "May", "Ramos", "Holland", "Washington", "Tucker", "Barrett",
            "Casey", "Boone", "Cortez", "Bryan", "Adkins", "Santana", "Mann", "Gilbert",
            "Buchanan", "Ortega", "Warner", "Mack", "Briggs", "Walton", "Brady", "Oliver",
            "Mccarthy", "Mckinney", "Love", "Banks", "Santos", "Neal", "Cannon", "Todd",
            "Yates", "Sparks", "Duran", "Conner", "Huang", "Zimmerman", "Wall", "Booker",
            "Welch", "Hansen", "Schroeder", "Franklin", "Lawson", "Fields", "Howell", "Leonard",
            "Douglas", "Lane", "Little", "Griffin", "Haynes", "Harvey", "Mills", "Jacobs",
            "Medina", "Day", "Boyd", "Johnston", "West", "Carr", "Duncan", "Armstrong",
            "Berry", "Andrews", "Ray", "Riley", "Carpenter", "Perkins", "Aguilar", "Silva"
        ]
        
        # Generate name based on user number
        first_name = first_names[user_num % len(first_names)]
        last_name = last_names[(user_num // len(first_names)) % len(last_names)]
        
        return f"{first_name} {last_name}"
    
    def predict_risk_tolerance(self, user_id: str) -> Dict[str, Any]:
        """Predict risk tolerance for a user"""
        user = self.get_user_profile(user_id)
        
        # Prepare features (use encoded versions for categorical features)
        risk_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Years_Contributed', 'Investment_Experience_Level_encoded', 'Portfolio_Diversity_Score',
            'Savings_Rate', 'Debt_Level'
        ]
        
        # Get feature values with proper handling of missing encoded features
        feature_values = []
        for feature in risk_features:
            if feature in user:
                value = user[feature]
                # Handle NaN values
                if pd.isna(value) or value is None:
                    feature_values.append(0)
                else:
                    feature_values.append(value)
            else:
                # Handle missing encoded features by creating them on-the-fly
                if feature == 'Investment_Experience_Level_encoded':
                    exp_level = user.get('Investment_Experience_Level', 'Beginner')
                    if exp_level in self.label_encoders.get('Investment_Experience_Level', {}).classes_:
                        encoded_value = self.label_encoders['Investment_Experience_Level'].transform([exp_level])[0]
                    else:
                        encoded_value = 0  # Default to Beginner
                    feature_values.append(encoded_value)
                elif feature == 'Debt_Level':
                    debt_level = user.get('Debt_Level', 'Low')
                    if debt_level in self.label_encoders.get('Debt_Level', {}).classes_:
                        encoded_value = self.label_encoders['Debt_Level'].transform([debt_level])[0]
                    else:
                        encoded_value = 0  # Default to Low
                    feature_values.append(encoded_value)
                else:
                    feature_values.append(0)  # Default value for other features
        
        # Scale features using the risk prediction scaler
        feature_array = np.array(feature_values).reshape(1, -1)
        scaled_features = self.risk_scaler.transform(feature_array)
        
        # Predict
        risk_prediction = self.models['risk_prediction'].predict(scaled_features)[0]
        risk_probabilities = self.models['risk_prediction'].predict_proba(scaled_features)[0]
        
        # Map back to risk levels
        risk_levels = ['Low', 'Medium', 'High']
        predicted_risk = risk_levels[risk_prediction]
        
        return {
            'user_id': user_id,
            'predicted_risk': predicted_risk,
            'confidence': float(max(risk_probabilities)),
            'current_risk': user.get('Risk_Tolerance', 'Unknown'),
            'risk_probabilities': {
                'Low': float(risk_probabilities[0]),
                'Medium': float(risk_probabilities[1]),
                'High': float(risk_probabilities[2])
            }
        }
    
    def get_user_segment(self, user_id: str) -> Dict[str, Any]:
        """Get user's cluster/segment"""
        user = self.get_user_profile(user_id)
        
        # Prepare clustering features
        clustering_features = [
            'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
            'Risk_Tolerance', 'Years_Contributed', 'Portfolio_Diversity_Score'
        ]
        
        # Get feature values with proper handling of missing values
        feature_values = []
        for feature in clustering_features:
            if feature in user:
                if feature == 'Risk_Tolerance':
                    # Encode risk tolerance
                    risk_levels = ['Low', 'Medium', 'High']
                    risk_value = user[feature]
                    if pd.isna(risk_value) or risk_value is None:
                        feature_values.append(1)  # Default to Medium
                    else:
                        feature_values.append(risk_levels.index(risk_value) if risk_value in risk_levels else 1)
                else:
                    value = user[feature]
                    # Handle NaN values
                    if pd.isna(value) or value is None:
                        feature_values.append(0)
                    else:
                        feature_values.append(value)
            else:
                feature_values.append(0)
        
        # Scale features
        feature_array = np.array(feature_values).reshape(1, -1)
        scaled_features = self.scaler.transform(feature_array)
        
        # Predict cluster
        cluster = self.models['kmeans'].predict(scaled_features)[0]
        
        # Get peer statistics
        peer_stats = self.get_peer_statistics(user_id, cluster)
        
        return {
            'user_id': user_id,
            'cluster': int(cluster),
            'peer_stats': peer_stats
        }
    
    def get_peer_statistics(self, user_id: str, cluster: int) -> Dict[str, Any]:
        """Get peer group statistics"""
        user = self.get_user_profile(user_id)
        
        try:
            # Get users in same cluster from Supabase
            if 'Cluster' in self.df.columns and not self.df.empty:
                cluster_users = self.df[self.df['Cluster'] == cluster]
            else:
                # If no cluster data, get all users for comparison
                cluster_users = self.df if not self.df.empty else pd.DataFrame()
            
            # If DataFrame is empty, fetch from Supabase
            if cluster_users.empty:
                response = supabase.table(USER_PROFILES_TABLE).select("*").execute()
                if response.data:
                    cluster_users = pd.DataFrame(response.data)
                else:
                    cluster_users = pd.DataFrame()
            
            # Calculate peer statistics
            if not cluster_users.empty:
                peer_stats = {
                    'total_peers': len(cluster_users),
                    'avg_age': float(cluster_users['Age'].mean()) if 'Age' in cluster_users.columns else 0,
                    'avg_income': float(cluster_users['Annual_Income'].mean()) if 'Annual_Income' in cluster_users.columns else 0,
                    'avg_savings': float(cluster_users['Current_Savings'].mean()) if 'Current_Savings' in cluster_users.columns else 0,
                    'avg_contribution': float(cluster_users['Contribution_Amount'].mean()) if 'Contribution_Amount' in cluster_users.columns else 0,
                    'common_investment_types': cluster_users['Investment_Type'].value_counts().head(3).to_dict() if 'Investment_Type' in cluster_users.columns else {},
                    'common_fund_names': cluster_users['Fund_Name'].value_counts().head(3).to_dict() if 'Fund_Name' in cluster_users.columns else {},
                    'risk_distribution': cluster_users['Risk_Tolerance'].value_counts().to_dict() if 'Risk_Tolerance' in cluster_users.columns else {},
                    'contribution_percentile': float((cluster_users['Contribution_Amount'] <= user['Contribution_Amount']).mean() * 100) if 'Contribution_Amount' in cluster_users.columns and 'Contribution_Amount' in user else 50
                }
            else:
                # Default stats if no data available
                peer_stats = {
                    'total_peers': 0,
                    'avg_age': 0,
                    'avg_income': 0,
                    'avg_savings': 0,
                    'avg_contribution': 0,
                    'common_investment_types': {},
                    'common_fund_names': {},
                    'risk_distribution': {},
                    'contribution_percentile': 50
                }
            
            return peer_stats
            
        except Exception as e:
            print(f"Error calculating peer statistics: {e}")
            # Return default stats on error
            return {
                'total_peers': 0,
                'avg_age': 0,
                'avg_income': 0,
                'avg_savings': 0,
                'avg_contribution': 0,
                'common_investment_types': {},
                'common_fund_names': {},
                'risk_distribution': {},
                'contribution_percentile': 50
            }
    
    def predict_pension_projection(self, user_id: str, extra_monthly: float = 0) -> Dict[str, Any]:
        """Predict pension projection for a user"""
        try:
            print(f"Getting user profile for {user_id}")
            user = self.get_user_profile(user_id)
            print(f"User profile loaded: {user.get('User_ID', 'Unknown')}")
            
            # Adjust contribution if extra monthly amount provided
            adjusted_contribution = user['Contribution_Amount'] + (extra_monthly * 12)
            print(f"Adjusted contribution: {adjusted_contribution}")
            
            # Prepare features (use encoded feature names as trained)
            investment_features = [
                'Age', 'Annual_Income', 'Current_Savings', 'Contribution_Amount',
                'Risk_Tolerance_encoded', 'Investment_Type_encoded', 'Annual_Return_Rate',
                'Volatility', 'Fees_Percentage', 'Years_Contributed', 'Savings_Rate',
                'Portfolio_Diversity_Score', 'Investment_Experience_Level_encoded'
            ]
            
            # Get feature values with proper handling of missing encoded features
            feature_values = []
            for feature in investment_features:
                if feature == 'Contribution_Amount':
                    feature_values.append(adjusted_contribution)
                elif feature in user:
                    value = user[feature]
                    # Handle NaN values
                    if pd.isna(value) or value is None:
                        feature_values.append(0)
                    else:
                        feature_values.append(value)
                else:
                    # Handle missing encoded features by creating them on-the-fly
                    if feature == 'Risk_Tolerance_encoded':
                        risk_tolerance = user.get('Risk_Tolerance', 'Medium')
                        if risk_tolerance in self.label_encoders.get('Risk_Tolerance', {}).classes_:
                            encoded_value = self.label_encoders['Risk_Tolerance'].transform([risk_tolerance])[0]
                        else:
                            encoded_value = 1  # Default to Medium
                        feature_values.append(encoded_value)
                    elif feature == 'Investment_Type_encoded':
                        investment_type = user.get('Investment_Type', 'ETF')
                        if investment_type in self.label_encoders.get('Investment_Type', {}).classes_:
                            encoded_value = self.label_encoders['Investment_Type'].transform([investment_type])[0]
                        else:
                            encoded_value = 0  # Default to ETF
                        feature_values.append(encoded_value)
                    elif feature == 'Investment_Experience_Level_encoded':
                        exp_level = user.get('Investment_Experience_Level', 'Beginner')
                        if exp_level in self.label_encoders.get('Investment_Experience_Level', {}).classes_:
                            encoded_value = self.label_encoders['Investment_Experience_Level'].transform([exp_level])[0]
                        else:
                            encoded_value = 0  # Default to Beginner
                        feature_values.append(encoded_value)
                    else:
                        feature_values.append(0)  # Default value for other features
            
            print(f"Feature values: {feature_values}")
            
            # Scale features using investment scaler
            feature_array = np.array(feature_values).reshape(1, -1)
            scaled_features = self.investment_scaler.transform(feature_array)
            print(f"Scaled features: {scaled_features}")
            
            # Predict pension amount
            projected_pension = self.models['investment_recommendation'].predict(scaled_features)[0]
            print(f"Projected pension: {projected_pension}")
            
            # Calculate years to retirement
            retirement_age = user.get('Retirement_Age_Goal', 65)
            years_to_retirement = max(1, retirement_age - user['Age'])
            
            # Calculate monthly income at retirement
            monthly_income = projected_pension / 12
            
            result = {
                'user_id': user_id,
                'current_projection': float(user.get('Projected_Pension_Amount', 0)) if user.get('Projected_Pension_Amount', 0) is not None else 0.0,
                'adjusted_projection': float(projected_pension),
                'extra_monthly_contribution': extra_monthly,
                'years_to_retirement': years_to_retirement,
                'monthly_income_at_retirement': float(monthly_income),
                'improvement': float(projected_pension - (user.get('Projected_Pension_Amount', 0) if user.get('Projected_Pension_Amount', 0) is not None else 0))
            }
            print(f"Result: {result}")
            return result
            
        except Exception as e:
            print(f"Error in predict_pension_projection: {e}")
            raise
    
    def get_summary_stats(self, user_id: str) -> Dict[str, Any]:
        """Get summary statistics for dashboard"""
        user = self.get_user_profile(user_id)
        
        # Calculate percentage to goal
        goal_amount = user.get('Projected_Pension_Amount', 0)
        current_savings = user.get('Current_Savings', 0)
        percent_to_goal = (current_savings / goal_amount * 100) if goal_amount > 0 else 0
        
        return {
            'user_id': user_id,
            'current_savings': float(current_savings) if current_savings is not None else 0.0,
            'projected_pension': float(goal_amount) if goal_amount is not None else 0.0,
            'percent_to_goal': float(percent_to_goal),
            'monthly_income_at_retirement': float(user.get('Expected_Annual_Payout', 0) / 12) if user.get('Expected_Annual_Payout', 0) is not None else 0.0,
            'employer_contribution': float(user.get('Employer_Contribution', 0)) if user.get('Employer_Contribution', 0) is not None else 0.0,
            'total_annual_contribution': float(user.get('Total_Annual_Contribution', 0)) if user.get('Total_Annual_Contribution', 0) is not None else 0.0,
            'risk_tolerance': user.get('Risk_Tolerance', 'Unknown'),
            'investment_type': user.get('Investment_Type', 'Unknown'),
            'fund_name': user.get('Fund_Name', 'Unknown')
        }

if __name__ == "__main__":
    # Test inference
    inference = SuperannuationInference()
    
    # Test with first user from Supabase
    if not inference.df.empty:
        test_user = inference.df['User_ID'].iloc[0]
        print(f"Testing with user: {test_user}")
        
        # Test predictions
        risk_pred = inference.predict_risk_tolerance(test_user)
        print(f"Risk prediction: {risk_pred}")
        
        segment = inference.get_user_segment(test_user)
        print(f"User segment: {segment}")
        
        projection = inference.predict_pension_projection(test_user, 200)
        print(f"Pension projection: {projection}")
        
        summary = inference.get_summary_stats(test_user)
        print(f"Summary stats: {summary}")
    else:
        print("No users found in Supabase database")
