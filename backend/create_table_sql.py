#!/usr/bin/env python3
"""
Create user_profiles table using SQL
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables from project root
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
try:
    load_dotenv(env_path)
except Exception as e:
    print(f"Warning: Could not load .env file at {env_path}: {e}")

# Supabase configuration with service role key
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://imtmbgbktomztqtoyuvh.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltdG1iZ2JrdG9tenRxdG95dXZoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzQzMzUwMSwiZXhwIjoyMDczMDA5NTAxfQ.skNHNAimBcivIo18Lm9XzEB6oi7Fz7WHP3EMmVbpRQc")

def create_table():
    """Create the user_profiles table using SQL"""
    try:
        # Create Supabase client with service role key
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("✅ Supabase client created with service role key")
        
        # SQL to create the table
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS user_profiles (
          id TEXT PRIMARY KEY,
          email TEXT NOT NULL,
          name TEXT NOT NULL,
          age INTEGER NOT NULL,
          gender TEXT NOT NULL,
          country TEXT NOT NULL,
          employment_status TEXT NOT NULL,
          annual_income DECIMAL(15,2) NOT NULL,
          current_savings DECIMAL(15,2) NOT NULL,
          retirement_age_goal INTEGER NOT NULL,
          risk_tolerance TEXT NOT NULL,
          contribution_amount DECIMAL(15,2) NOT NULL,
          contribution_frequency TEXT NOT NULL DEFAULT 'Monthly',
          employer_contribution DECIMAL(15,2) NOT NULL,
          years_contributed INTEGER NOT NULL DEFAULT 0,
          investment_type TEXT NOT NULL DEFAULT 'Balanced',
          fund_name TEXT NOT NULL DEFAULT 'Default Fund',
          marital_status TEXT NOT NULL,
          number_of_dependents INTEGER NOT NULL DEFAULT 0,
          education_level TEXT NOT NULL DEFAULT 'Bachelor',
          health_status TEXT NOT NULL DEFAULT 'Good',
          home_ownership_status TEXT NOT NULL DEFAULT 'Renting',
          investment_experience_level TEXT NOT NULL,
          financial_goals TEXT NOT NULL,
          insurance_coverage TEXT NOT NULL DEFAULT 'Basic',
          pension_type TEXT NOT NULL DEFAULT 'Superannuation',
          withdrawal_strategy TEXT NOT NULL DEFAULT 'Fixed',
          created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
          updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
        """
        
        print("📝 Creating user_profiles table...")
        
        # Execute the SQL
        result = supabase.rpc('exec_sql', {'sql': create_table_sql}).execute()
        
        print("✅ Table creation SQL executed")
        
        # Now try to insert sample data
        sample_users = [
            {
                "id": "user-001",
                "email": "john.smith@example.com",
                "name": "John Smith",
                "age": 35,
                "gender": "Male",
                "country": "Australia",
                "employment_status": "Employed",
                "annual_income": 85000,
                "current_savings": 45000,
                "retirement_age_goal": 65,
                "risk_tolerance": "Medium",
                "contribution_amount": 1200,
                "employer_contribution": 600,
                "marital_status": "Married",
                "number_of_dependents": 2,
                "investment_experience_level": "Intermediate",
                "financial_goals": "Retirement planning and children's education"
            },
            {
                "id": "user-002",
                "email": "sarah.johnson@example.com",
                "name": "Sarah Johnson",
                "age": 28,
                "gender": "Female",
                "country": "Australia",
                "employment_status": "Employed",
                "annual_income": 65000,
                "current_savings": 18000,
                "retirement_age_goal": 60,
                "risk_tolerance": "High",
                "contribution_amount": 800,
                "employer_contribution": 400,
                "marital_status": "Single",
                "number_of_dependents": 0,
                "investment_experience_level": "Beginner",
                "financial_goals": "Early retirement and travel"
            }
        ]
        
        print("📝 Inserting sample users...")
        
        for user in sample_users:
            try:
                result = supabase.table('user_profiles').insert(user).execute()
                if result.data:
                    print(f"✅ Inserted: {user['name']}")
                else:
                    print(f"⚠️  Failed to insert: {user['name']}")
            except Exception as e:
                print(f"❌ Error inserting {user['name']}: {e}")
        
        # Test reading the data
        print("\n🔍 Testing data retrieval...")
        result = supabase.table('user_profiles').select('*').execute()
        
        if result.data:
            print(f"✅ Successfully retrieved {len(result.data)} users from database")
            print("\n📋 Users in database:")
            for user in result.data:
                print(f"   • {user['name']} ({user['email']}) - ${user['annual_income']:,}")
        else:
            print("❌ No data retrieved from database")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Alternative approach:")
        print("1. Go to Supabase dashboard → SQL Editor")
        print("2. Copy and paste the SQL from 'simple_supabase_setup.sql'")
        print("3. Click 'Run' to execute")
        return False

if __name__ == "__main__":
    print("🚀 Creating Supabase Table")
    print("=" * 50)
    
    if create_table():
        print("\n✅ Table created and data inserted successfully!")
    else:
        print("\n❌ Table creation failed")
        print("📋 Please create the table manually in Supabase dashboard")
