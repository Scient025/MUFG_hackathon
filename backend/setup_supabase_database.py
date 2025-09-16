#!/usr/bin/env python3
"""
Set up Supabase database with service role key
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

def setup_database():
    """Set up the Supabase database with table and sample data"""
    try:
        # Create Supabase client with service role key
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("✅ Supabase client created with service role key")
        
        # Test connection
        print("🔍 Testing Supabase connection...")
        
        # Try to create the table using SQL
        print("📝 Creating user_profiles table...")
        
        # First, let's try to insert sample data directly
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
                "contribution_frequency": "Monthly",
                "employer_contribution": 600,
                "years_contributed": 8,
                "investment_type": "Superannuation",
                "fund_name": "AustralianSuper",
                "marital_status": "Married",
                "number_of_dependents": 2,
                "education_level": "Bachelor's Degree",
                "health_status": "Good",
                "home_ownership_status": "Own",
                "investment_experience_level": "Intermediate",
                "financial_goals": "Retirement planning and children's education",
                "insurance_coverage": "Comprehensive",
                "pension_type": "Superannuation",
                "withdrawal_strategy": "Balanced"
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
                "contribution_frequency": "Monthly",
                "employer_contribution": 400,
                "years_contributed": 3,
                "investment_type": "Superannuation",
                "fund_name": "Rest Super",
                "marital_status": "Single",
                "number_of_dependents": 0,
                "education_level": "Master's Degree",
                "health_status": "Excellent",
                "home_ownership_status": "Renting",
                "investment_experience_level": "Beginner",
                "financial_goals": "Early retirement and travel",
                "insurance_coverage": "Basic",
                "pension_type": "Superannuation",
                "withdrawal_strategy": "Aggressive"
            },
            {
                "id": "user-003",
                "email": "michael.brown@example.com",
                "name": "Michael Brown",
                "age": 52,
                "gender": "Male",
                "country": "Australia",
                "employment_status": "Self-employed",
                "annual_income": 120000,
                "current_savings": 180000,
                "retirement_age_goal": 65,
                "risk_tolerance": "Low",
                "contribution_amount": 2000,
                "contribution_frequency": "Monthly",
                "employer_contribution": 0,
                "years_contributed": 25,
                "investment_type": "Superannuation",
                "fund_name": "UniSuper",
                "marital_status": "Married",
                "number_of_dependents": 1,
                "education_level": "PhD",
                "health_status": "Good",
                "home_ownership_status": "Own",
                "investment_experience_level": "Expert",
                "financial_goals": "Comfortable retirement",
                "insurance_coverage": "Comprehensive",
                "pension_type": "Superannuation",
                "withdrawal_strategy": "Conservative"
            },
            {
                "id": "user-004",
                "email": "emma.wilson@example.com",
                "name": "Emma Wilson",
                "age": 41,
                "gender": "Female",
                "country": "Australia",
                "employment_status": "Employed",
                "annual_income": 95000,
                "current_savings": 75000,
                "retirement_age_goal": 62,
                "risk_tolerance": "Medium",
                "contribution_amount": 1500,
                "contribution_frequency": "Monthly",
                "employer_contribution": 750,
                "years_contributed": 15,
                "investment_type": "Superannuation",
                "fund_name": "Hostplus",
                "marital_status": "Divorced",
                "number_of_dependents": 1,
                "education_level": "Bachelor's Degree",
                "health_status": "Good",
                "home_ownership_status": "Own",
                "investment_experience_level": "Intermediate",
                "financial_goals": "Financial independence",
                "insurance_coverage": "Standard",
                "pension_type": "Superannuation",
                "withdrawal_strategy": "Balanced"
            },
            {
                "id": "user-005",
                "email": "david.lee@example.com",
                "name": "David Lee",
                "age": 29,
                "gender": "Male",
                "country": "Australia",
                "employment_status": "Employed",
                "annual_income": 72000,
                "current_savings": 25000,
                "retirement_age_goal": 65,
                "risk_tolerance": "High",
                "contribution_amount": 900,
                "contribution_frequency": "Monthly",
                "employer_contribution": 450,
                "years_contributed": 4,
                "investment_type": "Superannuation",
                "fund_name": "Cbus",
                "marital_status": "Single",
                "number_of_dependents": 0,
                "education_level": "Bachelor's Degree",
                "health_status": "Excellent",
                "home_ownership_status": "Renting",
                "investment_experience_level": "Beginner",
                "financial_goals": "Wealth building and property investment",
                "insurance_coverage": "Basic",
                "pension_type": "Superannuation",
                "withdrawal_strategy": "Aggressive"
            }
        ]
        
        # Try to insert users one by one
        print("📝 Inserting sample users...")
        inserted_count = 0
        
        for user in sample_users:
            try:
                result = supabase.table('user_profiles').insert(user).execute()
                if result.data:
                    inserted_count += 1
                    print(f"✅ Inserted: {user['name']}")
                else:
                    print(f"⚠️  Failed to insert: {user['name']}")
            except Exception as e:
                print(f"❌ Error inserting {user['name']}: {e}")
        
        print(f"\n📊 Successfully inserted {inserted_count} users")
        
        # Test reading the data
        print("\n🔍 Testing data retrieval...")
        result = supabase.table('user_profiles').select('*').execute()
        
        if result.data:
            print(f"✅ Successfully retrieved {len(result.data)} users from database")
            print("\n📋 Sample users in database:")
            for user in result.data[:3]:  # Show first 3 users
                print(f"   • {user['name']} ({user['email']}) - ${user['annual_income']:,}")
        else:
            print("❌ No data retrieved from database")
        
        print("\n🎉 Database setup completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        print("\n🔧 This might mean:")
        print("1. The user_profiles table doesn't exist yet")
        print("2. The table structure is different")
        print("3. There are permission issues")
        
        print("\n📋 Manual setup required:")
        print("1. Go to Supabase dashboard → SQL Editor")
        print("2. Run the SQL from 'simple_supabase_setup.sql'")
        print("3. Then run this script again")
        
        return False

if __name__ == "__main__":
    print("🚀 Setting up Supabase Database")
    print("=" * 50)
    
    if setup_database():
        print("\n✅ Database is ready for use!")
        print("🔄 You can now remove the mock data fallback from the code")
    else:
        print("\n❌ Database setup failed")
        print("📋 Please set up the table manually in Supabase dashboard")
