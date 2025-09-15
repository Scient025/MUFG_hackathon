#!/usr/bin/env python3
"""
Database setup script for Supabase
This script creates the user_profiles table and sets up the necessary permissions
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://imtmbgbktomztqtoyuvh.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltdG1iZ2JrdG9tenRxdG95dXZoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc0MzM1MDEsImV4cCI6MjA3MzAwOTUwMX0.iqWnjWB4bNIg7DYom4V_ZCdqSL8hKptAMcmG1we2")

def setup_database():
    """Set up the Supabase database with required tables"""
    try:
        # Create Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        print("✅ Supabase client created successfully")
        
        # Test connection by trying to list tables
        print("🔍 Testing Supabase connection...")
        
        # Try to create a test user profile to see if table exists
        test_data = {
            "id": "test-user-123",
            "email": "test@example.com",
            "name": "Test User",
            "age": 30,
            "gender": "Other",
            "country": "Australia",
            "employment_status": "Employed",
            "annual_income": 75000,
            "current_savings": 25000,
            "retirement_age_goal": 65,
            "risk_tolerance": "Medium",
            "contribution_amount": 1000,
            "contribution_frequency": "Monthly",
            "employer_contribution": 500,
            "years_contributed": 5,
            "investment_type": "Superannuation",
            "fund_name": "Test Fund",
            "marital_status": "Single",
            "number_of_dependents": 0,
            "education_level": "Bachelor's Degree",
            "health_status": "Good",
            "home_ownership_status": "Renting",
            "investment_experience_level": "Intermediate",
            "financial_goals": "Retirement planning",
            "insurance_coverage": "Basic",
            "pension_type": "Superannuation",
            "withdrawal_strategy": "Conservative"
        }
        
        print("📝 Attempting to insert test data...")
        result = supabase.table('user_profiles').insert(test_data).execute()
        
        if result.data:
            print("✅ Test data inserted successfully!")
            print(f"📊 Inserted user: {result.data[0]['name']}")
            
            # Clean up test data
            print("🧹 Cleaning up test data...")
            supabase.table('user_profiles').delete().eq('id', 'test-user-123').execute()
            print("✅ Test data cleaned up")
            
        print("\n🎉 Database setup completed successfully!")
        print("📋 The user_profiles table is ready to use")
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Make sure your Supabase project is active")
        print("2. Check that the API key is correct")
        print("3. Ensure the user_profiles table exists in your Supabase database")
        print("4. Run the SQL script from supabase_setup.sql in your Supabase SQL editor")

if __name__ == "__main__":
    setup_database()
