#!/usr/bin/env python3
"""
Test Supabase connection and set up database
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
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltdG1iZ2JrdG9tenRxdG95dXZoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzQzMzUwMSwiZXhwIjoyMDczMDA5NTAxfQ.skNHNAimBcivIo18Lm9XzEB6oi7Fz7WHP3EMmVbpRQc")

def test_supabase_connection():
    """Test Supabase connection and database setup"""
    try:
        # Create Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        print("✅ Supabase client created successfully")
        
        # Test basic connection
        print("🔍 Testing Supabase connection...")
        
        # Try to read from MUFG table
        try:
            result = supabase.table('MUFG').select('*').limit(1).execute()
            print("✅ Successfully connected to MUFG table")
            print(f"📊 Found {len(result.data)} records")
            
            if result.data:
                print("📋 Sample record:")
                sample = result.data[0]
                print(f"   User_ID: {sample.get('User_ID', 'N/A')}")
                print(f"   Name: {sample.get('Name', 'N/A')}")
                print(f"   Age: {sample.get('Age', 'N/A')}")
                print(f"   Gender: {sample.get('Gender', 'N/A')}")
                print(f"   Annual_Income: ${sample.get('Annual_Income', 'N/A')}")
                print(f"   Risk_Tolerance: {sample.get('Risk_Tolerance', 'N/A')}")
            
            return True
            
        except Exception as table_error:
            print(f"❌ Error accessing MUFG table: {table_error}")
            print("\n🔧 The table might not exist or have different permissions. You need to:")
            print("1. Go to your Supabase dashboard")
            print("2. Check if the MUFG table exists in Table Editor")
            print("3. Verify the table permissions")
            print("4. Check if the service role key has access")
            
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to Supabase: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check your Supabase URL and API key")
        print("2. Make sure your Supabase project is active")
        print("3. Verify the API key has the correct permissions")
        return False

def setup_sample_data():
    """Set up sample data in Supabase"""
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        
        sample_data = {
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
            "employer_contribution": 500,
            "marital_status": "Single",
            "number_of_dependents": 0,
            "investment_experience_level": "Beginner",
            "financial_goals": "Retirement planning"
        }
        
        print("📝 Inserting test data...")
        result = supabase.table('user_profiles').insert(sample_data).execute()
        
        if result.data:
            print("✅ Test data inserted successfully!")
            print(f"📊 Inserted user: {result.data[0]['name']}")
            
            # Clean up test data
            print("🧹 Cleaning up test data...")
            supabase.table('user_profiles').delete().eq('id', 'test-user-123').execute()
            print("✅ Test data cleaned up")
            return True
        else:
            print("❌ Failed to insert test data")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up sample data: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Supabase Connection")
    print("=" * 50)
    
    # Test connection
    if test_supabase_connection():
        print("\n🎉 Supabase connection successful!")
        
        # Try to set up sample data
        print("\n📝 Testing data operations...")
        if setup_sample_data():
            print("\n✅ All tests passed! Supabase is ready to use.")
        else:
            print("\n⚠️  Connection works but data operations failed.")
    else:
        print("\n❌ Supabase connection failed.")
        print("\n📋 Next steps:")
        print("1. Check your Supabase project settings")
        print("2. Verify your API key")
        print("3. Run the SQL setup script in Supabase dashboard")
