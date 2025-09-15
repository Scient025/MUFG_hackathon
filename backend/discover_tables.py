#!/usr/bin/env python3
"""
Discover what tables exist in Supabase database
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

# Supabase configuration with service role key
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://imtmbgbktomztqtoyuvh.supabase.co")
SUPABASE_SERVICE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltdG1iZ2JrdG9tenRxdG95dXZoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzQzMzUwMSwiZXhwIjoyMDczMDA5NTAxfQ.skNHNAimBcivIo18Lm9XzEB6oi7Fz7WHP3EMmVbpRQc"

def discover_tables():
    """Discover what tables exist in the database"""
    try:
        # Create Supabase client with service role key
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("✅ Supabase client created with service role key")
        
        # Try different common table names
        possible_table_names = [
            'user_profiles',
            'users',
            'profiles',
            'user_data',
            'customers',
            'clients',
            'members'
        ]
        
        print("🔍 Checking for existing tables...")
        
        for table_name in possible_table_names:
            try:
                print(f"   Checking table: {table_name}")
                result = supabase.table(table_name).select('*').limit(1).execute()
                print(f"   ✅ Found table '{table_name}' with {len(result.data)} records")
                
                if result.data:
                    print(f"   📋 Sample record from '{table_name}':")
                    sample = result.data[0]
                    for key, value in sample.items():
                        print(f"      {key}: {value}")
                    print()
                    
            except Exception as e:
                error_msg = str(e)
                if hasattr(e, 'message'):
                    error_msg = e.message
                print(f"   ❌ Table '{table_name}' not found: {error_msg}")
        
        # Try to get all tables using information_schema
        print("\n🔍 Trying to get all tables from information_schema...")
        try:
            # This might not work with all Supabase setups, but worth trying
            result = supabase.rpc('get_tables').execute()
            if result.data:
                print("📋 All tables in database:")
                for table in result.data:
                    print(f"   • {table}")
        except Exception as e:
            print(f"   ❌ Could not get table list: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Discovering Supabase Tables")
    print("=" * 50)
    
    if discover_tables():
        print("\n✅ Table discovery completed!")
    else:
        print("\n❌ Table discovery failed")
        print("📋 Please check your Supabase dashboard to see what tables exist")
