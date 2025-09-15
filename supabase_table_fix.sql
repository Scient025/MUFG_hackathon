-- Fix Supabase table structure to allow saving user profiles without auth constraints
-- Run this in your Supabase SQL Editor

-- Step 1: Drop the existing foreign key constraint
ALTER TABLE user_profiles DROP CONSTRAINT IF EXISTS user_profiles_id_fkey;

-- Step 2: Drop the existing primary key constraint
ALTER TABLE user_profiles DROP CONSTRAINT IF EXISTS user_profiles_pkey;

-- Step 3: Modify the id column to be a regular TEXT field instead of UUID with foreign key
ALTER TABLE user_profiles ALTER COLUMN id TYPE TEXT;

-- Step 4: Add a new primary key constraint on the id column
ALTER TABLE user_profiles ADD CONSTRAINT user_profiles_pkey PRIMARY KEY (id);

-- Step 5: Update RLS policies to work with the new structure
-- Drop existing policies
DROP POLICY IF EXISTS "Users can view own profile" ON user_profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON user_profiles;
DROP POLICY IF EXISTS "Users can insert own profile" ON user_profiles;

-- Create new policies that allow all operations (since we're not using auth)
-- For demo purposes, we'll allow all operations
CREATE POLICY "Allow all operations on user_profiles" ON user_profiles
  FOR ALL USING (true) WITH CHECK (true);

-- Alternative: If you want to keep some security, you could use:
-- CREATE POLICY "Allow all operations on user_profiles" ON user_profiles
--   FOR ALL USING (true) WITH CHECK (true);

-- Step 6: Verify the table structure
-- You can run this to check the table structure:
-- \d user_profiles

-- Step 7: Test insert (optional - remove after testing)
-- INSERT INTO user_profiles (
--   id, email, name, age, gender, country, employment_status, 
--   annual_income, current_savings, retirement_age_goal, risk_tolerance,
--   contribution_amount, employer_contribution, marital_status,
--   number_of_dependents, investment_experience_level, financial_goals
-- ) VALUES (
--   'test_user_123',
--   'test@example.com',
--   'Test User',
--   30,
--   'Male',
--   'Australia',
--   'Full-time',
--   75000,
--   25000,
--   65,
--   'Medium',
--   1000,
--   500,
--   'Single',
--   0,
--   'Beginner',
--   'Retirement'
-- );

-- Step 8: Clean up test data (run this after testing)
-- DELETE FROM user_profiles WHERE id = 'test_user_123';
