# Manual Supabase Setup Instructions

## Step 1: Create the Table

1. Go to your Supabase dashboard: https://supabase.com/dashboard/project/imtmbgbktomztqtoyuvh
2. Click on "SQL Editor" in the left sidebar
3. Copy and paste the following SQL:

```sql
-- Create user_profiles table
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
```

4. Click "Run" to execute the SQL

## Step 2: Insert Sample Data

After creating the table, run this SQL to insert sample users:

```sql
-- Insert sample users
INSERT INTO user_profiles (
  id, email, name, age, gender, country, employment_status, 
  annual_income, current_savings, retirement_age_goal, risk_tolerance,
  contribution_amount, employer_contribution, marital_status,
  number_of_dependents, investment_experience_level, financial_goals
) VALUES 
(
  'user-001',
  'john.smith@example.com',
  'John Smith',
  35,
  'Male',
  'Australia',
  'Employed',
  85000,
  45000,
  65,
  'Medium',
  1200,
  600,
  'Married',
  2,
  'Intermediate',
  'Retirement planning and children education'
),
(
  'user-002',
  'sarah.johnson@example.com',
  'Sarah Johnson',
  28,
  'Female',
  'Australia',
  'Employed',
  65000,
  18000,
  60,
  'High',
  800,
  400,
  'Single',
  0,
  'Beginner',
  'Early retirement and travel'
),
(
  'user-003',
  'michael.brown@example.com',
  'Michael Brown',
  52,
  'Male',
  'Australia',
  'Self-employed',
  120000,
  180000,
  65,
  'Low',
  2000,
  0,
  'Married',
  1,
  'Expert',
  'Comfortable retirement'
),
(
  'user-004',
  'emma.wilson@example.com',
  'Emma Wilson',
  41,
  'Female',
  'Australia',
  'Employed',
  95000,
  75000,
  62,
  'Medium',
  1500,
  750,
  'Divorced',
  1,
  'Intermediate',
  'Financial independence'
),
(
  'user-005',
  'david.lee@example.com',
  'David Lee',
  29,
  'Male',
  'Australia',
  'Employed',
  72000,
  25000,
  65,
  'High',
  900,
  450,
  'Single',
  0,
  'Beginner',
  'Wealth building and property investment'
);
```

## Step 3: Test the Setup

After running both SQL scripts, run this command to test:

```bash
python test_supabase.py
```

## Step 4: Remove Mock Data

Once the database is working, we'll remove the mock data fallback from the code.

## Troubleshooting

If you get errors:
1. Make sure you're using the service role key (not the anon key)
2. Check that your Supabase project is active
3. Verify the table was created successfully in the "Table Editor"
4. Make sure the sample data was inserted correctly
