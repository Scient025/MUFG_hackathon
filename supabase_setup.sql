-- Create user_profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
  id UUID REFERENCES auth.users(id) PRIMARY KEY,
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

-- Enable Row Level Security
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Create policies
CREATE POLICY "Users can view own profile" ON user_profiles
  FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON user_profiles
  FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON user_profiles
  FOR INSERT WITH CHECK (auth.uid() = id);

-- Create function to automatically update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger to automatically update updated_at
CREATE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert some sample data (optional)
INSERT INTO user_profiles (
  id, email, name, age, gender, country, employment_status, 
  annual_income, current_savings, retirement_age_goal, risk_tolerance,
  contribution_amount, employer_contribution, marital_status,
  number_of_dependents, investment_experience_level, financial_goals
) VALUES (
  '00000000-0000-0000-0000-000000000001',
  'demo@example.com',
  'Demo User',
  30,
  'Male',
  'Australia',
  'Full-time',
  75000,
  25000,
  65,
  'Medium',
  1000,
  500,
  'Single',
  0,
  'Beginner',
  'Retirement'
) ON CONFLICT (id) DO NOTHING;
