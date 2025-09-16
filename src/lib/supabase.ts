import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://imtmbgbktomztqtoyuvh.supabase.co'
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltdG1iZ2JrdG9tenRxdG95dXZoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc0MzM1MDEsImV4cCI6MjA3MzAwOTUwMX0.iqWnjWB4bNIg7DYom4V_ZCdqSL8hKptAMcmG1we2'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Database types - Updated to match MUFG table schema
export interface UserProfile {
  User_ID: string
  Name: string
  Age: number
  Gender: string
  Country: string
  Employment_Status: string
  Annual_Income: number
  Current_Savings: number
  Retirement_Age_Goal: number
  Risk_Tolerance: string
  Contribution_Amount: number
  Contribution_Frequency: string
  Employer_Contribution: number
  Years_Contributed: number
  Investment_Type: string
  Fund_Name: string
  Marital_Status: string
  Number_of_Dependents: number
  Education_Level: string
  Health_Status: string
  Home_Ownership_Status: string
  Investment_Experience_Level: string
  Financial_Goals: string
  Insurance_Coverage: string
  Pension_Type: string
  Withdrawal_Strategy: string
  // Additional fields from MUFG table
  Total_Annual_Contribution?: number
  Annual_Return_Rate?: number
  Volatility?: number
  Fees_Percentage?: number
  Projected_Pension_Amount?: number
  Expected_Annual_Payout?: number
  Inflation_Adjusted_Payout?: number
  Years_of_Payout?: number
  Survivor_Benefits?: string
  Life_Expectancy_Estimate?: number
  Debt_Level?: string
  Monthly_Expenses?: number
  Savings_Rate?: number
  Portfolio_Diversity_Score?: number
  Tax_Benefits_Eligibility?: string
  Government_Pension_Eligibility?: string
  Private_Pension_Eligibility?: string
  Account_Age?: number
  Risk_Tolerance_encoded?: number
  Investment_Type_encoded?: number
  Investment_Experience_Level_encoded?: number
}

export interface AuthUser {
  id: string
  email: string
  created_at: string
}
