import { createClient } from '@supabase/supabase-js'

const supabaseUrl = 'https://imtmbgbktomztqtoyuvh.supabase.co'
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltdG1iZ2JrdG9tenRxdG95dXZoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc0MzM1MDEsImV4cCI6MjA3MzAwOTUwMX0.iqWnjWB4bNIg7DYom4V_ZCdqSL8hKptAMcmG1we2'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)

// Database types
export interface UserProfile {
  id: string
  email: string
  name: string
  age: number
  gender: string
  country: string
  employment_status: string
  annual_income: number
  current_savings: number
  retirement_age_goal: number
  risk_tolerance: string
  contribution_amount: number
  contribution_frequency: string
  employer_contribution: number
  years_contributed: number
  investment_type: string
  fund_name: string
  marital_status: string
  number_of_dependents: number
  education_level: string
  health_status: string
  home_ownership_status: string
  investment_experience_level: string
  financial_goals: string
  insurance_coverage: string
  pension_type: string
  withdrawal_strategy: string
  created_at: string
  updated_at: string
}

export interface AuthUser {
  id: string
  email: string
  created_at: string
}
