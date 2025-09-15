// Data service for superannuation advisor dashboard
// Connects to ML backend API

const API_BASE_URL = '/api';

export interface UserProfile {
  User_ID: string;
  Name?: string; // Generated name from backend
  Age: number;
  Gender: string;
  Country: string;
  Employment_Status: string;
  Annual_Income: number;
  Current_Savings: number;
  Retirement_Age_Goal: number;
  Risk_Tolerance: 'Low' | 'Medium' | 'High';
  Contribution_Amount: number;
  Contribution_Frequency: string;
  Employer_Contribution: number;
  Total_Annual_Contribution: number;
  Years_Contributed: number;
  Investment_Type: string;
  Fund_Name: string;
  Annual_Return_Rate: number;
  Volatility: number;
  Fees_Percentage: number;
  Projected_Pension_Amount: number;
  Expected_Annual_Payout: number;
  Inflation_Adjusted_Payout: number;
  Years_of_Payout: number;
  Survivor_Benefits: string;
  Marital_Status: string;
  Number_of_Dependents: number;
  Education_Level: string;
  Health_Status: string;
  Life_Expectancy_Estimate: number;
  Home_Ownership_Status: string;
  Debt_Level: number;
  Monthly_Expenses: number;
  Savings_Rate: number;
  Investment_Experience_Level: string;
  Financial_Goals: string;
  Insurance_Coverage: string;
  Portfolio_Diversity_Score: number;
  Tax_Benefits_Eligibility: string;
  Government_Pension_Eligibility: string;
  Private_Pension_Eligibility: string;
  Pension_Type: string;
  Withdrawal_Strategy: string;
}

// Sample user IDs from the dataset (first 10 users)
export const sampleUserIds = [
  "U1000", "U1001", "U1002", "U1003", "U1004", 
  "U1005", "U1006", "U1007", "U1008", "U1009"
];

// API service for connecting to ML backend
export const dataService = {
  // Get user profile from API
  getUserById: async (userId: string): Promise<UserProfile | null> => {
    try {
      console.log(`Fetching user ${userId} from ${API_BASE_URL}/user/${userId}`);
      const response = await fetch(`${API_BASE_URL}/user/${userId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: User ${userId} not found`);
      }
      const data = await response.json();
      console.log('User data received:', data);
      return data.success ? data.data : null;
    } catch (error) {
      console.error('Error fetching user:', error);
      throw error; // Re-throw to let caller handle
    }
  },
  
  // Get all available user IDs
  getAllUserIds: (): string[] => {
    return sampleUserIds;
  },
  
  // Get summary statistics for dashboard
  getSummaryStats: async (userId: string) => {
    try {
      console.log(`Fetching summary for ${userId}`);
      const response = await fetch(`${API_BASE_URL}/summary/${userId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Summary not found for user ${userId}`);
      }
      const data = await response.json();
      console.log('Summary data received:', data);
      return data.success ? data.data : null;
    } catch (error) {
      console.error('Error fetching summary:', error);
      throw error;
    }
  },
  
  // Get peer comparison data
  getPeerComparison: async (userId: string) => {
    try {
      console.log(`Fetching peer stats for ${userId}`);
      const response = await fetch(`${API_BASE_URL}/peer_stats/${userId}`);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Peer stats not found for user ${userId}`);
      }
      const data = await response.json();
      console.log('Peer stats received:', data);
      return data.success ? data.data : null;
    } catch (error) {
      console.error('Error fetching peer stats:', error);
      throw error;
    }
  },
  
  // Get pension projection
  getPensionProjection: async (userId: string, extraMonthly: number = 0) => {
    try {
      console.log(`Fetching projection for ${userId} with extra monthly: ${extraMonthly}`);
      const response = await fetch(`${API_BASE_URL}/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          extra_monthly: extraMonthly
        })
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Projection not found for user ${userId}`);
      }
      const data = await response.json();
      console.log('Projection data received:', data);
      return data.success ? data.data : null;
    } catch (error) {
      console.error('Error fetching projection:', error);
      throw error;
    }
  },
  
  // Get risk prediction
  getRiskPrediction: async (userId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/risk/${userId}`);
      if (!response.ok) {
        throw new Error(`Risk prediction not found for user ${userId}`);
      }
      const data = await response.json();
      return data.success ? data.data : null;
    } catch (error) {
      console.error('Error fetching risk prediction:', error);
      return null;
    }
  },
  
  // Send chat message to AI
  sendChatMessage: async (userId: string, message: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          message: message
        })
      });
      if (!response.ok) {
        throw new Error(`Chat response error`);
      }
      const data = await response.json();
      return data.success ? data.data : null;
    } catch (error) {
      console.error('Error sending chat message:', error);
      return null;
    }
  },

  // Signup interfaces
  signupUser: async (signupData: SignupData) => {
    try {
      const response = await fetch(`${API_BASE_URL}/signup`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(signupData)
      });
      
      if (!response.ok) {
        throw new Error(`Signup failed: ${response.statusText}`);
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error signing up user:', error);
      throw error;
    }
  },

  getAllUsers: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/users`);
      if (!response.ok) {
        throw new Error(`Failed to fetch users: ${response.statusText}`);
      }
      const data = await response.json();
      return data.users || [];
    } catch (error) {
      console.error('Error fetching users:', error);
      return [];
    }
  }
};

// Signup data interface
export interface SignupData {
  name: string;
  age: number;
  gender: string;
  country: string;
  employment_status: string;
  annual_income: number;
  current_savings: number;
  retirement_age_goal: number;
  risk_tolerance: string;
  contribution_amount: number;
  contribution_frequency: string;
  employer_contribution: number;
  years_contributed: number;
  investment_type: string;
  fund_name: string;
  marital_status: string;
  number_of_dependents: number;
  education_level: string;
  health_status: string;
  home_ownership_status: string;
  investment_experience_level: string;
  financial_goals: string;
  insurance_coverage: string;
  pension_type: string;
  withdrawal_strategy: string;
}

// User list interface
export interface User {
  User_ID: string;
  Name: string;
  Age: number;
  Risk_Tolerance: string;
  Annual_Income: number;
  Current_Savings: number;
}
