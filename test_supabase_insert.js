// Test script to directly insert data into Supabase MUFG table
// Run this in browser console on your app page

// First, let's check if Supabase is available
console.log('Checking if Supabase is available...');

// Get the Supabase client from your app
const supabaseUrl = 'https://imtmbgbktomztqtoyuvh.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImltdG1iZ2JrdG9tenRxdG95dXZoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc0MzM1MDEsImV4cCI6MjA3MzAwOTUwMX0.iqWnjWB4bNIg7DYom4V_ZCdqSL8hKptAMcmG1we2';

// Create Supabase client directly
const supabase = window.supabase || createClient(supabaseUrl, supabaseAnonKey);

const testData = {
  "User_ID": "test_user_12345",
  "Name": "Test User",
  "Age": 30,
  "Gender": "Male",
  "Country": "Australia",
  "Employment_Status": "Full-time",
  "Annual_Income": 75000,
  "Current_Savings": 25000,
  "Retirement_Age_Goal": 65,
  "Risk_Tolerance": "Medium",
  "Contribution_Amount": 1000,
  "Contribution_Frequency": "Monthly",
  "Employer_Contribution": 500,
  "Years_Contributed": 5,
  "Investment_Type": "Balanced",
  "Fund_Name": "Default Fund",
  "Marital_Status": "Single",
  "Number_of_Dependents": 0,
  "Education_Level": "Bachelor",
  "Health_Status": "Good",
  "Home_Ownership_Status": "Renting",
  "Investment_Experience_Level": "Beginner",
  "Financial_Goals": "Retirement",
  "Insurance_Coverage": "Basic",
  "Pension_Type": "Superannuation",
  "Withdrawal_Strategy": "Fixed",
  "Total_Annual_Contribution": null,
  "Annual_Return_Rate": null,
  "Volatility": null,
  "Fees_Percentage": null,
  "Projected_Pension_Amount": null,
  "Expected_Annual_Payout": null,
  "Inflation_Adjusted_Payout": null,
  "Years_of_Payout": null,
  "Survivor_Benefits": null,
  "Transaction_ID": null,
  "Transaction_Amount": null,
  "Transaction_Date": null,
  "Suspicious_Flag": null,
  "Anomaly_Score": null,
  "Life_Expectancy_Estimate": null,
  "Debt_Level": null,
  "Monthly_Expenses": null,
  "Savings_Rate": null,
  "Portfolio_Diversity_Score": null,
  "Tax_Benefits_Eligibility": null,
  "Government_Pension_Eligibility": null,
  "Private_Pension_Eligibility": null,
  "Transaction_Channel": null,
  "IP_Address": null,
  "Device_ID": null,
  "Geo_Location": null,
  "Time_of_Transaction": null,
  "Transaction_Pattern_Score": null,
  "Previous_Fraud_Flag": null,
  "Account_Age": null,
  "Risk_Tolerance_encoded": null,
  "Investment_Type_encoded": null,
  "Investment_Experience_Level_encoded": null
};

console.log('Test data prepared:', testData);

// Test the insertion
async function testSupabaseInsert() {
  try {
    console.log('Attempting to insert test data...');
    
    const { data, error } = await supabase
      .from('MUFG')
      .insert(testData)
      .select()
      .single();
    
    if (error) {
      console.error('❌ Supabase insertion failed:', error);
      console.error('Error details:', {
        message: error.message,
        details: error.details,
        hint: error.hint,
        code: error.code
      });
    } else {
      console.log('✅ Supabase insertion successful:', data);
    }
    
  } catch (err) {
    console.error('❌ Test failed:', err);
  }
}

// Run the test
testSupabaseInsert();
