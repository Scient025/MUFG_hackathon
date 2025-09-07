// Mock data service for superannuation advisor dashboard
// This would typically connect to your Excel dataset

export interface UserProfile {
  User_ID: string;
  name: string;
  age: number;
  maritalStatus: string;
  dependents: number;
  riskProfile: 'Low' | 'Medium' | 'High';
  currentSavings: number;
  projectedPensionAmount: number;
  goal: number;
  expectedAnnualPayout: number;
  investmentType: string[];
  fundName: string[];
  financialGoals: string[];
  taxBenefitsEligibility: boolean;
  governmentPensionEligibility: boolean;
  privatePensionEligibility: boolean;
  riskTolerance: 'Low' | 'Medium' | 'High';
}

// Mock data representing 10 sample users from the dataset
export const sampleUsers: UserProfile[] = [
  {
    User_ID: "USER001",
    name: "Margaret Smith",
    age: 58,
    maritalStatus: "Married",
    dependents: 1,
    riskProfile: "Medium",
    currentSavings: 548750,
    projectedPensionAmount: 847000,
    goal: 1000000,
    expectedAnnualPayout: 38400,
    investmentType: ["ETFs", "Managed Funds"],
    fundName: ["Vanguard Balanced", "AustralianSuper Growth"],
    financialGoals: ["Retirement by 65", "Travel Fund", "Emergency Health Fund"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: true,
    privatePensionEligibility: true,
    riskTolerance: "Medium"
  },
  {
    User_ID: "USER002",
    name: "Robert Johnson",
    age: 52,
    maritalStatus: "Single",
    dependents: 0,
    riskProfile: "High",
    currentSavings: 425000,
    projectedPensionAmount: 720000,
    goal: 800000,
    expectedAnnualPayout: 32400,
    investmentType: ["Stocks", "ETFs"],
    fundName: ["ASX200 Index", "International Shares"],
    financialGoals: ["Early Retirement", "Property Investment"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: false,
    privatePensionEligibility: true,
    riskTolerance: "High"
  },
  {
    User_ID: "USER003",
    name: "Susan Williams",
    age: 45,
    maritalStatus: "Married",
    dependents: 2,
    riskProfile: "Low",
    currentSavings: 285000,
    projectedPensionAmount: 650000,
    goal: 750000,
    expectedAnnualPayout: 29250,
    investmentType: ["Fixed Income", "Cash"],
    fundName: ["Conservative Fund", "Term Deposits"],
    financialGoals: ["Children's Education", "Debt Clearance"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: true,
    privatePensionEligibility: false,
    riskTolerance: "Low"
  },
  {
    User_ID: "USER004",
    name: "Michael Brown",
    age: 61,
    maritalStatus: "Married",
    dependents: 0,
    riskProfile: "Medium",
    currentSavings: 720000,
    projectedPensionAmount: 950000,
    goal: 1200000,
    expectedAnnualPayout: 42750,
    investmentType: ["ETFs", "Property"],
    fundName: ["Property REITs", "Balanced Growth"],
    financialGoals: ["Retirement Comfort", "Travel"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: true,
    privatePensionEligibility: true,
    riskTolerance: "Medium"
  },
  {
    User_ID: "USER005",
    name: "Jennifer Davis",
    age: 38,
    maritalStatus: "Single",
    dependents: 1,
    riskProfile: "High",
    currentSavings: 185000,
    projectedPensionAmount: 580000,
    goal: 900000,
    expectedAnnualPayout: 26100,
    investmentType: ["Stocks", "ETFs", "Managed Funds"],
    fundName: ["Growth Fund", "International Shares"],
    financialGoals: ["Property Purchase", "Early Retirement"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: false,
    privatePensionEligibility: true,
    riskTolerance: "High"
  },
  {
    User_ID: "USER006",
    name: "David Wilson",
    age: 55,
    maritalStatus: "Married",
    dependents: 2,
    riskProfile: "Low",
    currentSavings: 485000,
    projectedPensionAmount: 680000,
    goal: 800000,
    expectedAnnualPayout: 30600,
    investmentType: ["Fixed Income", "Cash", "ETFs"],
    fundName: ["Conservative Balanced", "Cash Management"],
    financialGoals: ["Children's Support", "Health Fund"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: true,
    privatePensionEligibility: false,
    riskTolerance: "Low"
  },
  {
    User_ID: "USER007",
    name: "Linda Anderson",
    age: 49,
    maritalStatus: "Divorced",
    dependents: 1,
    riskProfile: "Medium",
    currentSavings: 320000,
    projectedPensionAmount: 620000,
    goal: 700000,
    expectedAnnualPayout: 27900,
    investmentType: ["ETFs", "Managed Funds"],
    fundName: ["Balanced Fund", "Australian Shares"],
    financialGoals: ["Financial Independence", "Travel"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: true,
    privatePensionEligibility: true,
    riskTolerance: "Medium"
  },
  {
    User_ID: "USER008",
    name: "James Taylor",
    age: 42,
    maritalStatus: "Married",
    dependents: 3,
    riskProfile: "High",
    currentSavings: 245000,
    projectedPensionAmount: 750000,
    goal: 1000000,
    expectedAnnualPayout: 33750,
    investmentType: ["Stocks", "ETFs", "Property"],
    fundName: ["Growth Fund", "Property REITs"],
    financialGoals: ["Children's Education", "Property Investment"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: false,
    privatePensionEligibility: true,
    riskTolerance: "High"
  },
  {
    User_ID: "USER009",
    name: "Patricia Thomas",
    age: 64,
    maritalStatus: "Widowed",
    dependents: 0,
    riskProfile: "Low",
    currentSavings: 680000,
    projectedPensionAmount: 720000,
    goal: 800000,
    expectedAnnualPayout: 32400,
    investmentType: ["Fixed Income", "Cash"],
    fundName: ["Conservative Fund", "Term Deposits"],
    financialGoals: ["Retirement Security", "Health Care"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: true,
    privatePensionEligibility: false,
    riskTolerance: "Low"
  },
  {
    User_ID: "USER010",
    name: "Christopher Jackson",
    age: 35,
    maritalStatus: "Single",
    dependents: 0,
    riskProfile: "High",
    currentSavings: 125000,
    projectedPensionAmount: 650000,
    goal: 1200000,
    expectedAnnualPayout: 29250,
    investmentType: ["Stocks", "ETFs"],
    fundName: ["Growth Fund", "International Shares"],
    financialGoals: ["Early Retirement", "Property Investment"],
    taxBenefitsEligibility: true,
    governmentPensionEligibility: false,
    privatePensionEligibility: true,
    riskTolerance: "High"
  }
];

// Peer comparison data
export const peerComparisonData = {
  ageGroups: {
    "35-45": { count: 3, avgContribution: 18500, avgBalance: 218333 },
    "46-55": { count: 4, avgContribution: 22100, avgBalance: 384000 },
    "56-65": { count: 3, avgContribution: 19800, avgBalance: 616667 }
  },
  riskGroups: {
    "Low": { count: 3, avgReturn: 6.2, commonInvestments: ["Fixed Income", "Cash"] },
    "Medium": { count: 4, avgReturn: 7.8, commonInvestments: ["ETFs", "Managed Funds"] },
    "High": { count: 3, avgReturn: 9.1, commonInvestments: ["Stocks", "ETFs"] }
  },
  investmentTypes: {
    "ETFs": { count: 6, percentage: 60 },
    "Stocks": { count: 4, percentage: 40 },
    "Managed Funds": { count: 5, percentage: 50 },
    "Fixed Income": { count: 3, percentage: 30 },
    "Cash": { count: 3, percentage: 30 },
    "Property": { count: 2, percentage: 20 }
  }
};

export const dataService = {
  getUserById: (userId: string): UserProfile | undefined => {
    return sampleUsers.find(user => user.User_ID === userId);
  },
  
  getAllUsers: (): UserProfile[] => {
    return sampleUsers;
  },
  
  getPeerComparison: (user: UserProfile) => {
    const ageGroup = user.age >= 56 ? "56-65" : user.age >= 46 ? "46-55" : "35-45";
    const riskGroup = user.riskProfile;
    
    return {
      ageGroup: peerComparisonData.ageGroups[ageGroup],
      riskGroup: peerComparisonData.riskGroups[riskGroup],
      investmentTypes: peerComparisonData.investmentTypes
    };
  },
  
  calculateRetirementProjection: (user: UserProfile) => {
    const yearsToRetirement = 65 - user.age;
    const monthlyContribution = (user.goal - user.currentSavings) / (yearsToRetirement * 12);
    const monthlyIncrease = Math.max(0, monthlyContribution - (user.expectedAnnualPayout / 12));
    
    return {
      retirementAmount: user.projectedPensionAmount,
      monthlyIncreaseNeeded: monthlyIncrease,
      targetAmount: user.goal,
      percentToGoal: Math.round((user.currentSavings / user.goal) * 100),
      monthlyIncomeAt65: user.expectedAnnualPayout / 12
    };
  }
};
