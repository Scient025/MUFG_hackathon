import { useState, useEffect, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { DashboardHeader } from "@/components/dashboard/DashboardHeader";
import { SummaryCard } from "@/components/dashboard/SummaryCard";
import { MetricsGrid } from "@/components/dashboard/MetricsGrid";
import { NavigationTabs } from "@/components/dashboard/NavigationTabs";
import { PortfolioPage } from "@/components/dashboard/PortfolioPage";
import { GoalsPage } from "@/components/dashboard/GoalsPage";
import { EducationPage } from "@/components/dashboard/EducationPage";
import { RiskPage } from "@/components/dashboard/RiskPage";
import { ChatbotPageWithSpeech } from "@/components/dashboard/ChatbotPageWithSpeech";
import { FloatingChatButton } from "@/components/dashboard/FloatingChatButton";
import { Button } from "@/components/ui/button";
import { useAuth } from "@/contexts/AuthContext";
import { useAdminAuth } from "@/contexts/AdminAuthContext";
import { SupabaseService } from "@/services/supabaseService";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Mail, Loader2, Shield } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

import { LogOut, User, ArrowLeft } from "lucide-react";

export default function Dashboard() {
  console.log('Dashboard component rendering...');
  const { user, loading: authLoading, signOut } = useAuth();
  const { adminUser, isAdminMode, logout: adminLogout } = useAdminAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [activeTab, setActiveTab] = useState("chatbot");
  const [goals, setGoals] = useState<any[]>([]);
  const [currentUser, setCurrentUser] = useState<any | null>(null);
  const [summaryStats, setSummaryStats] = useState<any>(null);
  const [peerComparison, setPeerComparison] = useState<any>(null);
  const [projection, setProjection] = useState<any>(null);
  const [customUserId, setCustomUserId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [initializing, setInitializing] = useState(true);
  const [advancedMetrics, setAdvancedMetrics] = useState<any>(null);

  const initRef = useRef(false);
  useEffect(() => {
    if (initRef.current) return;
    initRef.current = true;
    
    const userIdFromState = (location.state as any)?.userId;
    const currentUserFromStorage = localStorage.getItem('currentUser');
    const userSessionFromStorage = localStorage.getItem('userSession');
    console.log('Checking for custom userId:', { userIdFromState, currentUserFromStorage });
    if (userIdFromState) {
      console.log('Got userId from navigation state:', userIdFromState);
      setCustomUserId(userIdFromState);
    } else if (currentUserFromStorage) {
      try {
        const userData = JSON.parse(currentUserFromStorage);
        console.log('Got userId from localStorage:', userData.userId);
        setCustomUserId(userData.userId);
      } catch (error) {
        console.error('Error parsing currentUser from localStorage:', error);
      }
    } else if (userSessionFromStorage) {
      try {
        const session = JSON.parse(userSessionFromStorage);
        if (session?.userId) {
          console.log('Got userId from userSession:', session.userId);
          setCustomUserId(session.userId);
        }
      } catch (error) {
        console.error('Error parsing userSession from localStorage:', error);
      }
    } else {
      console.log('No custom userId found');
    }
    // Mark initialization as complete
    setInitializing(false);
  }, []);

  const loadAdvancedMetrics = async (userId: string) => {
    try {
      console.log('Loading advanced metrics for userId:', userId);
      const response = await fetch(`/api/advanced_analysis/${userId}`);
      console.log('Advanced metrics response status:', response.status);
      const data = await response.json();
      console.log('Advanced metrics API response:', data);
      
      if (data.success) {
        const analysis = data.data;
        console.log('Analysis data:', analysis);
        const metrics = {
          financial_health_score: analysis.financial_health?.financial_health_score || 0,
          churn_risk_percentage: (analysis.churn_risk?.churn_probability || 0) * 100,
          anomaly_score: analysis.anomaly_detection?.anomaly_percentage || 0
        };
        console.log('Setting advanced metrics:', metrics);
        setAdvancedMetrics(metrics);
      } else {
        console.log('Advanced metrics API failed, using fallback calculations');
        // Fallback calculations based on user data
        const user = currentUser;
        if (user) {
          const healthScore = calculateFinancialHealthScore(user);
          const churnRisk = calculateChurnRisk(user);
          // Anomaly score is not available in Supabase - only from ML API
          const anomalyScore = 0; // Default to 0 if ML API fails
          
          console.log('Fallback calculations:', { healthScore, churnRisk, anomalyScore });
          setAdvancedMetrics({
            financial_health_score: healthScore,
            churn_risk_percentage: churnRisk,
            anomaly_score: anomalyScore
          });
        }
      }
    } catch (error) {
      console.error('Error loading advanced metrics:', error);
      // Fallback calculations
      const user = currentUser;
      if (user) {
        const healthScore = calculateFinancialHealthScore(user);
        const churnRisk = calculateChurnRisk(user);
        // Anomaly score is not available in Supabase - only from ML API
        const anomalyScore = 0; // Default to 0 if ML API fails
        
        console.log('Error fallback calculations:', { healthScore, churnRisk, anomalyScore });
        setAdvancedMetrics({
          financial_health_score: healthScore,
          churn_risk_percentage: churnRisk,
          anomaly_score: anomalyScore
        });
      }
    }
  };

  const calculateFinancialHealthScore = (user: any) => {
    let score = 0;
    
    console.log('Calculating financial health score for user:', user);
    
    // Income component (20 points)
    const income = user.Annual_Income || 0;
    console.log('User income:', income);
    if (income > 100000) score += 20;
    else if (income > 75000) score += 15;
    else if (income > 50000) score += 10;
    else score += 5;
    
    // Savings component (25 points)
    const savingsRatio = (user.Current_Savings || 0) / (income || 1);
    if (savingsRatio > 2.0) score += 25;
    else if (savingsRatio > 1.0) score += 20;
    else if (savingsRatio > 0.5) score += 15;
    else if (savingsRatio > 0.2) score += 10;
    else score += 5;
    
    // Contribution component (20 points)
    const contribRatio = (user.Contribution_Amount || 0) / (income || 1);
    if (contribRatio > 0.15) score += 20;
    else if (contribRatio > 0.10) score += 15;
    else if (contribRatio > 0.05) score += 10;
    else score += 5;
    
    // Debt component (15 points) - simplified
    const debtLevel = user.Debt_Level || 0;
    if (debtLevel < 20000) score += 15;
    else if (debtLevel < 50000) score += 10;
    else if (debtLevel < 100000) score += 5;
    
    // Age component (10 points)
    const age = user.Age || 0;
    if (age >= 25 && age <= 35) score += 10;
    else if (age >= 36 && age <= 45) score += 8;
    else if (age >= 46 && age <= 55) score += 6;
    else score += 4;
    
    // Investment experience (10 points)
    const expLevel = user.Investment_Experience_Level || 'Beginner';
    if (expLevel === 'Expert') score += 10;
    else if (expLevel === 'Intermediate') score += 7;
    else score += 4;
    
    const finalScore = Math.min(100, Math.max(0, score));
    console.log('Final financial health score:', finalScore);
    return finalScore;
  };

  const calculateChurnRisk = (user: any) => {
    let risk = 0;
    
    console.log('Calculating churn risk for user:', user);
    
    // Age factor
    const age = user.Age || 0;
    console.log('User age:', age);
    if (age < 30) risk += 15;
    else if (age < 40) risk += 10;
    else if (age < 50) risk += 5;
    
    // Employment status
    const employment = user.Employment_Status || '';
    if (employment === 'Unemployed') risk += 20;
    else if (employment === 'Part-time') risk += 10;
    else if (employment === 'Self-employed') risk += 5;
    
    // Contribution frequency
    const freq = user.Contribution_Frequency || '';
    if (freq === 'Annually') risk += 15;
    else if (freq === 'Quarterly') risk += 8;
    else if (freq === 'Monthly') risk += 2;
    
    // Years contributed
    const years = user.Years_Contributed || 0;
    if (years < 2) risk += 20;
    else if (years < 5) risk += 10;
    else if (years < 10) risk += 5;
    
    // Income stability
    const income = user.Annual_Income || 0;
    if (income < 40000) risk += 15;
    else if (income < 60000) risk += 8;
    else if (income < 80000) risk += 3;
    
    const finalRisk = Math.min(100, Math.max(0, risk));
    console.log('Final churn risk:', finalRisk);
    return finalRisk;
  };

  const loadUserData = async () => {
    // Use custom userId if available, otherwise fall back to admin mode
    const userIdToUse = customUserId || (isAdminMode ? adminUser?.User_ID : null);
    
    if (!userIdToUse) {
      console.log('No userId available for loading data');
      return;
    }

    console.log('Loading user data for userId:', userIdToUse);
    
    if (isAdminMode && adminUser) {
      console.log('Admin user data:', adminUser);
      console.log('Admin user Employer_Contribution:', adminUser.Employer_Contribution);
      console.log('Admin user Contribution_Amount:', adminUser.Contribution_Amount);
      console.log('Admin user Anomaly_Score:', adminUser.Anomaly_Score);
      setCurrentUser(adminUser);
      
      // Get real projection data from ML API for admin user
      try {
        const projectionResponse = await fetch(`/api/projection/${adminUser.User_ID}`);
        const projectionData = await projectionResponse.json();
        
        if (projectionData.success) {
          // Calculate monthly increase needed
          // The user wants to reach their Projected_Pension_Amount goal
          // The ML model predicts they'll have adjusted_projection with current contributions
          // We need to calculate how much more they need to contribute monthly to reach their goal
          const retirementGoal = adminUser.Projected_Pension_Amount || adminUser.Current_Savings * 5; // What they want to have
          const projectedAmount = projectionData.data.adjusted_projection; // What they'll actually have with current contributions
          const currentSavings = adminUser.Current_Savings; // What they have now
          const yearsToRetirement = projectionData.data.years_to_retirement || 35;
          
          console.log('Projection calculation debug:', {
            retirementGoal,
            projectedAmount,
            currentSavings,
            yearsToRetirement,
            adjusted_projection: projectionData.data.adjusted_projection,
            years_to_retirement: projectionData.data.years_to_retirement,
            gap: retirementGoal - projectedAmount,
            totalMonths: yearsToRetirement * 12,
            isExceedingGoal: projectedAmount > retirementGoal
          });
          
          // Calculate how much more they need to contribute monthly to reach their goal
          // If they're already exceeding their goal, show 0
          // If they're short, calculate the monthly increase needed
          const monthlyIncreaseNeeded = Math.max(0, (retirementGoal - projectedAmount) / (yearsToRetirement * 12));
          
          console.log('Monthly increase needed calculation:', {
            numerator: retirementGoal - projectedAmount,
            denominator: yearsToRetirement * 12,
            result: monthlyIncreaseNeeded,
            isExceedingGoal: projectedAmount > retirementGoal,
            excessAmount: projectedAmount - retirementGoal
          });
          
          setProjection({
            ...projectionData.data,
            monthly_increase_needed: Number(monthlyIncreaseNeeded) // Ensure it's a native JavaScript number
          });
          setSummaryStats({
            current_savings: adminUser.Current_Savings,
            projected_pension: projectionData.data.adjusted_projection || adminUser.Projected_Pension_Amount,
            percent_to_goal: ((adminUser.Current_Savings / (projectionData.data.adjusted_projection || adminUser.Projected_Pension_Amount)) * 100),
            monthly_income_at_retirement: projectionData.data.monthly_income_at_retirement || 0
          });
        } else {
          // Fallback to mock data if API fails
          const targetAmount = adminUser.Projected_Pension_Amount || adminUser.Current_Savings * 5;
          const monthlyIncreaseNeeded = Math.max(0, (targetAmount - adminUser.Current_Savings) / (35 * 12));
          
          setSummaryStats({
            current_savings: adminUser.Current_Savings,
            projected_pension: targetAmount,
            percent_to_goal: 20,
            monthly_income_at_retirement: targetAmount / 12
          });
          setProjection({
            current_projection: adminUser.Current_Savings * 5,
            adjusted_projection: targetAmount,
            monthly_income_at_retirement: targetAmount / 12,
            monthly_increase_needed: Number(monthlyIncreaseNeeded) // Ensure it's a native JavaScript number
          });
        }
      } catch (error) {
        console.error('Error loading projection data for admin user:', error);
        // Fallback to mock data
        const targetAmount = adminUser.Projected_Pension_Amount || adminUser.Current_Savings * 5;
        const monthlyIncreaseNeeded = Math.max(0, (targetAmount - adminUser.Current_Savings) / (35 * 12));
        
        setSummaryStats({
          current_savings: adminUser.Current_Savings,
          projected_pension: targetAmount,
          percent_to_goal: 20,
          monthly_income_at_retirement: targetAmount / 12
        });
        setProjection({
          current_projection: adminUser.Current_Savings * 5,
          adjusted_projection: targetAmount,
          monthly_income_at_retirement: targetAmount / 12,
          monthly_increase_needed: Number(monthlyIncreaseNeeded) // Ensure it's a native JavaScript number
        });
      }
      
      // Get real peer comparison data from ML API for admin user
      try {
        const peerResponse = await fetch(`/api/peer_stats/${adminUser.User_ID}`);
        const peerData = await peerResponse.json();
        
        if (peerData.success) {
          // Convert common_investment_types to investment_types format expected by SummaryCard
          const investmentTypes = {};
          const peerStats = peerData.data.peer_stats || {};
          
          console.log('Admin Peer data received:', peerData);
          console.log('Admin Peer stats:', peerStats);
          console.log('Admin Common investment types:', peerStats.common_investment_types);
          
          if (peerStats.common_investment_types) {
            const totalPeers = peerStats.total_peers || 1;
            Object.entries(peerStats.common_investment_types).forEach(([type, count]) => {
              investmentTypes[type] = {
                count: count,
                percentage: Math.round((count / totalPeers) * 100)
              };
            });
          }
          
          console.log('Admin Processed investment types:', investmentTypes);
          
          setPeerComparison({
            total_peers: peerStats.total_peers || 0,
            avg_age: peerStats.avg_age || 0,
            avg_income: peerStats.avg_income || 0,
            avg_savings: peerStats.avg_savings || 0,
            avg_contribution: peerStats.avg_contribution || 0,
            investment_types: investmentTypes
          });
        } else {
          // Fallback to mock data
          setPeerComparison({
            total_peers: 100,
            avg_age: 35,
            avg_income: 75000,
            avg_savings: 25000,
            avg_contribution: 1200,
            investment_types: {
              'ETF': { count: 45, percentage: 45 },
              'Managed Fund': { count: 30, percentage: 30 },
              'Index Fund': { count: 25, percentage: 25 }
            }
          });
        }
      } catch (error) {
        console.error('Error loading peer data for admin user:', error);
        // Fallback to mock data
        setPeerComparison({
          total_peers: 100,
          avg_age: 35,
          avg_income: 75000,
          avg_savings: 25000,
          avg_contribution: 1200,
          investment_types: {
            'ETF': { count: 45, percentage: 45 },
            'Managed Fund': { count: 30, percentage: 30 },
            'Index Fund': { count: 25, percentage: 25 }
          }
        });
      }
      
      return;
    }
    
    // For custom login, use the custom userId
    if (customUserId) {
      try {
        setError(null);
        setLoading(true);
        
        console.log('Loading user profile for custom userId:', customUserId);
        
        // Load user profile from Supabase using the custom userId
        const userProfile = await SupabaseService.getUserProfile(customUserId);
        
        if (userProfile) {
          setCurrentUser(userProfile);
          
          // Get real projection data from ML API
          try {
            const projectionResponse = await fetch(`/api/projection/${customUserId}`);
            const projectionData = await projectionResponse.json();
            
            if (projectionData.success) {
              // Calculate monthly increase needed
              // The user wants to reach their Projected_Pension_Amount goal
              // The ML model predicts they'll have adjusted_projection with current contributions
              // We need to calculate how much more they need to contribute monthly to reach their goal
              const retirementGoal = userProfile.Projected_Pension_Amount || userProfile.Current_Savings * 5; // What they want to have
              const projectedAmount = projectionData.data.adjusted_projection; // What they'll actually have with current contributions
              const currentSavings = userProfile.Current_Savings; // What they have now
              const yearsToRetirement = projectionData.data.years_to_retirement || 35;
              
              console.log('Custom user projection calculation debug:', {
                retirementGoal,
                projectedAmount,
                currentSavings,
                yearsToRetirement,
                adjusted_projection: projectionData.data.adjusted_projection,
                years_to_retirement: projectionData.data.years_to_retirement,
                gap: retirementGoal - projectedAmount,
                totalMonths: yearsToRetirement * 12,
                isExceedingGoal: projectedAmount > retirementGoal
              });
              
              // Calculate how much more they need to contribute monthly to reach their goal
              // If they're already exceeding their goal, show 0
              // If they're short, calculate the monthly increase needed
              const monthlyIncreaseNeeded = Math.max(0, (retirementGoal - projectedAmount) / (yearsToRetirement * 12));
              
              console.log('Custom user monthly increase needed calculation:', {
                numerator: retirementGoal - projectedAmount,
                denominator: yearsToRetirement * 12,
                result: monthlyIncreaseNeeded,
                isExceedingGoal: projectedAmount > retirementGoal,
                excessAmount: projectedAmount - retirementGoal
              });
              
              setProjection({
                ...projectionData.data,
                monthly_increase_needed: Number(monthlyIncreaseNeeded) // Ensure it's a native JavaScript number
              });
              setSummaryStats({
                current_savings: userProfile.Current_Savings,
                projected_pension: projectionData.data.adjusted_projection || userProfile.Projected_Pension_Amount,
                percent_to_goal: ((userProfile.Current_Savings / (projectionData.data.adjusted_projection || userProfile.Projected_Pension_Amount)) * 100),
                monthly_income_at_retirement: projectionData.data.monthly_income_at_retirement || 0
              });
            } else {
              // Fallback to mock data if API fails
              const targetAmount = userProfile.Projected_Pension_Amount || userProfile.Current_Savings * 5;
              const monthlyIncreaseNeeded = Math.max(0, (targetAmount - userProfile.Current_Savings) / (35 * 12)); // Assume 35 years to retirement
              
              setSummaryStats({
                current_savings: userProfile.Current_Savings,
                projected_pension: targetAmount,
                percent_to_goal: 20,
                monthly_income_at_retirement: targetAmount / 12
              });
              setProjection({
                current_projection: userProfile.Current_Savings * 5,
                adjusted_projection: targetAmount,
                monthly_income_at_retirement: targetAmount / 12,
                monthly_increase_needed: Number(monthlyIncreaseNeeded) // Ensure it's a native JavaScript number
              });
            }
          } catch (error) {
            console.error('Error loading projection data:', error);
            // Fallback to mock data
            const targetAmount = userProfile.Projected_Pension_Amount || userProfile.Current_Savings * 5;
            const monthlyIncreaseNeeded = Math.max(0, (targetAmount - userProfile.Current_Savings) / (35 * 12));
            
            setSummaryStats({
              current_savings: userProfile.Current_Savings,
              projected_pension: targetAmount,
              percent_to_goal: 20,
              monthly_income_at_retirement: targetAmount / 12
            });
            setProjection({
              current_projection: userProfile.Current_Savings * 5,
              adjusted_projection: targetAmount,
              monthly_income_at_retirement: targetAmount / 12,
              monthly_increase_needed: Number(monthlyIncreaseNeeded) // Ensure it's a native JavaScript number
            });
          }
          
          // Get real peer comparison data from ML API
          try {
            const peerResponse = await fetch(`/api/peer_stats/${customUserId}`);
            const peerData = await peerResponse.json();
            
            if (peerData.success) {
              // Convert common_investment_types to investment_types format expected by SummaryCard
              const investmentTypes = {};
              const peerStats = peerData.data.peer_stats || {};
              
              console.log('Peer data received:', peerData);
              console.log('Peer stats:', peerStats);
              console.log('Common investment types:', peerStats.common_investment_types);
              
              if (peerStats.common_investment_types) {
                const totalPeers = peerStats.total_peers || 1;
                Object.entries(peerStats.common_investment_types).forEach(([type, count]) => {
                  investmentTypes[type] = {
                    count: count,
                    percentage: Math.round((count / totalPeers) * 100)
                  };
                });
              }
              
              console.log('Processed investment types:', investmentTypes);
              
              setPeerComparison({
                total_peers: peerStats.total_peers || 0,
                avg_age: peerStats.avg_age || 0,
                avg_income: peerStats.avg_income || 0,
                avg_savings: peerStats.avg_savings || 0,
                avg_contribution: peerStats.avg_contribution || 0,
                investment_types: investmentTypes
              });
            } else {
              // Fallback to mock data
              setPeerComparison({
                total_peers: 100,
                avg_age: 35,
                avg_income: 75000,
                avg_savings: 25000,
                avg_contribution: 1200,
                investment_types: {
                  'ETF': { count: 45, percentage: 45 },
                  'Managed Fund': { count: 30, percentage: 30 },
                  'Index Fund': { count: 25, percentage: 25 }
                }
              });
            }
          } catch (error) {
            console.error('Error loading peer data:', error);
            // Fallback to mock data
            setPeerComparison({
              total_peers: 100,
              avg_age: 35,
              avg_income: 75000,
              avg_savings: 25000,
              avg_contribution: 1200,
              investment_types: {
                'ETF': { count: 45, percentage: 45 },
                'Managed Fund': { count: 30, percentage: 30 },
                'Index Fund': { count: 25, percentage: 25 }
              }
            });
          }
        } else {
          setError('User profile not found');
        }
      } catch (error) {
        console.error('Error loading custom user data:', error);
        setError('Failed to load user data');
      } finally {
        setLoading(false);
      }
      
      return;
    }
    
    // Fallback to Supabase auth user (if any)
    if (!user) return;
    
    try {
      setError(null);
      setLoading(true);
      
      const userProfile = await SupabaseService.getUserProfile(user.id);
      
      if (userProfile) {
        setCurrentUser(userProfile);
        // Generate mock data for now - you can replace with real calculations
        setSummaryStats({
          current_savings: userProfile.Current_Savings,
          projected_pension: userProfile.Current_Savings * 5, 
          percent_to_goal: 20,
          monthly_income_at_retirement: userProfile.Current_Savings * 0.04 / 12
        });
        setPeerComparison({
          total_peers: 100,
          avg_age: 35,
          avg_income: 75000,
          avg_savings: 25000,
          avg_contribution: 1200
        });
        setProjection({
          current_projection: userProfile.Current_Savings * 5,
          adjusted_projection: (userProfile.Current_Savings || 0) * 6,
          years_to_retirement: (userProfile.Retirement_Age_Goal || 65) - (userProfile.Age || 0),
          monthly_income_at_retirement: userProfile.Current_Savings * 0.04 / 12
        });
        
      }
    } catch (error) {
      console.error('Error loading user data:', error);
      setError(error instanceof Error ? error.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSignOut = async () => {
    if (isAdminMode) {
      adminLogout();
      navigate('/user-manager');
    } else if (customUserId) {
      // Handle custom login sign out
      localStorage.removeItem('currentUser');
      localStorage.removeItem('userSession');
      setCustomUserId(null);
      navigate('/login');
    } else {
      try { localStorage.removeItem('userSession'); } catch (e) {}
      await signOut();
      navigate('/login');
    }
  };

  // Redirect to login if not authenticated (unless in admin mode)
  const redirectingRef = useRef(false);
  useEffect(() => {
    console.log('Auth check:', { authLoading, user: !!user, isAdminMode, customUserId, initializing });
    if (redirectingRef.current) return;
    // Only redirect if we're sure there's no authentication method available and initialization is complete
    if (!authLoading && !user && !isAdminMode && !customUserId && !initializing) {
      console.log('No authentication found, redirecting to login');
      redirectingRef.current = true;
      try {
        localStorage.removeItem('userSession');
        localStorage.removeItem('currentUser');
      } catch (e) {}
      navigate('/login', { replace: true });
    }
  }, [user, authLoading, isAdminMode, customUserId, initializing, navigate]);

  // Load user data when user is authenticated or in admin mode
  useEffect(() => {
    if (user || isAdminMode || customUserId) {
      loadUserData();
    }
  }, [user, isAdminMode, customUserId]);

  // Load advanced metrics when currentUser is set
  useEffect(() => {
    if (currentUser && (isAdminMode || customUserId || user)) {
      const userId = customUserId || (isAdminMode ? currentUser.User_ID : user?.id);
      if (userId) {
        loadAdvancedMetrics(userId);
      }
    }
  }, [currentUser, isAdminMode, customUserId, user]);

  const handleRiskChange = (riskTolerance: string) => {
    console.log('Risk tolerance changed to:', riskTolerance);
  };

  const handleGoalChange = (newGoals: any[]) => {
    setGoals(newGoals);
  };

  console.log('Dashboard state check:', { authLoading, loading, initializing, currentUser: !!currentUser });
  
  if (authLoading || loading || initializing) {
    console.log('Dashboard showing loading screen');
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary mx-auto mb-4"></div>
          <h2 className="text-2xl font-semibold text-card-foreground">Loading your dashboard...</h2>
          <p className="text-muted-foreground mt-2">Fetching your financial data</p>
        </div>
      </div>
    );
  }

  if (!currentUser) {
    console.log('Dashboard showing no currentUser screen');
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-semibold text-card-foreground mb-4">Unable to load user data</h2>
          <p className="text-muted-foreground mb-4">Please complete your profile setup</p>
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              <p className="text-sm">Error: {error}</p>
            </div>
          )}
          <div className="space-x-4">
            <Button onClick={loadUserData} className="btn-action">
              Try Again
            </Button>
            <Button onClick={handleSignOut} variant="outline">
              Sign Out
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Generate allocation data based on user's investment types
  const allocationData = [
    { name: "Australian Shares", current: 35, recommended: 30 },
    { name: "International Shares", current: 25, recommended: 25 },
    { name: "Property/REITs", current: 20, recommended: 20 },
    { name: "Fixed Income", current: 15, recommended: 20 },
    { name: "Cash", current: 5, recommended: 5 }
  ];

  // Generate growth projection data
  const currentSavings = currentUser?.Current_Savings || 0;
  const totalContribution = currentUser?.Total_Annual_Contribution || 0;
  
  const growthData = [
    { year: 2024, balance: currentSavings, contributions: totalContribution, milestone: "Current Position" },
    { year: 2025, balance: Math.round(currentSavings * 1.08), contributions: totalContribution },
    { year: 2026, balance: Math.round(currentSavings * 1.16), contributions: totalContribution },
    { year: 2027, balance: Math.round(currentSavings * 1.25), contributions: totalContribution, milestone: "Major Market Growth" },
    { year: 2028, balance: Math.round(currentSavings * 1.35), contributions: totalContribution },
    { year: 2029, balance: Math.round(currentSavings * 1.46), contributions: totalContribution },
    { year: 2030, balance: Math.round(currentSavings * 1.58), contributions: totalContribution, milestone: "Halfway Point" },
    { year: 2035, balance: Math.round(currentSavings * 2.16), contributions: totalContribution },
    { year: 2040, balance: Math.round(currentSavings * 2.95), contributions: totalContribution },
    { year: 2045, balance: Math.round(currentSavings * 4.04), contributions: totalContribution },
    { year: 2050, balance: Math.round(currentSavings * 5.52), contributions: totalContribution, milestone: "Retirement Goal" }
  ];

  // Calculate goal progress
  const projectedPension = currentUser?.Projected_Pension_Amount || currentSavings * 5;
  const goalProgress = {
    current: currentSavings,
    target: projection?.adjusted_projection || projectedPension,
    percentage: Math.min(100, (currentSavings / (projection?.adjusted_projection || projectedPension)) * 100)
  };

  // Handle calculator tab - navigate to retirement calculator page
  const handleTabChange = (tab: string) => {
    if (tab === "calculator") {
      navigate(`/retirement-calculator/${currentUser?.User_ID}`);
      return;
    }
    setActiveTab(tab);
  };

  // Render the appropriate page based on active tab
  const renderPage = () => {
    switch (activeTab) {
      case "dashboard":
        return (
          <div className="space-y-8">
            <SummaryCard 
              user={currentUser} 
              projection={projection} 
              peerComparison={peerComparison}
              summaryStats={summaryStats}
            />
            <MetricsGrid 
              user={currentUser} 
              summaryStats={summaryStats}
              advancedMetrics={advancedMetrics}
            />
          </div>
        );
      case "portfolio":
        return (
          <PortfolioPage 
            user={currentUser} 
            allocationData={allocationData} 
            growthData={growthData}
          />
        );
      case "goals":
        return (
          <GoalsPage 
            user={currentUser} 
            onGoalChange={handleGoalChange}
          />
        );
      case "education":
        return <EducationPage user={currentUser} />;
      case "risk":
        return (
          <RiskPage 
            user={currentUser} 
            onRiskChange={handleRiskChange}
          />
        );
      case "chatbot":
        return <ChatbotPageWithSpeech user={currentUser} />;
      default:
        return (
          <div className="space-y-8">
            <SummaryCard 
              user={currentUser} 
              projection={projection} 
              peerComparison={peerComparison}
            />
            <MetricsGrid 
              user={currentUser} 
              summaryStats={summaryStats}
              advancedMetrics={advancedMetrics}
            />
          </div>
        );
    }
  };

  const NewsUpdateSection = () => {
    const [isSendingEmail, setIsSendingEmail] = useState(false);
    const [newsSource, setNewsSource] = useState("newsapi");
    const { toast } = useToast();

    const handleSendEmail = async () => {
      setIsSendingEmail(true);
      try {
        const response = await fetch(`http://localhost:8000/trigger-email/${newsSource}`, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
        });
        
        const result = await response.json();
        
        if (result.success) {
          const sourceName = newsSource === "gemini" ? "Gemini AI" : "NewsAPI";
          toast({
            title: "Email Sent!",
            description: `Financial update email sent successfully using ${sourceName}.`,
          });
        } else {
          toast({
            title: "Email Failed",
            description: "Failed to send email. Please try again.",
            variant: "destructive",
          });
        }
      } catch (error) {
        console.error('Error sending email:', error);
        toast({
          title: "Error",
          description: "Failed to send email. Please check your connection.",
          variant: "destructive",
        });
      } finally {
        setIsSendingEmail(false);
      }
    };

    return (
      <div className="mt-4 space-y-3">
        <div className="text-sm font-medium text-black-700">News Source:</div>
        <Select value={newsSource} onValueChange={setNewsSource}>
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Select news source" />
          </SelectTrigger>
          <SelectContent className="bg-white border border-black-300">
            <SelectItem value="newsapi">NewsAPI (Real-time)</SelectItem>
            <SelectItem value="gemini">Gemini AI (Generated)</SelectItem>
          </SelectContent>
        </Select>
        
        <Button
          onClick={handleSendEmail}
          disabled={isSendingEmail}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white"
        >
          {isSendingEmail ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Sending...
            </>
          ) : (
            <>
              <Mail className="w-4 h-4 mr-2" />
              Get Financial Update
            </>
          )}
        </Button>
      </div>
    );
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low': return 'bg-green-100 text-green-800 border-green-200';
      case 'Medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'High': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

return (
  <div className="min-h-screen bg-background flex">
    {/* Left Sidebar - 1/3 width */}
    <div className="w-1/3 bg-white border-r border-black-200 p-6 flex flex-col">
      {/* User Profile Section */}
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-6">
          <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center">
            <User className="w-8 h-8 text-blue-600" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-black-900">
              Welcome back, {currentUser?.name || `User ${currentUser?.userId}`}!
            </h2>
          </div>
        </div>

        {/* User Stats */}
        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <p className="text-black-600">Age</p>
              <p className="font-semibold">{currentUser?.age || 'N/A'}</p>
            </div>
            <div>
              <p className="text-black-600">Risk</p>
              <span className={`inline-block px-2 py-1 rounded text-xs font-semibold ${getRiskColor(currentUser?.risk_tolerance || 'Medium')}`}>
                {currentUser?.risk_tolerance || 'Medium'}
              </span>
            </div>
            <div>
              <p className="text-black-600">Status</p>
              <p className="font-semibold">{currentUser?.marital_status || 'Single'}</p>
            </div>
            <div>
              <p className="text-black-600">Dependents</p>
              <p className="font-semibold">{currentUser?.number_of_dependents || 0}</p>
            </div>
          </div>
        </div>

        {/* Goal Progress - Full Width */}
        <div className="mb-6">
          <div className="mb-4">
            <div className="flex items-center justify-between mb-3">
              <p className="text-sm font-medium text-black-700">Retirement Goal Progress</p>
              <span className="text-2xl font-bold text-green-600">{Math.round(goalProgress.percentage)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-4 mb-3">
              <div 
                className="bg-gradient-to-r from-green-500 to-green-600 h-4 rounded-full transition-all duration-500 flex items-center justify-end pr-2"
                style={{ width: `${Math.min(100, goalProgress.percentage)}%` }}
              >
                {goalProgress.percentage > 10 && (
                  <span className="text-xs text-white font-semibold">{Math.round(goalProgress.percentage)}%</span>
                )}
              </div>
            </div>
            <div className="bg-green-50 border border-green-200 rounded-lg p-3 text-center">
              <p className="text-base font-semibold text-green-700 mb-1">
                🎉 You're on track! Keep it up!
              </p>
              <p className="text-xs text-green-600">
                Your consistent contributions are building your future
              </p>
            </div>
          </div>
          
          {/* News Source Selection and Email Button */}
          <NewsUpdateSection />
        </div>
      </div>

      {/* Navigation Menu */}
      <nav className="flex-1">
        <div className="space-y-2">
          {[
            { id: 'chatbot', label: 'Chatbot', icon: '💬' },
            { id: 'dashboard', label: 'Dashboard', icon: '📊' },
            { id: 'portfolio', label: 'Portfolio', icon: '📈' },
            { id: 'goals', label: 'Goals', icon: '🎯' },
            { id: 'calculator', label: 'Calculator', icon: '🧮' },
            { id: 'education', label: 'Guide', icon: '📚' },
            { id: 'risk', label: 'Risk', icon: '⚖️' }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-left transition-colors ${
                activeTab === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'text-black-700 hover:bg-black-100'
              }`}
            >
              <span className="text-lg">{tab.icon}</span>
              <span className="font-medium">{tab.label}</span>
            </button>
          ))}
        </div>
      </nav>
      
    </div>

    {/* Right Content Area - 2/3 width */}
    <div className="flex-1 p-8 relative">
      {/* Top Bar with User ID and Admin Controls */}
      <div className="absolute top-4 left-4 right-4 flex items-center justify-between z-10">
          {/* SuperWise Logo */}
          <div className="flex items-center">
            <Shield className="h-7 w-7 text-blue-600" />
            <span className="ml-2 text-lg font-bold text-gray-900">SuperWise</span>
          </div>
        {/* User ID and Admin Mode Badge */}
        <div className="flex items-center gap-4">
          <p className="text-sm text-black-500">
            ID: {currentUser?.User_ID}
          </p>
          {isAdminMode && (
            <span className="inline-block text-xs bg-orange-100 text-orange-800 px-2 py-1 rounded-full">
              Admin Mode
            </span>
          )}
        </div>
        
        {/* Admin Controls */}
          {isAdminMode ? (
            <>
              <Button 
                onClick={() => navigate('/user-manager')}
                variant="outline"
                className="flex items-center gap-2"
              >
                <ArrowLeft className="w-4 h-4" />
                Back to User Manager
              </Button>
              <Button 
                onClick={handleSignOut}
                variant="outline"
                className="flex items-center gap-2"
              >
                <LogOut className="w-4 h-4" />
                Exit Admin Mode
              </Button>
            </>
          ) : (
            <Button 
              onClick={handleSignOut}
              variant="outline"
              className="flex items-center gap-2"
            >
              <LogOut className="w-4 h-4" />
              Sign Out
            </Button>
          )}
      </div>
      
      {/* Main Content */}
      <div className={isAdminMode ? "mt-16" : "mt-16"}>
        {renderPage()}
      </div>
      
      {/* Floating Chat Button */}
      <FloatingChatButton user={currentUser} />
    </div>
  </div>
);
}