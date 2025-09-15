import { useState, useEffect } from "react";
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
import { UserProfile } from "@/lib/supabase";
import { LogOut, User, ArrowLeft } from "lucide-react";
import { AdminDebug } from "@/components/debug/AdminDebug";

export default function Dashboard() {
  const { user, loading: authLoading, signOut } = useAuth();
  const { adminUser, isAdminMode, logout: adminLogout } = useAdminAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [activeTab, setActiveTab] = useState("chatbot");
  const [goals, setGoals] = useState<any[]>([]);
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [summaryStats, setSummaryStats] = useState<any>(null);
  const [peerComparison, setPeerComparison] = useState<any>(null);
  const [projection, setProjection] = useState<any>(null);
  const [customUserId, setCustomUserId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [initializing, setInitializing] = useState(true);

  // Handle custom login (from our custom login system)
  useEffect(() => {
    // Check for userId from navigation state or localStorage
    const userIdFromState = location.state?.userId;
    const currentUserFromStorage = localStorage.getItem('currentUser');
    
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
    } else {
      console.log('No custom userId found');
    }
    
    // Mark initialization as complete
    setInitializing(false);
  }, [location.state]);

  const loadUserData = async () => {
    // Use custom userId if available, otherwise fall back to admin mode
    const userIdToUse = customUserId || (isAdminMode ? adminUser?.id : null);
    
    if (!userIdToUse) {
      console.log('No userId available for loading data');
      return;
    }

    console.log('Loading user data for userId:', userIdToUse);
    
    // In admin mode, use the admin user directly
    if (isAdminMode && adminUser) {
      setCurrentUser(adminUser);
      setSummaryStats({
        current_savings: adminUser.current_savings,
        projected_pension: adminUser.current_savings * 5, // Mock calculation
        percent_to_goal: 20,
        monthly_income_at_retirement: adminUser.current_savings * 0.04 / 12
      });
      setPeerComparison({
        total_peers: 100,
        avg_age: 35,
        avg_income: 75000,
        avg_savings: 25000,
        avg_contribution: 1200
      });
      setProjection({
        current_projection: adminUser.current_savings * 5,
        optimistic_projection: adminUser.current_savings * 7,
        pessimistic_projection: adminUser.current_savings * 3
      });
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
          setSummaryStats({
            current_savings: userProfile.Current_Savings,
            projected_pension: userProfile.Current_Savings * 5, // Mock calculation
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
            optimistic_projection: userProfile.Current_Savings * 7,
            pessimistic_projection: userProfile.Current_Savings * 3
          });
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
          current_savings: userProfile.current_savings,
          projected_pension: userProfile.current_savings * 5, // Mock calculation
          percent_to_goal: 20,
          monthly_income_at_retirement: userProfile.current_savings * 0.04 / 12
        });
        setPeerComparison({
          total_peers: 100,
          avg_age: 35,
          avg_income: 75000,
          avg_savings: 25000,
          avg_contribution: 1200
        });
        setProjection({
          current_projection: userProfile.current_savings * 5,
          adjusted_projection: userProfile.current_savings * 6,
          years_to_retirement: userProfile.retirement_age_goal - userProfile.age,
          monthly_income_at_retirement: userProfile.current_savings * 0.04 / 12
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
      setCustomUserId(null);
      navigate('/login');
    } else {
      await signOut();
      navigate('/login');
    }
  };

  // Redirect to login if not authenticated (unless in admin mode)
  useEffect(() => {
    console.log('Auth check:', { authLoading, user: !!user, isAdminMode, customUserId, initializing });
    
    // Only redirect if we're sure there's no authentication method available and initialization is complete
    if (!authLoading && !user && !isAdminMode && !customUserId && !initializing) {
      console.log('No authentication found, redirecting to login');
      navigate('/login');
    }
  }, [user, authLoading, isAdminMode, customUserId, initializing, navigate]);

  // Load user data when user is authenticated or in admin mode
  useEffect(() => {
    if (user || isAdminMode || customUserId) {
      loadUserData();
    }
  }, [user, isAdminMode, customUserId]);

  const handleRiskChange = (riskTolerance: string) => {
    console.log('Risk tolerance changed to:', riskTolerance);
  };

  const handleGoalChange = (newGoals: any[]) => {
    setGoals(newGoals);
  };

  if (authLoading || loading || initializing) {
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
  const currentSavings = currentUser.Current_Savings || currentUser.current_savings || 0;
  const totalContribution = currentUser.Total_Annual_Contribution || currentUser.total_annual_contribution || 0;
  
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
  const projectedPension = currentUser.Projected_Pension_Amount || currentUser.projected_pension_amount || currentSavings * 5;
  const goalProgress = {
    current: currentSavings,
    target: projection?.adjusted_projection || projectedPension,
    percentage: Math.min(100, (currentSavings / (projection?.adjusted_projection || projectedPension)) * 100)
  };

  // Handle calculator tab - navigate to retirement calculator page
  const handleTabChange = (tab: string) => {
    if (tab === "calculator") {
      navigate(`/retirement-calculator/${currentUser?.id}`);
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
            />
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* User Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-4">
              <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center">
                <User className="w-6 h-6 text-primary-foreground" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">
                  {currentUser.name || `User ${currentUser.id}`}
                  {isAdminMode && (
                    <span className="ml-2 text-sm bg-orange-100 text-orange-800 px-2 py-1 rounded-full">
                      Admin Mode
                    </span>
                  )}
                </h2>
                <p className="text-muted-foreground">
                  {currentUser.email || `ID: ${currentUser.id}`}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {isAdminMode && (
                <Button 
                  onClick={() => navigate('/user-manager')}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back to User Manager
                </Button>
              )}
              <Button 
                onClick={handleSignOut}
                variant="outline"
                className="flex items-center gap-2"
              >
                <LogOut className="w-4 h-4" />
                {isAdminMode ? 'Exit Admin Mode' : 'Sign Out'}
              </Button>
            </div>
          </div>
        </div>

        {/* Dashboard Header */}
        <DashboardHeader 
          user={currentUser} 
          goalProgress={goalProgress}
        />

        {/* Navigation Tabs */}
        <NavigationTabs 
          activeTab={activeTab} 
          onTabChange={handleTabChange}
        />

        {/* Main Content */}
        <div className="mt-8">
          {renderPage()}
        </div>

        {/* Floating Chat Button */}
        <FloatingChatButton user={currentUser} />
        <AdminDebug />
      </div>
    </div>
  );
}