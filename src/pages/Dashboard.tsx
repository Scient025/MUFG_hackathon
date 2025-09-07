import { useState, useEffect } from "react";
import { DashboardHeader } from "@/components/dashboard/DashboardHeader";
import { SummaryCard } from "@/components/dashboard/SummaryCard";
import { MetricsGrid } from "@/components/dashboard/MetricsGrid";
import { NavigationTabs } from "@/components/dashboard/NavigationTabs";
import { UserSelection } from "@/components/dashboard/UserSelection";
import { PortfolioPage } from "@/components/dashboard/PortfolioPage";
import { GoalsPage } from "@/components/dashboard/GoalsPage";
import { EducationPage } from "@/components/dashboard/EducationPage";
import { RiskPage } from "@/components/dashboard/RiskPage";
import { ChatbotPage } from "@/components/dashboard/ChatbotPage";
import { FloatingChatButton } from "@/components/dashboard/FloatingChatButton";
import { SignupForm } from "@/components/auth/SignupForm";
import { Button } from "@/components/ui/button";
import { dataService, UserProfile, User } from "@/services/dataService";
import { Plus } from "lucide-react";

export default function Dashboard() {
  const [selectedUserId, setSelectedUserId] = useState<string>("");
  const [activeTab, setActiveTab] = useState("dashboard");
  const [goals, setGoals] = useState<any[]>([]);
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [summaryStats, setSummaryStats] = useState<any>(null);
  const [peerComparison, setPeerComparison] = useState<any>(null);
  const [projection, setProjection] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showSignup, setShowSignup] = useState(false);
  const [availableUsers, setAvailableUsers] = useState<User[]>([]);

  const loadAvailableUsers = async () => {
    try {
      const users = await dataService.getAllUsers();
      setAvailableUsers(users);
      if (users.length > 0 && !selectedUserId) {
        setSelectedUserId(users[0].User_ID);
      }
    } catch (error) {
      console.error('Error loading users:', error);
    }
  };

  const loadUserData = async () => {
    if (!selectedUserId) return;
    
    try {
      setError(null);
      setLoading(true);
      console.log('Loading data for user:', selectedUserId);
      
      const [user, summary, peer, proj] = await Promise.all([
        dataService.getUserById(selectedUserId),
        dataService.getSummaryStats(selectedUserId),
        dataService.getPeerComparison(selectedUserId),
        dataService.getPensionProjection(selectedUserId)
      ]);

      console.log('Data loaded successfully:', { user, summary, peer, proj });

      setCurrentUser(user);
      setSummaryStats(summary || { current_savings: 0, projected_pension: 0, percent_to_goal: 0, monthly_income_at_retirement: 0 });
      setPeerComparison(peer || { total_peers: 0, avg_age: 0, avg_income: 0, avg_savings: 0, avg_contribution: 0 });
      setProjection(proj || { current_projection: 0, adjusted_projection: 0, years_to_retirement: 0, monthly_income_at_retirement: 0 });
    } catch (error) {
      console.error('Error loading user data:', error);
      setError(error instanceof Error ? error.message : 'Unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSignupSuccess = (userId: string) => {
    setSelectedUserId(userId);
    setShowSignup(false);
    loadAvailableUsers(); // Refresh user list
  };

  // Load available users on component mount
  useEffect(() => {
    loadAvailableUsers();
  }, []);

  // Load user data when selectedUserId changes
  useEffect(() => {
    if (selectedUserId) {
      loadUserData();
    }
  }, [selectedUserId]);

  const handleUserChange = (userId: string) => {
    setSelectedUserId(userId);
  };

  const handleRiskChange = (riskTolerance: string) => {
    console.log('Risk tolerance changed to:', riskTolerance);
  };

  const handleGoalChange = (newGoals: any[]) => {
    setGoals(newGoals);
  };

  if (loading) {
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
          <p className="text-muted-foreground mb-4">Please make sure the ML backend is running</p>
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              <p className="text-sm">Error: {error}</p>
            </div>
          )}
          <Button onClick={loadUserData} className="btn-action">
            Try Again
          </Button>
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
  const growthData = [
    { year: 2024, balance: currentUser.Current_Savings, contributions: currentUser.Total_Annual_Contribution, milestone: "Current Position" },
    { year: 2025, balance: Math.round(currentUser.Current_Savings * 1.08), contributions: currentUser.Total_Annual_Contribution },
    { year: 2026, balance: Math.round(currentUser.Current_Savings * 1.16), contributions: currentUser.Total_Annual_Contribution },
    { year: 2027, balance: Math.round(currentUser.Current_Savings * 1.25), contributions: currentUser.Total_Annual_Contribution, milestone: "Major Market Growth" },
    { year: 2028, balance: Math.round(currentUser.Current_Savings * 1.35), contributions: currentUser.Total_Annual_Contribution },
    { year: 2029, balance: Math.round(currentUser.Current_Savings * 1.46), contributions: currentUser.Total_Annual_Contribution },
    { year: 2030, balance: Math.round(currentUser.Current_Savings * 1.58), contributions: currentUser.Total_Annual_Contribution, milestone: "Halfway Point" },
    { year: 2035, balance: Math.round(currentUser.Current_Savings * 2.16), contributions: currentUser.Total_Annual_Contribution },
    { year: 2040, balance: Math.round(currentUser.Current_Savings * 2.95), contributions: currentUser.Total_Annual_Contribution },
    { year: 2045, balance: Math.round(currentUser.Current_Savings * 4.04), contributions: currentUser.Total_Annual_Contribution },
    { year: 2050, balance: Math.round(currentUser.Current_Savings * 5.52), contributions: currentUser.Total_Annual_Contribution, milestone: "Retirement Goal" }
  ];

  // Calculate goal progress
  const goalProgress = {
    current: currentUser.Current_Savings,
    target: projection?.adjusted_projection || currentUser.Projected_Pension_Amount,
    percentage: Math.min(100, (currentUser.Current_Savings / (projection?.adjusted_projection || currentUser.Projected_Pension_Amount)) * 100)
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
            goals={goals} 
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
        return <ChatbotPage user={currentUser} />;
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

  // Show signup form if no users available or signup requested
  if (showSignup || availableUsers.length === 0) {
    return (
      <SignupForm 
        onSignupSuccess={handleSignupSuccess}
        onCancel={() => setShowSignup(false)}
      />
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* User Selection */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">User Profiles</h2>
            <Button 
              onClick={() => setShowSignup(true)}
              className="flex items-center gap-2"
            >
              <Plus className="w-4 h-4" />
              Add New User
            </Button>
          </div>
          <UserSelection 
            selectedUserId={selectedUserId} 
            onUserChange={handleUserChange}
            availableUsers={availableUsers}
          />
        </div>

        {/* Dashboard Header */}
        <DashboardHeader 
          user={currentUser} 
          goalProgress={goalProgress}
        />

        {/* Navigation Tabs */}
        <NavigationTabs 
          activeTab={activeTab} 
          onTabChange={setActiveTab}
        />

        {/* Main Content */}
        <div className="mt-8">
          {renderPage()}
        </div>

        {/* Floating Chat Button */}
        <FloatingChatButton user={currentUser} />
      </div>
    </div>
  );
}