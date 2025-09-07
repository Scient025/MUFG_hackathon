import { useState } from "react";
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
import { dataService, sampleUsers, UserProfile } from "@/services/dataService";


export default function Dashboard() {
  const [selectedUserId, setSelectedUserId] = useState("USER001");
  const [activeTab, setActiveTab] = useState("dashboard");
  const [goals, setGoals] = useState<any[]>([]);

  // Get current user data
  const currentUser = dataService.getUserById(selectedUserId) || sampleUsers[0];
  const projection = dataService.calculateRetirementProjection(currentUser);
  const peerComparison = dataService.getPeerComparison(currentUser);

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
    { year: 2024, balance: currentUser.currentSavings, contributions: 23400, milestone: "Current Position" },
    { year: 2025, balance: Math.round(currentUser.currentSavings * 1.08), contributions: 24000 },
    { year: 2026, balance: Math.round(currentUser.currentSavings * 1.16), contributions: 24600 },
    { year: 2027, balance: Math.round(currentUser.currentSavings * 1.25), contributions: 25200, milestone: "Major Market Growth" },
    { year: 2028, balance: Math.round(currentUser.currentSavings * 1.34), contributions: 25800 },
    { year: 2029, balance: Math.round(currentUser.currentSavings * 1.44), contributions: 26400, milestone: "Retirement Age" },
    { year: 2030, balance: currentUser.projectedPensionAmount, contributions: 0 }
  ];

  const handleUserChange = (userId: string) => {
    setSelectedUserId(userId);
  };

  const handleRiskChange = (riskTolerance: string) => {
    // In a real app, this would update the user's risk profile
    console.log('Risk tolerance changed to:', riskTolerance);
  };

  const handleGoalChange = (newGoals: any[]) => {
    setGoals(newGoals);
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* User Selection */}
        <UserSelection 
          users={sampleUsers}
          selectedUserId={selectedUserId}
          onUserChange={handleUserChange}
        />
        
        {/* Header */}
        <DashboardHeader 
          user={currentUser} 
          goalProgress={projection.percentToGoal}
        />
        
        {/* Summary Card */}
        <SummaryCard 
          projection={projection}
          peerComparison={peerComparison}
        />
        
        {/* Key Metrics */}
        <MetricsGrid metrics={{
          currentBalance: currentUser.currentSavings,
          percentToGoal: projection.percentToGoal,
          monthlyIncomeAt65: projection.monthlyIncomeAt65,
          employerContribution: 8500, // Mock data
          totalAnnualContribution: 23400 // Mock data
        }} />
        
        {/* Navigation Tabs */}
        <NavigationTabs activeTab={activeTab} onTabChange={setActiveTab} />
        
        {/* Tab Content */}
        <div className="mb-8">
          {activeTab === "dashboard" && (
            <div className="grid lg:grid-cols-2 gap-8">
              <PortfolioPage 
                user={currentUser}
                allocationData={allocationData}
                growthData={growthData}
              />
            </div>
          )}
          
          {activeTab === "portfolio" && (
            <PortfolioPage 
              user={currentUser}
              allocationData={allocationData}
              growthData={growthData}
            />
          )}
          
          {activeTab === "goals" && (
            <GoalsPage 
              user={currentUser}
              onGoalChange={handleGoalChange}
            />
          )}
          
          {activeTab === "education" && (
            <EducationPage user={currentUser} />
          )}
          
          {activeTab === "risk" && (
            <RiskPage 
              user={currentUser}
              onRiskChange={handleRiskChange}
            />
          )}
          
          {activeTab === "chatbot" && (
            <ChatbotPage user={currentUser} />
          )}
        </div>
        
        {/* Floating Chat Button */}
        <FloatingChatButton />
      </div>
    </div>
  );
}