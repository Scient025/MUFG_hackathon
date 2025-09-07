import { useState } from "react";
import { DashboardHeader } from "@/components/dashboard/DashboardHeader";
import { SummaryCard } from "@/components/dashboard/SummaryCard";
import { MetricsGrid } from "@/components/dashboard/MetricsGrid";
import { AssetAllocation } from "@/components/dashboard/AssetAllocation";
import { PortfolioGrowth } from "@/components/dashboard/PortfolioGrowth";
import { QuickActions } from "@/components/dashboard/QuickActions";
import { NavigationTabs } from "@/components/dashboard/NavigationTabs";
import { FloatingChatButton } from "@/components/dashboard/FloatingChatButton";

// Sample data for the dashboard
const userData = {
  name: "Margaret Smith",
  age: 58,
  avatar: "",
  maritalStatus: "Married",
  dependents: 1,
  riskProfile: "Medium",
  goalProgress: 73
};

const projectionData = {
  retirementAmount: "$847,000",
  monthlyIncrease: "$350",
  targetAmount: "$1,000,000",
  currentAge: 58,
  retirementAge: 65
};

const metricsData = {
  currentBalance: "$548,750",
  percentToGoal: 73,
  monthlyIncomeAt65: "$3,200",
  employerContribution: "$8,500",
  totalAnnualContribution: "$23,400"
};

const allocationData = [
  { name: "Australian Shares", current: 35, recommended: 30, color: "#1e40af" },
  { name: "International Shares", current: 25, recommended: 25, color: "#059669" },
  { name: "Property/REITs", current: 20, recommended: 20, color: "#dc2626" },
  { name: "Fixed Income", current: 15, recommended: 20, color: "#7c3aed" },
  { name: "Cash", current: 5, recommended: 5, color: "#ea580c" }
];

const growthData = [
  { year: 2024, balance: 548750, contributions: 23400, milestone: "Current Position" },
  { year: 2025, balance: 591000, contributions: 24000 },
  { year: 2026, balance: 636200, contributions: 24600 },
  { year: 2027, balance: 684500, contributions: 25200, milestone: "Major Market Growth" },
  { year: 2028, balance: 736100, contributions: 25800 },
  { year: 2029, balance: 791200, contributions: 26400, milestone: "Retirement Age" },
  { year: 2030, balance: 847000, contributions: 0 }
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState("performance");

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <DashboardHeader user={userData} />
        
        {/* Summary Card */}
        <SummaryCard projection={projectionData} />
        
        {/* Key Metrics */}
        <MetricsGrid metrics={metricsData} />
        
        {/* Navigation Tabs */}
        <NavigationTabs activeTab={activeTab} onTabChange={setActiveTab} />
        
        {/* Tab Content */}
        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {activeTab === "performance" && (
            <>
              <PortfolioGrowth 
                data={growthData} 
                currentAge={userData.age}
                retirementAge={projectionData.retirementAge}
              />
              <AssetAllocation allocation={allocationData} />
            </>
          )}
          
          {activeTab === "allocation" && (
            <>
              <AssetAllocation allocation={allocationData} />
              <div className="dashboard-card">
                <h3 className="text-2xl font-semibold text-card-foreground mb-4">
                  Rebalancing Recommendations
                </h3>
                <div className="space-y-4">
                  <div className="p-4 bg-warning/10 border border-warning/20 rounded-xl">
                    <h4 className="font-semibold text-warning mb-2">Action Required</h4>
                    <p className="text-muted-foreground">
                      Consider moving 5% from Australian Shares to Fixed Income to match your risk profile.
                    </p>
                  </div>
                  <div className="p-4 bg-success/10 border border-success/20 rounded-xl">
                    <h4 className="font-semibold text-success mb-2">Well Balanced</h4>
                    <p className="text-muted-foreground">
                      Your international shares and property allocations are optimal.
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
          
          {activeTab === "goals" && (
            <>
              <div className="dashboard-card">
                <h3 className="text-2xl font-semibold text-card-foreground mb-6">
                  Retirement Goals
                </h3>
                <div className="space-y-6">
                  <div>
                    <h4 className="text-xl font-medium mb-4">Primary Goal</h4>
                    <div className="p-6 bg-primary/5 border border-primary/20 rounded-xl">
                      <div className="flex justify-between items-center mb-2">
                        <span className="font-semibold">Retirement by 65</span>
                        <span className="text-success font-bold">73%</span>
                      </div>
                      <div className="w-full bg-muted rounded-full h-3">
                        <div className="bg-success h-3 rounded-full" style={{ width: "73%" }}></div>
                      </div>
                      <p className="text-muted-foreground mt-2">Target: $1,000,000</p>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-xl font-medium mb-4">Secondary Goals</h4>
                    <div className="space-y-3">
                      <div className="p-4 bg-card border border-card-border rounded-xl">
                        <div className="flex justify-between items-center">
                          <span>Emergency Health Fund</span>
                          <span className="text-success">✓ Complete</span>
                        </div>
                      </div>
                      <div className="p-4 bg-card border border-card-border rounded-xl">
                        <div className="flex justify-between items-center">
                          <span>Travel Fund (Europe Trip)</span>
                          <span className="text-warning">45%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <div className="dashboard-card">
                <h3 className="text-2xl font-semibold text-card-foreground mb-6">
                  Scenario Planning
                </h3>
                <div className="space-y-6">
                  <div>
                    <label className="text-lg font-medium mb-3 block">Monthly Contribution</label>
                    <input 
                      type="range" 
                      min="1000" 
                      max="5000" 
                      defaultValue="1950"
                      className="w-full h-3 bg-muted rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-sm text-muted-foreground mt-2">
                      <span>$1,000</span>
                      <span className="font-semibold">$1,950</span>
                      <span>$5,000</span>
                    </div>
                  </div>
                  
                  <div>
                    <label className="text-lg font-medium mb-3 block">Retirement Age</label>
                    <input 
                      type="range" 
                      min="60" 
                      max="70" 
                      defaultValue="65"
                      className="w-full h-3 bg-muted rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-sm text-muted-foreground mt-2">
                      <span>60 years</span>
                      <span className="font-semibold">65 years</span>
                      <span>70 years</span>
                    </div>
                  </div>
                  
                  <div className="p-4 bg-success/10 border border-success/20 rounded-xl">
                    <h4 className="font-semibold text-success mb-2">Impact Projection</h4>
                    <p className="text-muted-foreground">
                      With current settings: $847,000 at retirement
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
          
          {activeTab === "learn" && (
            <>
              <div className="dashboard-card">
                <h3 className="text-2xl font-semibold text-card-foreground mb-6">
                  Educational Resources
                </h3>
                <div className="space-y-4">
                  <div className="p-4 bg-card border border-card-border rounded-xl hover:shadow-md transition-shadow cursor-pointer">
                    <h4 className="font-semibold text-lg mb-2">Understanding Superannuation</h4>
                    <p className="text-muted-foreground mb-3">
                      Learn the basics of how super works in Australia
                    </p>
                    <span className="text-primary font-medium">5 min read →</span>
                  </div>
                  
                  <div className="p-4 bg-card border border-card-border rounded-xl hover:shadow-md transition-shadow cursor-pointer">
                    <h4 className="font-semibold text-lg mb-2">Asset Allocation Strategies</h4>
                    <p className="text-muted-foreground mb-3">
                      How to balance risk and return in your portfolio
                    </p>
                    <span className="text-primary font-medium">8 min read →</span>
                  </div>
                  
                  <div className="p-4 bg-card border border-card-border rounded-xl hover:shadow-md transition-shadow cursor-pointer">
                    <h4 className="font-semibold text-lg mb-2">Retirement Planning Checklist</h4>
                    <p className="text-muted-foreground mb-3">
                      Essential steps to prepare for retirement
                    </p>
                    <span className="text-primary font-medium">12 min read →</span>
                  </div>
                </div>
              </div>
              
              <div className="dashboard-card">
                <h3 className="text-2xl font-semibold text-card-foreground mb-6">
                  Frequently Asked Questions
                </h3>
                <div className="space-y-4">
                  <div className="p-4 bg-muted/50 rounded-xl">
                    <h4 className="font-semibold mb-2">
                      Q: How much should I contribute to super?
                    </h4>
                    <p className="text-muted-foreground">
                      A: Generally, contributing 12-15% of your income (including employer contributions) 
                      is recommended for a comfortable retirement.
                    </p>
                  </div>
                  
                  <div className="p-4 bg-muted/50 rounded-xl">
                    <h4 className="font-semibold mb-2">
                      Q: When can I access my super?
                    </h4>
                    <p className="text-muted-foreground">
                      A: You can generally access your super when you reach your preservation age 
                      (between 55-60 depending on when you were born) and retire.
                    </p>
                  </div>
                  
                  <div className="p-4 bg-muted/50 rounded-xl">
                    <h4 className="font-semibold mb-2">
                      Q: What happens to my super if I die?
                    </h4>
                    <p className="text-muted-foreground">
                      A: Your super will be paid to your nominated beneficiaries or your estate 
                      according to your binding death benefit nomination.
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
        
        {/* Quick Actions */}
        <QuickActions />
        
        {/* Floating Chat Button */}
        <FloatingChatButton />
      </div>
    </div>
  );
}