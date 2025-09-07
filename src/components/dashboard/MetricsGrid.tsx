import { Wallet, Target, TrendingUp, Building, PiggyBank } from "lucide-react";

interface MetricsGridProps {
  metrics: {
    currentBalance: string;
    percentToGoal: number;
    monthlyIncomeAt65: string;
    employerContribution: string;
    totalAnnualContribution: string;
  };
}

export function MetricsGrid({ metrics }: MetricsGridProps) {
  const metricCards = [
    {
      title: "Current Balance",
      value: metrics.currentBalance,
      icon: Wallet,
      status: "good",
      subtitle: "Total superannuation"
    },
    {
      title: "Goal Progress",
      value: `${metrics.percentToGoal}%`,
      icon: Target,
      status: metrics.percentToGoal >= 75 ? "good" : metrics.percentToGoal >= 50 ? "warning" : "risk",
      subtitle: "On track to retirement"
    },
    {
      title: "Monthly Income at 65",
      value: metrics.monthlyIncomeAt65,
      icon: TrendingUp,
      status: "good",
      subtitle: "Estimated pension"
    },
    {
      title: "Employer Contribution",
      value: metrics.employerContribution,
      icon: Building,
      status: "good",
      subtitle: "This financial year"
    },
    {
      title: "Total Annual Contribution",
      value: metrics.totalAnnualContribution,
      icon: PiggyBank,
      status: "good",
      subtitle: "Personal + employer"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6 mb-8">
      {metricCards.map((metric) => {
        const Icon = metric.icon;
        const statusClass = 
          metric.status === "good" ? "status-good" :
          metric.status === "warning" ? "status-warning" : "status-risk";
          
        return (
          <div key={metric.title} className="metric-card">
            <div className={`inline-flex items-center justify-center w-12 h-12 rounded-xl mb-4 ${statusClass}`}>
              <Icon className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-semibold text-card-foreground mb-2">
              {metric.title}
            </h3>
            <div className="text-3xl font-bold text-card-foreground mb-2">
              {metric.value}
            </div>
            <div className="text-muted-foreground">
              {metric.subtitle}
            </div>
          </div>
        );
      })}
    </div>
  );
}