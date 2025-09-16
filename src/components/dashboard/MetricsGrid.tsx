import { Wallet, Target, TrendingUp, Building, PiggyBank, Heart, AlertTriangle, Shield } from "lucide-react";

interface MetricsGridProps {
  user: any;
  summaryStats: any;
  advancedMetrics?: any;
}

export function MetricsGrid({ user, summaryStats, advancedMetrics }: MetricsGridProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-AU', {
      style: 'currency',
      currency: 'AUD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  // Safe data extraction with fallbacks
  const currentBalance = user?.Current_Savings || 0;
  const percentToGoal = summaryStats?.percent_to_goal || 0;
  const monthlyIncomeAt65 = summaryStats?.monthly_income_at_retirement || 0;
  const employerContribution = user?.Employer_Contribution || 0;
  const totalAnnualContribution = user?.Total_Annual_Contribution || 0;
  
  // Advanced metrics
  const financialHealthScore = advancedMetrics?.financial_health_score || 0;
  const churnRiskPercentage = advancedMetrics?.churn_risk_percentage || 0;
  const anomalyScore = advancedMetrics?.anomaly_score || 0;

  const metricCards = [
    {
      title: "Current Balance",
      value: formatCurrency(currentBalance),
      icon: Wallet,
      status: "good",
      subtitle: "Total superannuation"
    },
    {
      title: "% to Goal",
      value: `${percentToGoal.toFixed(2)}%`,
      icon: Target,
      status: percentToGoal >= 75 ? "good" : percentToGoal >= 50 ? "warning" : "risk",
      subtitle: "On track to retirement"
    },
    {
      title: "Financial Health Score",
      value: `${financialHealthScore.toFixed(0)}/100`,
      icon: Heart,
      status: financialHealthScore >= 80 ? "good" : financialHealthScore >= 60 ? "warning" : "risk",
      subtitle: "Overall financial wellness"
    },
    {
      title: "Churn Risk",
      value: `${churnRiskPercentage.toFixed(1)}%`,
      icon: AlertTriangle,
      status: churnRiskPercentage < 30 ? "good" : churnRiskPercentage < 60 ? "warning" : "risk",
      subtitle: "Risk of stopping contributions"
    },
    {
      title: "Anomaly Score",
      value: `${anomalyScore.toFixed(1)}%`,
      icon: Shield,
      status: anomalyScore < 30 ? "good" : anomalyScore < 70 ? "warning" : "risk",
      subtitle: "Account activity monitoring"
    },
    {
      title: "Estimated Monthly Income at 65",
      value: formatCurrency(monthlyIncomeAt65),
      icon: TrendingUp,
      status: "good",
      subtitle: "Expected pension"
    },
    {
      title: "Employer Contribution",
      value: formatCurrency(employerContribution),
      icon: Building,
      status: "good",
      subtitle: "This financial year"
    },
    {
      title: "Total Annual Contribution",
      value: formatCurrency(totalAnnualContribution),
      icon: PiggyBank,
      status: "good",
      subtitle: "Personal + employer"
    }
  ];

  const getStatusClass = (status: string) => {
    switch (status) {
      case "good": return "bg-green-100 text-green-800";
      case "warning": return "bg-yellow-100 text-yellow-800";
      case "risk": return "bg-red-100 text-red-800";
      default: return "bg-gray-100 text-gray-800";
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 mb-8">
      {metricCards.map((metric) => {
        const Icon = metric.icon;
        const statusClass = getStatusClass(metric.status);
          
        return (
          <div key={metric.title} className="metric-card">
            <div className={`inline-flex items-center justify-center w-14 h-14 rounded-xl mb-4 ${statusClass}`}>
              <Icon className="w-7 h-7" />
            </div>
            <h3 className="text-xl font-semibold text-card-foreground mb-3">
              {metric.title}
            </h3>
            <div className="text-3xl lg:text-4xl font-bold text-card-foreground mb-3">
              {metric.value}
            </div>
            <div className="text-muted-foreground text-lg">
              {metric.subtitle}
            </div>
          </div>
        );
      })}
    </div>
  );
}