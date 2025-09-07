import { Wallet, Target, TrendingUp, Building, PiggyBank } from "lucide-react";

interface MetricsGridProps {
  metrics: {
    currentBalance: number;
    percentToGoal: number;
    monthlyIncomeAt65: number;
    employerContribution: number;
    totalAnnualContribution: number;
  };
}

export function MetricsGrid({ metrics }: MetricsGridProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-AU', {
      style: 'currency',
      currency: 'AUD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const metricCards = [
    {
      title: "Current Balance",
      value: formatCurrency(metrics.currentBalance),
      icon: Wallet,
      status: "good",
      subtitle: "Total superannuation"
    },
    {
      title: "% to Goal",
      value: `${metrics.percentToGoal}%`,
      icon: Target,
      status: metrics.percentToGoal >= 75 ? "good" : metrics.percentToGoal >= 50 ? "warning" : "risk",
      subtitle: "On track to retirement"
    },
    {
      title: "Estimated Monthly Income at 65",
      value: formatCurrency(metrics.monthlyIncomeAt65),
      icon: TrendingUp,
      status: "good",
      subtitle: "Expected pension"
    },
    {
      title: "Employer Contribution",
      value: formatCurrency(metrics.employerContribution),
      icon: Building,
      status: "good",
      subtitle: "This financial year"
    },
    {
      title: "Total Annual Contribution",
      value: formatCurrency(metrics.totalAnnualContribution),
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
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6 mb-8">
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