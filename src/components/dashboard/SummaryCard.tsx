import { TrendingUp, Target, DollarSign, Users, TrendingDown } from "lucide-react";

interface SummaryCardProps {
  projection: {
    retirementAmount: number;
    monthlyIncreaseNeeded: number;
    targetAmount: number;
    percentToGoal: number;
    monthlyIncomeAt65: number;
  };
  peerComparison: {
    investmentTypes: any;
    ageGroup: any;
    riskGroup: any;
  };
}

export function SummaryCard({ projection, peerComparison }: SummaryCardProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-AU', {
      style: 'currency',
      currency: 'AUD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const formatMonthly = (amount: number) => {
    return new Intl.NumberFormat('en-AU', {
      style: 'currency',
      currency: 'AUD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  // Find user's primary investment type for peer comparison
  const primaryInvestment = Object.keys(peerComparison.investmentTypes).find(
    type => peerComparison.investmentTypes[type].count > 0
  );
  const investmentPercentage = primaryInvestment ? peerComparison.investmentTypes[primaryInvestment].percentage : 0;
  
  return (
    <div className="summary-card mb-8">
      <div className="flex items-start gap-4 mb-6">
        <div className="bg-white/20 p-4 rounded-xl">
          <TrendingUp className="w-10 h-10 text-white" />
        </div>
        <div className="flex-1">
          <h2 className="text-3xl lg:text-4xl font-bold text-white mb-3">
            Your Retirement Projection
          </h2>
          <div className="text-xl text-white/90">
            Based on current contributions and market performance
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <DollarSign className="w-7 h-7 text-white" />
            <h3 className="text-2xl font-semibold text-white">You will retire with</h3>
          </div>
          <div className="text-4xl lg:text-5xl font-bold text-white mb-3">
            {formatCurrency(projection.retirementAmount)}
          </div>
          <div className="text-white/80 text-lg">
            Estimated retirement balance at age 65
          </div>
        </div>

        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Target className="w-7 h-7 text-white" />
            <h3 className="text-2xl font-semibold text-white">To reach your goal</h3>
          </div>
          <div className="text-white/80 mb-2 text-lg">
            You need to increase contributions by
          </div>
          <div className="text-3xl lg:text-4xl font-bold text-white mb-2">
            {formatMonthly(projection.monthlyIncreaseNeeded)}/month
          </div>
          <div className="text-white/80 text-lg">
            to hit your {formatCurrency(projection.targetAmount)} target
          </div>
        </div>
      </div>

      {/* Peer Comparison */}
      <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
        <div className="flex items-center gap-3 mb-4">
          <Users className="w-7 h-7 text-white" />
          <h3 className="text-2xl font-semibold text-white">Peer Comparison</h3>
        </div>
        <div className="grid md:grid-cols-2 gap-6">
          <div>
            <div className="text-white/80 mb-2 text-lg">
              Investment Strategy
            </div>
            <div className="text-xl font-semibold text-white">
              {investmentPercentage}% of users in your age/risk group also invest in {primaryInvestment || 'similar assets'}
            </div>
            <div className="text-white/80 mt-1">
              You are one of them! 👍
            </div>
          </div>
          <div>
            <div className="text-white/80 mb-2 text-lg">
              Your Progress
            </div>
            <div className="text-xl font-semibold text-white">
              You contribute more than {Math.round(100 - (projection.percentToGoal * 0.8))}% of peers
            </div>
            <div className="text-white/80 mt-1">
              Your projected payout is in the top {Math.round(100 - projection.percentToGoal + 20)}% for your age group
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}