import { TrendingUp, Target, DollarSign } from "lucide-react";

interface SummaryCardProps {
  projection: {
    retirementAmount: string;
    monthlyIncrease: string;
    targetAmount: string;
    currentAge: number;
    retirementAge: number;
  };
}

export function SummaryCard({ projection }: SummaryCardProps) {
  const yearsToRetirement = projection.retirementAge - projection.currentAge;
  
  return (
    <div className="summary-card mb-8">
      <div className="flex items-start gap-4 mb-6">
        <div className="bg-white/20 p-3 rounded-xl">
          <TrendingUp className="w-8 h-8 text-white" />
        </div>
        <div className="flex-1">
          <h2 className="text-2xl md:text-3xl font-bold text-white mb-2">
            Your Retirement Projection
          </h2>
          <div className="text-lg text-white/90">
            Based on current contributions and market performance
          </div>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <DollarSign className="w-6 h-6 text-white" />
            <h3 className="text-xl font-semibold text-white">Current Path</h3>
          </div>
          <div className="text-3xl font-bold text-white mb-2">
            {projection.retirementAmount}
          </div>
          <div className="text-white/80">
            Estimated retirement balance in {yearsToRetirement} years
          </div>
        </div>

        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Target className="w-6 h-6 text-white" />
            <h3 className="text-xl font-semibold text-white">Reach Your Goal</h3>
          </div>
          <div className="text-white/80 mb-2">
            Increase contributions by
          </div>
          <div className="text-2xl font-bold text-white mb-1">
            {projection.monthlyIncrease}/month
          </div>
          <div className="text-white/80">
            to hit your {projection.targetAmount} target
          </div>
        </div>
      </div>
    </div>
  );
}