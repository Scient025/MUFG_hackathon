import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { Button } from "@/components/ui/button";
import { HelpCircle } from "lucide-react";

interface AssetAllocationProps {
  allocation: Array<{
    name: string;
    current: number;
    recommended: number;
    color: string;
  }>;
}

export function AssetAllocation({ allocation }: AssetAllocationProps) {
  const currentData = allocation.map(item => ({
    name: item.name,
    value: item.current,
    color: item.color,
    recommended: item.recommended
  }));

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card border border-card-border rounded-xl p-4 shadow-lg">
          <p className="font-semibold text-lg">{data.name}</p>
          <p className="text-success">Current: {data.value}%</p>
          <p className="text-neutral">Recommended: {data.recommended}%</p>
          {Math.abs(data.value - data.recommended) > 2 && (
            <p className="text-warning text-sm mt-1">
              {data.value > data.recommended ? 'Over-allocated' : 'Under-allocated'}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="dashboard-card">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-2xl font-semibold text-card-foreground">Asset Allocation</h3>
        <Button variant="outline" size="sm" className="flex items-center gap-2">
          <HelpCircle className="w-4 h-4" />
          Explain This
        </Button>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={currentData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={120}
                paddingAngle={2}
                dataKey="value"
              >
                {currentData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="space-y-4">
          <h4 className="text-xl font-semibold text-card-foreground mb-4">
            Current vs Recommended
          </h4>
          {allocation.map((item) => {
            const difference = item.current - item.recommended;
            const isAligned = Math.abs(difference) <= 2;
            
            return (
              <div key={item.name} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-4 h-4 rounded-full" 
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="font-medium text-card-foreground">{item.name}</span>
                  </div>
                  <div className={`text-sm font-medium ${
                    isAligned ? 'text-success' : 'text-warning'
                  }`}>
                    {isAligned ? '✓ Aligned' : 
                     difference > 0 ? `+${difference.toFixed(1)}%` : `${difference.toFixed(1)}%`}
                  </div>
                </div>
                <div className="text-sm text-muted-foreground">
                  Current: {item.current}% | Recommended: {item.recommended}%
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}