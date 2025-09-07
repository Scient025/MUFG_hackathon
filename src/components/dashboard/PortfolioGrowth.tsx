import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { Button } from "@/components/ui/button";
import { HelpCircle, TrendingUp } from "lucide-react";

interface PortfolioGrowthProps {
  data: Array<{
    year: number;
    balance: number;
    contributions: number;
    milestone?: string;
  }>;
  currentAge: number;
  retirementAge: number;
}

export function PortfolioGrowth({ data, currentAge, retirementAge }: PortfolioGrowthProps) {
  const formatCurrency = (value: number) => {
    return `$${(value / 1000).toFixed(0)}k`;
  };

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      const age = currentAge + (data.year - new Date().getFullYear());
      
      return (
        <div className="bg-card border border-card-border rounded-xl p-4 shadow-lg">
          <p className="font-semibold text-lg">Year {data.year} (Age {age})</p>
          <p className="text-success">Balance: ${data.balance.toLocaleString()}</p>
          <p className="text-primary">Contributions: ${data.contributions.toLocaleString()}</p>
          {data.milestone && (
            <p className="text-warning font-medium mt-1">🎯 {data.milestone}</p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="dashboard-card">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="bg-success/10 p-3 rounded-xl">
            <TrendingUp className="w-6 h-6 text-success" />
          </div>
          <div>
            <h3 className="text-2xl font-semibold text-card-foreground">Portfolio Growth</h3>
            <p className="text-muted-foreground">Track your journey to retirement</p>
          </div>
        </div>
        <Button variant="outline" size="sm" className="flex items-center gap-2">
          <HelpCircle className="w-4 h-4" />
          Explain This
        </Button>
      </div>

      <div className="h-80 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" className="opacity-30" />
            <XAxis 
              dataKey="year" 
              className="text-sm"
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              tickFormatter={formatCurrency}
              className="text-sm"
              tick={{ fontSize: 12 }}
            />
            <Tooltip content={<CustomTooltip />} />
            
            <ReferenceLine 
              x={currentAge + (retirementAge - currentAge)} 
              stroke="#ef4444" 
              strokeDasharray="5 5"
              label={{ value: "Retirement", position: "top" }}
            />
            
            <Line 
              type="monotone" 
              dataKey="balance" 
              stroke="hsl(var(--success))" 
              strokeWidth={3}
              dot={{ fill: "hsl(var(--success))", strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6, stroke: "hsl(var(--success))", strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="grid md:grid-cols-3 gap-4">
        <div className="text-center p-4 bg-success/5 rounded-xl">
          <div className="text-2xl font-bold text-success mb-1">
            ${data[data.length - 1]?.balance.toLocaleString()}
          </div>
          <div className="text-sm text-muted-foreground">Projected at retirement</div>
        </div>
        
        <div className="text-center p-4 bg-primary/5 rounded-xl">
          <div className="text-2xl font-bold text-primary mb-1">
            {retirementAge - currentAge} years
          </div>
          <div className="text-sm text-muted-foreground">Until retirement</div>
        </div>
        
        <div className="text-center p-4 bg-warning/5 rounded-xl">
          <div className="text-2xl font-bold text-warning mb-1">
            7.2%
          </div>
          <div className="text-sm text-muted-foreground">Average annual growth</div>
        </div>
      </div>
    </div>
  );
}