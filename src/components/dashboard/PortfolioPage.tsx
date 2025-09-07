import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PieChart, Pie, Cell, ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts";
import { TrendingUp, TrendingDown, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip as UITooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface PortfolioPageProps {
  user: any;
  allocationData: any[];
  growthData: any[];
}

export function PortfolioPage({ user, allocationData, growthData }: PortfolioPageProps) {
  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-AU', {
      style: 'currency',
      currency: 'AUD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const COLORS = ['#1e40af', '#059669', '#dc2626', '#7c3aed', '#ea580c'];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-4 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-semibold text-lg">{`${label}`}</p>
          <p className="text-blue-600 font-medium">
            {`Balance: ${formatCurrency(payload[0].value)}`}
          </p>
          {payload[1] && (
            <p className="text-green-600 font-medium">
              {`Contributions: ${formatCurrency(payload[1].value)}`}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-8">
      {/* Asset Allocation Cards */}
      <div className="grid lg:grid-cols-2 gap-8">
        <Card className="dashboard-card">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-card-foreground flex items-center gap-3">
              Asset Allocation
              <UITooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="sm" className="p-1">
                    <Info className="w-5 h-5 text-muted-foreground" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-sm">Your current investment distribution across different asset classes</p>
                </TooltipContent>
              </UITooltip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {allocationData.map((asset, index) => (
                <div key={asset.name} className="p-4 bg-muted/30 rounded-xl">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-lg">{asset.name}</span>
                    <span className="text-lg font-bold">{asset.current}%</span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-3 mb-2">
                    <div 
                      className="h-3 rounded-full" 
                      style={{ 
                        width: `${asset.current}%`, 
                        backgroundColor: COLORS[index % COLORS.length] 
                      }}
                    ></div>
                  </div>
                  <div className="flex justify-between text-sm text-muted-foreground">
                    <span>Current: {asset.current}%</span>
                    <span>Recommended: {asset.recommended}%</span>
                  </div>
                  {asset.current !== asset.recommended && (
                    <div className="mt-2 p-2 bg-warning/10 border border-warning/20 rounded-lg">
                      <p className="text-warning text-sm font-medium">
                        {asset.current > asset.recommended ? 'Consider reducing' : 'Consider increasing'} by {Math.abs(asset.current - asset.recommended)}%
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="dashboard-card">
          <CardHeader>
            <CardTitle className="text-2xl font-bold text-card-foreground flex items-center gap-3">
              Allocation Overview
              <UITooltip>
                <TooltipTrigger asChild>
                  <Button variant="ghost" size="sm" className="p-1">
                    <Info className="w-5 h-5 text-muted-foreground" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p className="text-sm">Visual representation of your portfolio allocation</p>
                </TooltipContent>
              </UITooltip>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={allocationData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={120}
                    paddingAngle={5}
                    dataKey="current"
                  >
                    {allocationData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip 
                    formatter={(value: any) => [`${value}%`, 'Allocation']}
                    contentStyle={{
                      backgroundColor: 'white',
                      border: '1px solid #ccc',
                      borderRadius: '8px',
                      fontSize: '14px'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-4">
              {allocationData.map((asset, index) => (
                <div key={asset.name} className="flex items-center gap-2">
                  <div 
                    className="w-4 h-4 rounded" 
                    style={{ backgroundColor: COLORS[index % COLORS.length] }}
                  ></div>
                  <span className="text-sm font-medium">{asset.name}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Investment Growth Projection */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground flex items-center gap-3">
            Investment Growth Projection
            <UITooltip>
              <TooltipTrigger asChild>
                <Button variant="ghost" size="sm" className="p-1">
                  <Info className="w-5 h-5 text-muted-foreground" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p className="text-sm">Projected growth of your investments with contributions and market performance</p>
              </TooltipContent>
            </UITooltip>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={growthData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="year" 
                  fontSize={14}
                  tick={{ fontSize: 14 }}
                />
                <YAxis 
                  tickFormatter={(value) => formatCurrency(value)}
                  fontSize={14}
                  tick={{ fontSize: 14 }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="balance" 
                  stroke="#1e40af" 
                  strokeWidth={3}
                  name="Portfolio Balance"
                />
                <Line 
                  type="monotone" 
                  dataKey="contributions" 
                  stroke="#059669" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Annual Contributions"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          
          {/* Milestones */}
          <div className="mt-6 space-y-3">
            <h4 className="text-lg font-semibold text-card-foreground">Key Milestones</h4>
            {growthData
              .filter(item => item.milestone)
              .map((item, index) => (
                <div key={index} className="flex items-center gap-3 p-3 bg-muted/30 rounded-lg">
                  <div className="w-3 h-3 bg-primary rounded-full"></div>
                  <div>
                    <span className="font-medium">{item.milestone}</span>
                    <span className="text-muted-foreground ml-2">
                      {item.year} - {formatCurrency(item.balance)}
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </CardContent>
      </Card>

      {/* Asset Type Breakdown */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground">
            Your Investment Types
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {user.investmentType.map((type: string, index: number) => (
              <div key={index} className="p-4 bg-card border border-card-border rounded-xl">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                    <TrendingUp className="w-5 h-5 text-primary" />
                  </div>
                  <h4 className="font-semibold text-lg">{type}</h4>
                </div>
                <p className="text-muted-foreground text-sm">
                  {user.fundName[index] || 'No specific fund'}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
