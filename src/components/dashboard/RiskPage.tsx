import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Shield, TrendingUp, TrendingDown, AlertTriangle, CheckCircle } from "lucide-react";
import { useState } from "react";

interface RiskPageProps {
  user: any;
  onRiskChange: (riskTolerance: string) => void;
}

export function RiskPage({ user, onRiskChange }: RiskPageProps) {
  const [riskTolerance, setRiskTolerance] = useState(user.Risk_Tolerance || 'Medium');
  const [allocationInputs, setAllocationInputs] = useState({
    shares: 60,
    realEstate: 20,
    fixedIncome: 15,
    cash: 5
  });

  const riskLevels = {
    Low: {
      color: 'bg-green-100 text-green-800 border-green-200',
      icon: Shield,
      description: 'Conservative approach with focus on capital preservation',
      recommendedAllocation: {
        shares: 30,
        realEstate: 15,
        fixedIncome: 40,
        cash: 15
      }
    },
    Medium: {
      color: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      icon: TrendingUp,
      description: 'Balanced approach with moderate growth potential',
      recommendedAllocation: {
        shares: 50,
        realEstate: 20,
        fixedIncome: 25,
        cash: 5
      }
    },
    High: {
      color: 'bg-red-100 text-red-800 border-red-200',
      icon: TrendingUp,
      description: 'Growth-focused approach with higher volatility',
      recommendedAllocation: {
        shares: 70,
        realEstate: 15,
        fixedIncome: 10,
        cash: 5
      }
    }
  };

  const currentRisk = riskLevels[riskTolerance as keyof typeof riskLevels];
  const recommendedAllocation = currentRisk.recommendedAllocation;

  const handleRiskChange = (newRisk: string) => {
    setRiskTolerance(newRisk);
    onRiskChange(newRisk);
  };

  const handleAllocationChange = (category: string, value: number[]) => {
    setAllocationInputs(prev => ({
      ...prev,
      [category]: value[0]
    }));
  };

  const getRiskIcon = (level: string) => {
    const Icon = riskLevels[level as keyof typeof riskLevels].icon;
    return <Icon className="w-5 h-5" />;
  };

  const getRiskColor = (level: string) => {
    return riskLevels[level as keyof typeof riskLevels].color;
  };

  const calculateDeviation = () => {
    const deviations = {
      shares: Math.abs(allocationInputs.shares - recommendedAllocation.shares),
      realEstate: Math.abs(allocationInputs.realEstate - recommendedAllocation.realEstate),
      fixedIncome: Math.abs(allocationInputs.fixedIncome - recommendedAllocation.fixedIncome),
      cash: Math.abs(allocationInputs.cash - recommendedAllocation.cash)
    };
    
    return Object.values(deviations).reduce((sum, dev) => sum + dev, 0);
  };

  const totalAllocation = Object.values(allocationInputs).reduce((sum, val) => sum + val, 0);
  const deviation = calculateDeviation();

  return (
    <div className="space-y-8">
      {/* Risk Tolerance Selection */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground flex items-center gap-3">
            <Shield className="w-7 h-7" />
            Your Risk Tolerance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="text-center">
              <h3 className="text-xl font-semibold mb-4">Current Risk Profile: {riskTolerance}</h3>
              <div className="flex justify-center gap-4">
                {Object.keys(riskLevels).map((level) => (
                  <Button
                    key={level}
                    variant={riskTolerance === level ? "default" : "outline"}
                    onClick={() => handleRiskChange(level)}
                    className={`h-16 px-6 text-lg font-semibold ${
                      riskTolerance === level ? '' : getRiskColor(level)
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      {getRiskIcon(level)}
                      {level}
                    </div>
                  </Button>
                ))}
              </div>
            </div>
            
            <div className="p-6 bg-muted/30 rounded-xl">
              <div className="flex items-center gap-3 mb-3">
                {getRiskIcon(riskTolerance)}
                <h4 className="text-lg font-semibold">{riskTolerance} Risk Profile</h4>
              </div>
              <p className="text-lg text-muted-foreground">{currentRisk.description}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Asset Allocation Input */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground">
            Your Ideal Asset Allocation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-3">
                    <label className="text-lg font-medium">Sharemarket</label>
                    <span className="text-xl font-bold">{allocationInputs.shares}%</span>
                  </div>
                  <Slider
                    value={[allocationInputs.shares]}
                    onValueChange={(value) => handleAllocationChange('shares', value)}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-muted-foreground mt-1">
                    <span>0%</span>
                    <span>100%</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-3">
                    <label className="text-lg font-medium">Real Estate</label>
                    <span className="text-xl font-bold">{allocationInputs.realEstate}%</span>
                  </div>
                  <Slider
                    value={[allocationInputs.realEstate]}
                    onValueChange={(value) => handleAllocationChange('realEstate', value)}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-muted-foreground mt-1">
                    <span>0%</span>
                    <span>100%</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-3">
                    <label className="text-lg font-medium">Fixed Income</label>
                    <span className="text-xl font-bold">{allocationInputs.fixedIncome}%</span>
                  </div>
                  <Slider
                    value={[allocationInputs.fixedIncome]}
                    onValueChange={(value) => handleAllocationChange('fixedIncome', value)}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-muted-foreground mt-1">
                    <span>0%</span>
                    <span>100%</span>
                  </div>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-3">
                    <label className="text-lg font-medium">Cash</label>
                    <span className="text-xl font-bold">{allocationInputs.cash}%</span>
                  </div>
                  <Slider
                    value={[allocationInputs.cash]}
                    onValueChange={(value) => handleAllocationChange('cash', value)}
                    max={100}
                    step={1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-sm text-muted-foreground mt-1">
                    <span>0%</span>
                    <span>100%</span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-muted/30 rounded-xl">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-lg font-medium">Total Allocation</span>
                  <span className={`text-xl font-bold ${totalAllocation === 100 ? 'text-green-600' : 'text-red-600'}`}>
                    {totalAllocation}%
                  </span>
                </div>
                <Progress value={totalAllocation} className="h-3" />
                {totalAllocation !== 100 && (
                  <p className="text-red-600 text-sm mt-2">
                    Please adjust your allocation to total 100%
                  </p>
                )}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Comparison with Recommended */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground">
            Comparison with Recommended Allocation
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-lg font-semibold mb-4">Your Allocation</h4>
                <div className="space-y-3">
                  {Object.entries(allocationInputs).map(([category, value]) => (
                    <div key={category} className="flex justify-between items-center">
                      <span className="capitalize text-lg">{category.replace(/([A-Z])/g, ' $1').trim()}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-muted rounded-full h-2">
                          <div 
                            className="bg-primary h-2 rounded-full" 
                            style={{ width: `${value}%` }}
                          ></div>
                        </div>
                        <span className="font-semibold w-12 text-right">{value}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-lg font-semibold mb-4">Recommended for {riskTolerance} Risk</h4>
                <div className="space-y-3">
                  {Object.entries(recommendedAllocation).map(([category, value]) => (
                    <div key={category} className="flex justify-between items-center">
                      <span className="capitalize text-lg">{category.replace(/([A-Z])/g, ' $1').trim()}</span>
                      <div className="flex items-center gap-2">
                        <div className="w-24 bg-muted rounded-full h-2">
                          <div 
                            className="bg-green-500 h-2 rounded-full" 
                            style={{ width: `${value}%` }}
                          ></div>
                        </div>
                        <span className="font-semibold w-12 text-right">{value}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="p-4 bg-muted/30 rounded-xl">
              <div className="flex items-center gap-3 mb-3">
                {deviation <= 20 ? (
                  <CheckCircle className="w-6 h-6 text-green-600" />
                ) : (
                  <AlertTriangle className="w-6 h-6 text-yellow-600" />
                )}
                <h4 className="text-lg font-semibold">Allocation Analysis</h4>
              </div>
              <p className="text-lg text-muted-foreground">
                {deviation <= 20 
                  ? `Your allocation is well-aligned with the recommended ${riskTolerance} risk profile (${deviation}% total deviation).`
                  : `Your allocation differs significantly from the recommended ${riskTolerance} risk profile (${deviation}% total deviation). Consider adjusting to better match your risk tolerance.`
                }
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Risk Assessment Summary */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground">
            Risk Assessment Summary
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-card-foreground mb-2">
                {riskTolerance}
              </div>
              <div className="text-muted-foreground text-lg">Risk Tolerance</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-card-foreground mb-2">
                {deviation}%
              </div>
              <div className="text-muted-foreground text-lg">Allocation Deviation</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-card-foreground mb-2">
                {totalAllocation === 100 ? '✓' : '✗'}
              </div>
              <div className="text-muted-foreground text-lg">Allocation Complete</div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
