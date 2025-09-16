import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Heart, 
  AlertTriangle, 
  Shield, 
  TrendingUp, 
  Users, 
  Target,
  BarChart3,
  Lightbulb
} from "lucide-react";

interface AdvancedInsightsProps {
  user: any;
  advancedAnalysis?: any;
}

export function AdvancedInsights({ user, advancedAnalysis }: AdvancedInsightsProps) {
  if (!advancedAnalysis) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Advanced Insights</CardTitle>
          <CardDescription>Loading advanced analytics...</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center text-muted-foreground">
            Advanced insights are being calculated...
          </div>
        </CardContent>
      </Card>
    );
  }

  const financialHealth = advancedAnalysis.financial_health || {};
  const churnRisk = advancedAnalysis.churn_risk || {};
  const anomalyDetection = advancedAnalysis.anomaly_detection || {};
  const fundRecommendations = advancedAnalysis.fund_recommendations || {};
  const monteCarlo = advancedAnalysis.monte_carlo_simulation || {};
  const peerMatching = advancedAnalysis.peer_matching || {};
  const portfolioOptimization = advancedAnalysis.portfolio_optimization || {};

  const getHealthStatus = (score: number) => {
    if (score >= 80) return { status: "Excellent", color: "bg-green-100 text-green-800" };
    if (score >= 60) return { status: "Good", color: "bg-yellow-100 text-yellow-800" };
    return { status: "Needs Improvement", color: "bg-red-100 text-red-800" };
  };

  const getRiskStatus = (probability: number) => {
    if (probability < 0.3) return { status: "Low", color: "bg-green-100 text-green-800" };
    if (probability < 0.6) return { status: "Medium", color: "bg-yellow-100 text-yellow-800" };
    return { status: "High", color: "bg-red-100 text-red-800" };
  };

  const healthStatus = getHealthStatus(financialHealth.financial_health_score || 0);
  const churnStatus = getRiskStatus(churnRisk.churn_probability || 0);

  return (
    <div className="space-y-6">
      {/* Financial Health Score */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Heart className="w-5 h-5" />
            Financial Health Score
          </CardTitle>
          <CardDescription>Comprehensive assessment of your financial wellness</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-2xl font-bold">
                {financialHealth.financial_health_score?.toFixed(0) || 0}/100
              </span>
              <Badge className={healthStatus.color}>
                {healthStatus.status}
              </Badge>
            </div>
            <Progress 
              value={financialHealth.financial_health_score || 0} 
              className="h-3"
            />
            <div className="text-sm text-muted-foreground">
              Above {financialHealth.peer_percentile?.toFixed(0) || 0}% of your peers
            </div>
            {financialHealth.recommendations && (
              <div className="mt-3">
                <h4 className="font-semibold mb-2">Recommendations:</h4>
                <ul className="text-sm space-y-1">
                  {financialHealth.recommendations.map((rec: string, index: number) => (
                    <li key={index} className="flex items-center gap-2">
                      <Lightbulb className="w-3 h-3 text-yellow-500" />
                      {rec}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Risk Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Churn Risk */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <AlertTriangle className="w-5 h-5" />
              Churn Risk
            </CardTitle>
            <CardDescription>Probability of stopping contributions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xl font-bold">
                  {((churnRisk.churn_probability || 0) * 100).toFixed(1)}%
                </span>
                <Badge className={churnStatus.color}>
                  {churnStatus.status} Risk
                </Badge>
              </div>
              <Progress 
                value={(churnRisk.churn_probability || 0) * 100} 
                className="h-2"
              />
              {churnRisk.recommendations && (
                <div className="text-xs text-muted-foreground">
                  {churnRisk.recommendations[0]}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Anomaly Detection */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Account Monitoring
            </CardTitle>
            <CardDescription>Unusual activity detection</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xl font-bold">
                  {anomalyDetection.anomaly_percentage?.toFixed(1) || 0}%
                </span>
                <Badge className={anomalyDetection.is_anomaly ? "bg-red-100 text-red-800" : "bg-green-100 text-green-800"}>
                  {anomalyDetection.is_anomaly ? "Anomaly Detected" : "Normal Activity"}
                </Badge>
              </div>
              <Progress 
                value={anomalyDetection.anomaly_percentage || 0} 
                className="h-2"
              />
              {anomalyDetection.recommendations && (
                <div className="text-xs text-muted-foreground">
                  {anomalyDetection.recommendations[0]}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Fund Recommendations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Fund Recommendations
          </CardTitle>
          <CardDescription>Personalized fund suggestions based on your profile</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="text-sm text-muted-foreground">
              Current: <span className="font-medium">{fundRecommendations.current_fund || 'Unknown'}</span>
            </div>
            <div className="text-sm text-muted-foreground">
              {fundRecommendations.reasoning || 'Based on similar users'}
            </div>
            {fundRecommendations.recommendations && fundRecommendations.recommendations.length > 0 && (
              <div>
                <h4 className="font-semibold mb-2">Recommended Funds:</h4>
                <div className="flex flex-wrap gap-2">
                  {fundRecommendations.recommendations.slice(0, 3).map((fund: string, index: number) => (
                    <Badge key={index} variant="outline">
                      {fund}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Monte Carlo Simulation */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Retirement Stress Test
          </CardTitle>
          <CardDescription>Monte Carlo simulation results ({monteCarlo.simulations || 0} scenarios)</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
              <div>
                <div className="text-xs text-muted-foreground">10th Percentile</div>
                <div className="font-semibold">${(monteCarlo.percentiles?.p10 || 0).toLocaleString()}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">25th Percentile</div>
                <div className="font-semibold">${(monteCarlo.percentiles?.p25 || 0).toLocaleString()}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">50th Percentile</div>
                <div className="font-semibold">${(monteCarlo.percentiles?.p50 || 0).toLocaleString()}</div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">90th Percentile</div>
                <div className="font-semibold">${(monteCarlo.percentiles?.p90 || 0).toLocaleString()}</div>
              </div>
            </div>
            <div className="text-center">
              <div className="text-sm text-muted-foreground">
                Probability of meeting target: {((monteCarlo.probability_above_target || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Peer Insights */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="w-5 h-5" />
            Peer Insights
          </CardTitle>
          <CardDescription>How you compare to similar users</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="text-sm text-muted-foreground">
              Found {peerMatching.total_peers_found || 0} similar users
            </div>
            {peerMatching.peers && peerMatching.peers.length > 0 && (
              <div>
                <h4 className="font-semibold mb-2">Similar Users:</h4>
                <div className="space-y-2">
                  {peerMatching.peers.slice(0, 3).map((peer: any, index: number) => (
                    <div key={index} className="flex justify-between items-center text-sm">
                      <span>Age {peer.age}, {peer.risk_tolerance} risk</span>
                      <span className="text-muted-foreground">
                        {(peer.similarity_score * 100).toFixed(0)}% similar
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Portfolio Optimization */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="w-5 h-5" />
            Portfolio Optimization
          </CardTitle>
          <CardDescription>Optimal allocation recommendations</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {portfolioOptimization.portfolio_metrics && (
              <div className="grid grid-cols-3 gap-4 text-center">
                <div>
                  <div className="text-xs text-muted-foreground">Expected Return</div>
                  <div className="font-semibold">{(portfolioOptimization.portfolio_metrics.expected_return || 0).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Volatility</div>
                  <div className="font-semibold">{(portfolioOptimization.portfolio_metrics.volatility || 0).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Sharpe Ratio</div>
                  <div className="font-semibold">{(portfolioOptimization.portfolio_metrics.sharpe_ratio || 0).toFixed(2)}</div>
                </div>
              </div>
            )}
            {portfolioOptimization.optimized_allocation && portfolioOptimization.optimized_allocation.length > 0 && (
              <div>
                <h4 className="font-semibold mb-2">Recommended Allocation:</h4>
                <div className="space-y-2">
                  {portfolioOptimization.optimized_allocation.slice(0, 3).map((allocation: any, index: number) => (
                    <div key={index} className="flex justify-between items-center text-sm">
                      <span>{allocation.fund_name}</span>
                      <span className="font-medium">{allocation.allocation_percent.toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
