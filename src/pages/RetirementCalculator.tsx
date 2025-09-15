import { useState, useMemo } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from '@/components/ui/slider';
import { Loader2, Calculator, ArrowLeft, Target, AlertCircle } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { supabase } from "@/lib/supabase";
import { UserProfile } from "@/lib/supabase";

interface CalculatorInputs {
  currentAge: number;
  retirementAge: number;
  currentBalance: number;
  monthlyContribution: number;
  employerContribution: number;
  annualReturnRate: number;
  inflationRate: number;
  taxRate: number;
  yearlyWithdrawal: number;
  salaryGrowthRate: number;
  contributionIncreaseRate: number;
}

interface ProjectionData {
  age: number;
  year: number;
  balance: number;
  totalContributions: number;
  investmentGrowth: number;
  afterTaxBalance: number;
  inflationAdjustedBalance: number;
}

interface RetirementAnalysis {
  projectedBalance: number;
  totalContributions: number;
  investmentGrowth: number;
  yearsOfWithdrawal: number;
  monthlyRetirementIncome: number;
  inflationAdjustedIncome: number;
  shortfall: number;
}

export default function RetirementCalculator() {
  const { userId } = useParams<{ userId: string }>();
  const navigate = useNavigate();
  const [userProfile, setUserProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);

  // Calculator inputs derived from user profile
  const [calculatorInputs, setCalculatorInputs] = useState<CalculatorInputs>({
    currentAge: 30,
    retirementAge: 65,
    currentBalance: 10000,
    monthlyContribution: 1000,
    employerContribution: 500,
    annualReturnRate: 7.5,
    inflationRate: 2.5,
    taxRate: 15,
    yearlyWithdrawal: 60000,
    salaryGrowthRate: 3.0,
    contributionIncreaseRate: 2.0
  });

  // Load user profile on component mount
  useMemo(() => {
    const loadUserProfile = async () => {
      if (!userId) {
        navigate('/dashboard');
        return;
      }

      try {
        // Try to load from Supabase first
        const { data, error } = await supabase
          .from('MUFG')
          .select('*')
          .eq('User_ID', userId)
          .single();

        if (data && !error) {
          setUserProfile(data);
          
          // Update calculator inputs from user profile
          setCalculatorInputs(prev => ({
            ...prev,
            currentAge: data.Age,
            retirementAge: data.Retirement_Age_Goal,
            currentBalance: data.Current_Savings,
            monthlyContribution: data.Contribution_Amount,
            employerContribution: data.Employer_Contribution,
            annualReturnRate: data.Annual_Return_Rate || (data.Risk_Tolerance === 'High' ? 9 : data.Risk_Tolerance === 'Low' ? 6 : 7.5),
          }));
        } else {
          // If not found in Supabase, create a mock profile from localStorage or default values
          console.warn('User profile not found in Supabase, using default values');
          
          // Try to get user data from localStorage (stored during signup)
          const storedUserData = localStorage.getItem(`user_${userId}`);
          let mockProfile: UserProfile;
          
          if (storedUserData) {
            const userData = JSON.parse(storedUserData);
            mockProfile = {
              id: userId,
              email: `${userData.name?.toLowerCase().replace(/\s+/g, '') || 'demo'}@demo.com`,
              name: userData.name || 'Demo User',
              age: userData.age || 30,
              gender: userData.gender || 'Male',
              country: userData.country || 'Australia',
              employment_status: userData.employment_status || 'Full-time',
              annual_income: userData.annual_income || 50000,
              current_savings: userData.current_savings || 10000,
              retirement_age_goal: userData.retirement_age_goal || 65,
              risk_tolerance: userData.risk_tolerance || 'Medium',
              contribution_amount: userData.contribution_amount || 1000,
              contribution_frequency: userData.contribution_frequency || 'Monthly',
              employer_contribution: userData.employer_contribution || 500,
              years_contributed: userData.years_contributed || 5,
              investment_type: userData.investment_type || 'Balanced',
              fund_name: userData.fund_name || 'Default Fund',
              marital_status: userData.marital_status || 'Single',
              number_of_dependents: userData.number_of_dependents || 0,
              education_level: userData.education_level || 'Bachelor',
              health_status: userData.health_status || 'Good',
              home_ownership_status: userData.home_ownership_status || 'Renting',
              investment_experience_level: userData.investment_experience_level || 'Beginner',
              financial_goals: userData.financial_goals || 'Retirement',
              insurance_coverage: userData.insurance_coverage || 'Basic',
              pension_type: userData.pension_type || 'Superannuation',
              withdrawal_strategy: userData.withdrawal_strategy || 'Fixed',
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            };
          } else {
            // Create a default profile
            mockProfile = {
              id: userId,
              email: 'demo@demo.com',
              name: 'Demo User',
              age: 30,
              gender: 'Male',
              country: 'Australia',
              employment_status: 'Full-time',
              annual_income: 50000,
              current_savings: 10000,
              retirement_age_goal: 65,
              risk_tolerance: 'Medium',
              contribution_amount: 1000,
              contribution_frequency: 'Monthly',
              employer_contribution: 500,
              years_contributed: 5,
              investment_type: 'Balanced',
              fund_name: 'Default Fund',
              marital_status: 'Single',
              number_of_dependents: 0,
              education_level: 'Bachelor',
              health_status: 'Good',
              home_ownership_status: 'Renting',
              investment_experience_level: 'Beginner',
              financial_goals: 'Retirement',
              insurance_coverage: 'Basic',
              pension_type: 'Superannuation',
              withdrawal_strategy: 'Fixed',
              created_at: new Date().toISOString(),
              updated_at: new Date().toISOString()
            };
          }
          
          setUserProfile(mockProfile);
          
          // Update calculator inputs from mock profile
          setCalculatorInputs(prev => ({
            ...prev,
            currentAge: mockProfile.age,
            retirementAge: mockProfile.retirement_age_goal,
            currentBalance: mockProfile.current_savings,
            monthlyContribution: mockProfile.contribution_amount,
            employerContribution: mockProfile.employer_contribution,
            annualReturnRate: mockProfile.risk_tolerance === 'High' ? 9 : mockProfile.risk_tolerance === 'Low' ? 6 : 7.5,
          }));
        }
      } catch (error) {
        console.error('Error loading user profile:', error);
        navigate('/dashboard');
      } finally {
        setLoading(false);
      }
    };

    loadUserProfile();
  }, [userId, navigate]);

  const handleCalculatorInputChange = (key: keyof CalculatorInputs, value: number) => {
    setCalculatorInputs(prev => ({ ...prev, [key]: value }));
  };

  // Calculator logic
  const projectionData = useMemo(() => {
    const data: ProjectionData[] = [];
    const yearsToRetirement = calculatorInputs.retirementAge - calculatorInputs.currentAge;
    
    let balance = calculatorInputs.currentBalance;
    let totalContributions = calculatorInputs.currentBalance;
    let monthlyContrib = calculatorInputs.monthlyContribution;
    let employerContrib = calculatorInputs.employerContribution;
    
    for (let i = 0; i <= yearsToRetirement; i++) {
      const age = calculatorInputs.currentAge + i;
      const year = new Date().getFullYear() + i;
      
      if (i > 0) {
        monthlyContrib *= (1 + calculatorInputs.contributionIncreaseRate / 100);
        employerContrib *= (1 + calculatorInputs.salaryGrowthRate / 100);
      }
      
      const annualContribution = (monthlyContrib + employerContrib) * 12;
      
      balance += annualContribution;
      totalContributions += annualContribution;
      
      const investmentReturn = balance * (calculatorInputs.annualReturnRate / 100);
      balance += investmentReturn;
      
      const investmentGrowth = balance - totalContributions;
      
      const taxableGains = Math.max(0, investmentGrowth * 0.7);
      const tax = taxableGains * (calculatorInputs.taxRate / 100);
      const afterTaxBalance = balance - tax;
      
      const inflationAdjustedBalance = afterTaxBalance / Math.pow(1 + calculatorInputs.inflationRate / 100, i);
      
      data.push({
        age,
        year,
        balance: Math.round(balance),
        totalContributions: Math.round(totalContributions),
        investmentGrowth: Math.round(investmentGrowth),
        afterTaxBalance: Math.round(afterTaxBalance),
        inflationAdjustedBalance: Math.round(inflationAdjustedBalance)
      });
    }
    
    return data;
  }, [calculatorInputs]);

  const retirementAnalysis = useMemo((): RetirementAnalysis => {
    const finalProjection = projectionData[projectionData.length - 1];
    const projectedBalance = finalProjection.afterTaxBalance;
    const totalContributions = finalProjection.totalContributions;
    const investmentGrowth = finalProjection.investmentGrowth;
    
    const yearsOfWithdrawal = 25;
    const totalRetirementNeeds = calculatorInputs.yearlyWithdrawal * yearsOfWithdrawal;
    
    const inflationAdjustedNeeds = calculatorInputs.yearlyWithdrawal / Math.pow(1 + calculatorInputs.inflationRate / 100, calculatorInputs.retirementAge - calculatorInputs.currentAge);
    
    const monthlyRetirementIncome = projectedBalance / yearsOfWithdrawal / 12;
    const inflationAdjustedIncome = monthlyRetirementIncome / Math.pow(1 + calculatorInputs.inflationRate / 100, calculatorInputs.retirementAge - calculatorInputs.currentAge);
    
    const shortfall = Math.max(0, totalRetirementNeeds - projectedBalance);
    
    return {
      projectedBalance,
      totalContributions,
      investmentGrowth,
      yearsOfWithdrawal,
      monthlyRetirementIncome,
      inflationAdjustedIncome,
      shortfall
    };
  }, [projectionData, calculatorInputs]);

  const balanceBreakdown = [
    { name: 'Your Contributions', value: retirementAnalysis.totalContributions, color: '#3b82f6' },
    { name: 'Investment Growth', value: retirementAnalysis.investmentGrowth, color: '#10b981' },
  ];

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-AU', { 
      style: 'currency', 
      currency: 'AUD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const formatPercent = (value: number) => `${value}%`;

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <Card className="w-full max-w-md">
          <CardContent className="pt-6">
            <div className="text-center">
              <Loader2 className="w-16 h-16 text-blue-500 mx-auto mb-4 animate-spin" />
              <h2 className="text-2xl font-bold text-gray-700 mb-2">Loading Your Calculator...</h2>
              <p className="text-gray-600">Preparing your personalized retirement projections</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (!userProfile) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <Card className="w-full max-w-md">
          <CardContent className="pt-6">
            <div className="text-center">
              <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-red-700 mb-2">Profile Not Found</h2>
              <p className="text-gray-600 mb-4">We couldn't find your profile. Please try again.</p>
              <Button onClick={() => navigate('/dashboard')}>
                Back to Dashboard
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-4">
            <Button
              onClick={() => navigate('/dashboard')}
              variant="outline"
              className="flex items-center gap-2"
            >
              <ArrowLeft className="w-4 h-4" />
              Back to Dashboard
            </Button>
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Retirement Calculator</h1>
              <p className="text-xl text-gray-600">Personalized projections for {userProfile.Name}</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Calculator Controls */}
          <div className="lg:col-span-1 space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Calculator className="w-5 h-5" />
                  Adjust Your Scenario
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <Label>Retirement Age: {calculatorInputs.retirementAge} years</Label>
                  <Slider
                    value={[calculatorInputs.retirementAge]}
                    onValueChange={([value]) => handleCalculatorInputChange('retirementAge', value)}
                    min={50}
                    max={80}
                    step={1}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Monthly Contribution: {formatCurrency(calculatorInputs.monthlyContribution)}</Label>
                  <Slider
                    value={[calculatorInputs.monthlyContribution]}
                    onValueChange={([value]) => handleCalculatorInputChange('monthlyContribution', value)}
                    min={0}
                    max={5000}
                    step={100}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Expected Return: {formatPercent(calculatorInputs.annualReturnRate)}</Label>
                  <Slider
                    value={[calculatorInputs.annualReturnRate]}
                    onValueChange={([value]) => handleCalculatorInputChange('annualReturnRate', value)}
                    min={3}
                    max={15}
                    step={0.5}
                    className="mt-2"
                  />
                </div>

                <div>
                  <Label>Annual Retirement Income Needed: {formatCurrency(calculatorInputs.yearlyWithdrawal)}</Label>
                  <Slider
                    value={[calculatorInputs.yearlyWithdrawal]}
                    onValueChange={([value]) => handleCalculatorInputChange('yearlyWithdrawal', value)}
                    min={30000}
                    max={150000}
                    step={5000}
                    className="mt-2"
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <CardContent className="p-6 text-center">
                  <div className="text-3xl font-bold text-green-600 mb-2">
                    {formatCurrency(retirementAnalysis.projectedBalance)}
                  </div>
                  <div className="text-sm text-gray-600">Projected Balance at Retirement</div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6 text-center">
                  <div className="text-3xl font-bold text-blue-600 mb-2">
                    {formatCurrency(retirementAnalysis.monthlyRetirementIncome)}
                  </div>
                  <div className="text-sm text-gray-600">Monthly Retirement Income</div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-6 text-center">
                  <div className={`text-3xl font-bold mb-2 ${retirementAnalysis.shortfall > 0 ? 'text-red-600' : 'text-green-600'}`}>
                    {retirementAnalysis.shortfall > 0 ? '-' : '+'}{formatCurrency(Math.abs(retirementAnalysis.shortfall))}
                  </div>
                  <div className="text-sm text-gray-600">
                    {retirementAnalysis.shortfall > 0 ? 'Shortfall' : 'Surplus'}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Growth Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Your Retirement Journey</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={projectionData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="age" />
                    <YAxis tickFormatter={(value) => `$${(value/1000).toFixed(0)}k`} />
                    <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="afterTaxBalance" 
                      stroke="#10b981" 
                      strokeWidth={3}
                      name="Projected Balance"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="totalContributions" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      name="Total Contributions"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Balance Breakdown */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Balance Composition</CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={balanceBreakdown}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {balanceBreakdown.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(value) => formatCurrency(Number(value))} />
                      <Legend />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertCircle className="w-5 h-5" />
                    Your Retirement Summary
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Years to Retirement:</span>
                      <span className="font-semibold">{calculatorInputs.retirementAge - calculatorInputs.currentAge} years</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Total You'll Contribute:</span>
                      <span className="font-semibold">{formatCurrency(retirementAnalysis.totalContributions)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Investment Growth:</span>
                      <span className="font-semibold text-green-600">{formatCurrency(retirementAnalysis.investmentGrowth)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Return Multiple:</span>
                      <span className="font-semibold">{(retirementAnalysis.projectedBalance / retirementAnalysis.totalContributions).toFixed(1)}x</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Monthly Income at Retirement:</span>
                      <span className="font-semibold text-blue-600">{formatCurrency(retirementAnalysis.monthlyRetirementIncome)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Status:</span>
                      <span className={`font-semibold ${retirementAnalysis.shortfall > 0 ? 'text-red-600' : 'text-green-600'}`}>
                        {retirementAnalysis.shortfall > 0 ? 'Needs Improvement' : 'On Track'}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Recommendations */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Target className="w-5 h-5" />
                  Personalized Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {retirementAnalysis.shortfall > 0 ? (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                      <h4 className="font-semibold text-red-800 mb-2">Action Required</h4>
                      <p className="text-red-700 mb-3">
                        You have a projected shortfall of {formatCurrency(retirementAnalysis.shortfall)}. Here are some ways to improve your retirement outlook:
                      </p>
                      <ul className="list-disc list-inside space-y-1 text-red-700 text-sm">
                        <li>Consider increasing your monthly contributions by ${Math.round(retirementAnalysis.shortfall / ((calculatorInputs.retirementAge - calculatorInputs.currentAge) * 12))}</li>
                        <li>Extend your retirement age by 2-3 years if possible</li>
                        <li>Review your investment strategy - higher risk tolerance could increase returns</li>
                        <li>Consider additional income streams during retirement</li>
                      </ul>
                    </div>
                  ) : (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                      <h4 className="font-semibold text-green-800 mb-2">Great Progress!</h4>
                      <p className="text-green-700 mb-3">
                        You're on track for a comfortable retirement. Consider these optimization strategies:
                      </p>
                      <ul className="list-disc list-inside space-y-1 text-green-700 text-sm">
                        <li>Review your portfolio annually to ensure it matches your risk tolerance</li>
                        <li>Consider salary sacrificing for tax benefits</li>
                        <li>Look into government co-contribution opportunities</li>
                        <li>Regularly review and adjust your contributions as your income grows</li>
                      </ul>
                    </div>
                  )}

                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <h4 className="font-semibold text-blue-800 mb-2">Based on Your Profile</h4>
                  <div className="text-blue-700 text-sm space-y-1">
                    <p><strong>Risk Tolerance:</strong> {userProfile.Risk_Tolerance} - Your projected return rate is {calculatorInputs.annualReturnRate}%</p>
                    <p><strong>Investment Experience:</strong> {userProfile.Investment_Experience_Level} - Consider reviewing investment options</p>
                    <p><strong>Time Horizon:</strong> {calculatorInputs.retirementAge - calculatorInputs.currentAge} years - Time is your biggest advantage</p>
                    {userProfile.Number_of_Dependents > 0 && (
                      <p><strong>Dependents:</strong> {userProfile.Number_of_Dependents} - Consider life insurance and education planning</p>
                    )}
                  </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
