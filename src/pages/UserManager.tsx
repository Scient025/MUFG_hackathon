import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { UserSelectionPanel } from "@/components/dashboard/UserSelectionPanel";
import { useAdminAuth } from "@/contexts/AdminAuthContext";
import { ArrowLeft, User, Mail, Calendar, DollarSign, TrendingUp, Shield } from "lucide-react";

export default function UserManager() {
  const [selectedUser, setSelectedUser] = useState<any | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  const { loginAsUser } = useAdminAuth();

  const handleUserSelect = (user: any) => {
    setSelectedUser(user);
    setError(null);
  };

  const handleLoginAsUser = async (userId: string) => {
    try {
      setLoading(true);
      setError(null);
      
      if (!selectedUser) {
        setError("No user selected");
        return;
      }
      
      // Map backend user shape to MUFG-style shape expected by admin auth/Dashboard
      const mapped = {
        User_ID: selectedUser.User_ID || selectedUser.id,
        Name: selectedUser.Name || selectedUser.name || 'N/A',
        Age: selectedUser.Age || selectedUser.age || 0,
        Current_Savings: selectedUser.Current_Savings ?? selectedUser.current_savings ?? 0,
        Retirement_Age_Goal: selectedUser.Retirement_Age_Goal ?? selectedUser.retirement_age_goal ?? 65,
        Annual_Income: selectedUser.Annual_Income ?? selectedUser.annual_income ?? 0,
        Risk_Tolerance: selectedUser.Risk_Tolerance ?? selectedUser.risk_tolerance ?? 'Medium',
        // include other fields if needed
      } as any;

      // Login as the selected user using admin auth
      loginAsUser(mapped);
      
      // Navigate to dashboard with explicit userId for loading
      navigate('/dashboard', { state: { userId: mapped.User_ID } });
    } catch (err) {
      setError("Failed to login as user");
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-AU', {
      style: 'currency',
      currency: 'AUD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">User Manager</h1>
            <p className="text-xl text-gray-600">Select and manage user accounts</p>
          </div>
          <Button 
            onClick={() => navigate('/admin')}
            variant="outline"
            className="flex items-center gap-2"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Admin
          </Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* User Selection Panel */}
          <div>
            <UserSelectionPanel 
              onUserSelect={handleUserSelect}
              onLoginAsUser={handleLoginAsUser}
            />
          </div>

          {/* User Details Panel */}
          <div>
            {selectedUser ? (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <User className="w-5 h-5" />
                    User Details: {selectedUser.name}
                  </CardTitle>
                  <CardDescription>Complete profile information</CardDescription>
                </CardHeader>
                <CardContent>
                  {error && (
                    <Alert variant="destructive" className="mb-4">
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}

                  <div className="space-y-6">
                    {/* Personal Information */}
                    <div>
                      <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <User className="w-4 h-4" />
                        Personal Information
                      </h3>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Name:</span>
                          <div className="text-muted-foreground">{selectedUser.name}</div>
                        </div>
                        <div>
                          <span className="font-medium">Email:</span>
                          <div className="text-muted-foreground">{selectedUser.email}</div>
                        </div>
                        <div>
                          <span className="font-medium">Age:</span>
                          <div className="text-muted-foreground">{selectedUser.age}</div>
                        </div>
                        <div>
                          <span className="font-medium">Gender:</span>
                          <div className="text-muted-foreground">{selectedUser.gender}</div>
                        </div>
                        <div>
                          <span className="font-medium">Country:</span>
                          <div className="text-muted-foreground">{selectedUser.country}</div>
                        </div>
                        <div>
                          <span className="font-medium">Marital Status:</span>
                          <div className="text-muted-foreground">{selectedUser.marital_status}</div>
                        </div>
                        <div>
                          <span className="font-medium">Dependents:</span>
                          <div className="text-muted-foreground">{selectedUser.number_of_dependents}</div>
                        </div>
                        <div>
                          <span className="font-medium">Joined:</span>
                          <div className="text-muted-foreground">{formatDate(selectedUser.created_at)}</div>
                        </div>
                      </div>
                    </div>

                    {/* Financial Information */}
                    <div>
                      <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <DollarSign className="w-4 h-4" />
                        Financial Information
                      </h3>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Annual Income:</span>
                          <div className="text-muted-foreground">{formatCurrency(selectedUser.annual_income)}</div>
                        </div>
                        <div>
                          <span className="font-medium">Current Savings:</span>
                          <div className="text-muted-foreground">{formatCurrency(selectedUser.current_savings)}</div>
                        </div>
                        <div>
                          <span className="font-medium">Monthly Contribution:</span>
                          <div className="text-muted-foreground">{formatCurrency(selectedUser.contribution_amount)}</div>
                        </div>
                        <div>
                          <span className="font-medium">Employer Contribution:</span>
                          <div className="text-muted-foreground">{formatCurrency(selectedUser.employer_contribution)}</div>
                        </div>
                        <div>
                          <span className="font-medium">Retirement Goal Age:</span>
                          <div className="text-muted-foreground">{selectedUser.retirement_age_goal}</div>
                        </div>
                        <div>
                          <span className="font-medium">Years Contributed:</span>
                          <div className="text-muted-foreground">{selectedUser.years_contributed}</div>
                        </div>
                      </div>
                    </div>

                    {/* Investment Preferences */}
                    <div>
                      <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4" />
                        Investment Preferences
                      </h3>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Risk Tolerance:</span>
                          <div className="text-muted-foreground">
                            <span className={`px-2 py-1 rounded-full text-xs ${
                              selectedUser.risk_tolerance === 'High' ? 'bg-red-100 text-red-800' :
                              selectedUser.risk_tolerance === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                              'bg-green-100 text-green-800'
                            }`}>
                              {selectedUser.risk_tolerance}
                            </span>
                          </div>
                        </div>
                        <div>
                          <span className="font-medium">Investment Experience:</span>
                          <div className="text-muted-foreground">{selectedUser.investment_experience_level}</div>
                        </div>
                        <div>
                          <span className="font-medium">Investment Type:</span>
                          <div className="text-muted-foreground">{selectedUser.investment_type}</div>
                        </div>
                        <div>
                          <span className="font-medium">Fund Name:</span>
                          <div className="text-muted-foreground">{selectedUser.fund_name}</div>
                        </div>
                        <div>
                          <span className="font-medium">Financial Goals:</span>
                          <div className="text-muted-foreground">{selectedUser.financial_goals}</div>
                        </div>
                        <div>
                          <span className="font-medium">Pension Type:</span>
                          <div className="text-muted-foreground">{selectedUser.pension_type}</div>
                        </div>
                      </div>
                    </div>

                    {/* Additional Information */}
                    <div>
                      <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <Shield className="w-4 h-4" />
                        Additional Information
                      </h3>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="font-medium">Employment Status:</span>
                          <div className="text-muted-foreground">{selectedUser.employment_status}</div>
                        </div>
                        <div>
                          <span className="font-medium">Education Level:</span>
                          <div className="text-muted-foreground">{selectedUser.education_level}</div>
                        </div>
                        <div>
                          <span className="font-medium">Health Status:</span>
                          <div className="text-muted-foreground">{selectedUser.health_status}</div>
                        </div>
                        <div>
                          <span className="font-medium">Home Ownership:</span>
                          <div className="text-muted-foreground">{selectedUser.home_ownership_status}</div>
                        </div>
                        <div>
                          <span className="font-medium">Insurance Coverage:</span>
                          <div className="text-muted-foreground">{selectedUser.insurance_coverage}</div>
                        </div>
                        <div>
                          <span className="font-medium">Withdrawal Strategy:</span>
                          <div className="text-muted-foreground">{selectedUser.withdrawal_strategy}</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="mt-6 flex gap-4">
                    <Button 
                      onClick={() => handleLoginAsUser(selectedUser.id)}
                      disabled={loading}
                      className="flex-1"
                    >
                      Login as {selectedUser.name}
                    </Button>
                    <Button 
                      variant="outline"
                      onClick={() => setSelectedUser(null)}
                    >
                      Clear Selection
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ) : (
              <Card>
                <CardContent className="p-6">
                  <div className="text-center text-muted-foreground">
                    <User className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <h3 className="text-lg font-medium mb-2">No User Selected</h3>
                    <p>Select a user from the panel to view their details</p>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
