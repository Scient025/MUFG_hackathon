import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2 } from "lucide-react";
import { supabase, UserProfile } from "@/lib/supabase";

export default function ProfileSetup() {
  const navigate = useNavigate();
  const location = useLocation();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [user, setUser] = useState<any>(null);

  const [formData, setFormData] = useState({
    name: "",
    age: 30,
    gender: "Male",
    country: "Australia",
    employment_status: "Full-time",
    annual_income: 50000,
    current_savings: 10000,
    retirement_age_goal: 65,
    risk_tolerance: "Medium",
    contribution_amount: 1000,
    contribution_frequency: "Monthly",
    employer_contribution: 500,
    years_contributed: 5,
    investment_type: "Balanced",
    fund_name: "Default Fund",
    marital_status: "Single",
    number_of_dependents: 0,
    education_level: "Bachelor",
    health_status: "Good",
    home_ownership_status: "Renting",
    investment_experience_level: "Beginner",
    financial_goals: "Retirement",
    insurance_coverage: "Basic",
    pension_type: "Superannuation",
    withdrawal_strategy: "Fixed"
  });

  useEffect(() => {
    if (location.state?.user) {
      setUser(location.state.user);
    } else {
      // If no user data, redirect to signup
      navigate("/signup");
    }
  }, [location.state, navigate]);

  const handleInputChange = (field: keyof typeof formData, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      // Create user profile in Supabase
      const { data, error } = await supabase
        .from('user_profiles')
        .insert([
          {
            id: user.id,
            email: user.email,
            ...formData,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString()
          }
        ]);

      if (error) {
        setError(error.message);
      } else {
        // Redirect to dashboard
        navigate("/dashboard");
      }
    } catch (err) {
      setError("An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-4xl mx-auto">
        <Card>
          <CardHeader className="text-center">
            <CardTitle className="text-3xl font-bold">Complete Your Profile</CardTitle>
            <CardDescription className="text-lg">
              Help us personalize your SuperWise experience
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {error && (
                <Alert variant="destructive">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {/* Personal Information */}
              <div className="space-y-4">
                <h3 className="text-xl font-semibold">Personal Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="name">Full Name</Label>
                    <Input
                      id="name"
                      value={formData.name}
                      onChange={(e) => handleInputChange("name", e.target.value)}
                      required
                      className="h-12"
                    />
                  </div>
                  <div>
                    <Label htmlFor="age">Age</Label>
                    <Input
                      id="age"
                      type="number"
                      value={formData.age}
                      onChange={(e) => handleInputChange("age", parseInt(e.target.value))}
                      min="18"
                      max="100"
                      required
                      className="h-12"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Gender</Label>
                    <Select value={formData.gender} onValueChange={(value) => handleInputChange("gender", value)}>
                      <SelectTrigger className="h-12">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Male">Male</SelectItem>
                        <SelectItem value="Female">Female</SelectItem>
                        <SelectItem value="Other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Country</Label>
                    <Select value={formData.country} onValueChange={(value) => handleInputChange("country", value)}>
                      <SelectTrigger className="h-12">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Australia">Australia</SelectItem>
                        <SelectItem value="New Zealand">New Zealand</SelectItem>
                        <SelectItem value="United Kingdom">United Kingdom</SelectItem>
                        <SelectItem value="United States">United States</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              {/* Financial Information */}
              <div className="space-y-4">
                <h3 className="text-xl font-semibold">Financial Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="annual_income">Annual Income ($)</Label>
                    <Input
                      id="annual_income"
                      type="number"
                      value={formData.annual_income}
                      onChange={(e) => handleInputChange("annual_income", parseFloat(e.target.value))}
                      min="0"
                      required
                      className="h-12"
                    />
                  </div>
                  <div>
                    <Label htmlFor="current_savings">Current Savings ($)</Label>
                    <Input
                      id="current_savings"
                      type="number"
                      value={formData.current_savings}
                      onChange={(e) => handleInputChange("current_savings", parseFloat(e.target.value))}
                      min="0"
                      required
                      className="h-12"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="contribution_amount">Monthly Contribution ($)</Label>
                    <Input
                      id="contribution_amount"
                      type="number"
                      value={formData.contribution_amount}
                      onChange={(e) => handleInputChange("contribution_amount", parseFloat(e.target.value))}
                      min="0"
                      required
                      className="h-12"
                    />
                  </div>
                  <div>
                    <Label htmlFor="employer_contribution">Employer Contribution ($)</Label>
                    <Input
                      id="employer_contribution"
                      type="number"
                      value={formData.employer_contribution}
                      onChange={(e) => handleInputChange("employer_contribution", parseFloat(e.target.value))}
                      min="0"
                      required
                      className="h-12"
                    />
                  </div>
                </div>
              </div>

              {/* Investment Preferences */}
              <div className="space-y-4">
                <h3 className="text-xl font-semibold">Investment Preferences</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Risk Tolerance</Label>
                    <Select value={formData.risk_tolerance} onValueChange={(value) => handleInputChange("risk_tolerance", value)}>
                      <SelectTrigger className="h-12">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Low">Low Risk</SelectItem>
                        <SelectItem value="Medium">Medium Risk</SelectItem>
                        <SelectItem value="High">High Risk</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label>Investment Experience</Label>
                    <Select value={formData.investment_experience_level} onValueChange={(value) => handleInputChange("investment_experience_level", value)}>
                      <SelectTrigger className="h-12">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Beginner">Beginner</SelectItem>
                        <SelectItem value="Intermediate">Intermediate</SelectItem>
                        <SelectItem value="Expert">Expert</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Marital Status</Label>
                    <Select value={formData.marital_status} onValueChange={(value) => handleInputChange("marital_status", value)}>
                      <SelectTrigger className="h-12">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="Single">Single</SelectItem>
                        <SelectItem value="Married">Married</SelectItem>
                        <SelectItem value="Divorced">Divorced</SelectItem>
                        <SelectItem value="Widowed">Widowed</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="number_of_dependents">Number of Dependents</Label>
                    <Input
                      id="number_of_dependents"
                      type="number"
                      value={formData.number_of_dependents}
                      onChange={(e) => handleInputChange("number_of_dependents", parseInt(e.target.value))}
                      min="0"
                      required
                      className="h-12"
                    />
                  </div>
                </div>
              </div>

              <div className="flex gap-4 pt-6">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => navigate("/signup")}
                  className="flex-1 h-12"
                >
                  Back
                </Button>
                <Button
                  type="submit"
                  disabled={loading}
                  className="flex-1 h-12"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creating Profile...
                    </>
                  ) : (
                    "Complete Setup"
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
