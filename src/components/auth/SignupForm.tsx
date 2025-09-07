import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SignupData, dataService } from "@/services/dataService";
import { Loader2, CheckCircle } from "lucide-react";

interface SignupFormProps {
  onSignupSuccess: (userId: string) => void;
  onCancel: () => void;
}

export function SignupForm({ onSignupSuccess, onCancel }: SignupFormProps) {
  const [formData, setFormData] = useState<SignupData>({
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

  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleInputChange = (field: keyof SignupData, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const result = await dataService.signupUser(formData);
      setSuccess(true);
      setTimeout(() => {
        onSignupSuccess(result.user_id);
      }, 2000);
    } catch (error) {
      console.error("Signup failed:", error);
      alert("Signup failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <Card className="w-full max-w-md">
          <CardContent className="pt-6">
            <div className="text-center">
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-green-700 mb-2">Signup Successful!</h2>
              <p className="text-gray-600">Welcome to SuperWise! Redirecting to your dashboard...</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <Card className="w-full max-w-2xl">
        <CardHeader>
          <CardTitle className="text-3xl font-bold text-center">Join SuperWise</CardTitle>
          <CardDescription className="text-center text-lg">
            Create your personalized superannuation profile
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Personal Information */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="name" className="text-lg font-medium">Full Name</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => handleInputChange("name", e.target.value)}
                  className="text-lg h-12"
                  required
                />
              </div>
              <div>
                <Label htmlFor="age" className="text-lg font-medium">Age</Label>
                <Input
                  id="age"
                  type="number"
                  value={formData.age}
                  onChange={(e) => handleInputChange("age", parseInt(e.target.value))}
                  className="text-lg h-12"
                  min="18"
                  max="100"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label className="text-lg font-medium">Gender</Label>
                <Select value={formData.gender} onValueChange={(value) => handleInputChange("gender", value)}>
                  <SelectTrigger className="h-12 text-lg">
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
                <Label className="text-lg font-medium">Country</Label>
                <Select value={formData.country} onValueChange={(value) => handleInputChange("country", value)}>
                  <SelectTrigger className="h-12 text-lg">
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

            {/* Financial Information */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="annual_income" className="text-lg font-medium">Annual Income ($)</Label>
                <Input
                  id="annual_income"
                  type="number"
                  value={formData.annual_income}
                  onChange={(e) => handleInputChange("annual_income", parseFloat(e.target.value))}
                  className="text-lg h-12"
                  min="0"
                  required
                />
              </div>
              <div>
                <Label htmlFor="current_savings" className="text-lg font-medium">Current Savings ($)</Label>
                <Input
                  id="current_savings"
                  type="number"
                  value={formData.current_savings}
                  onChange={(e) => handleInputChange("current_savings", parseFloat(e.target.value))}
                  className="text-lg h-12"
                  min="0"
                  required
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label htmlFor="contribution_amount" className="text-lg font-medium">Monthly Contribution ($)</Label>
                <Input
                  id="contribution_amount"
                  type="number"
                  value={formData.contribution_amount}
                  onChange={(e) => handleInputChange("contribution_amount", parseFloat(e.target.value))}
                  className="text-lg h-12"
                  min="0"
                  required
                />
              </div>
              <div>
                <Label htmlFor="employer_contribution" className="text-lg font-medium">Employer Contribution ($)</Label>
                <Input
                  id="employer_contribution"
                  type="number"
                  value={formData.employer_contribution}
                  onChange={(e) => handleInputChange("employer_contribution", parseFloat(e.target.value))}
                  className="text-lg h-12"
                  min="0"
                  required
                />
              </div>
            </div>

            {/* Risk and Investment */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <Label className="text-lg font-medium">Risk Tolerance</Label>
                <Select value={formData.risk_tolerance} onValueChange={(value) => handleInputChange("risk_tolerance", value)}>
                  <SelectTrigger className="h-12 text-lg">
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
                <Label className="text-lg font-medium">Investment Experience</Label>
                <Select value={formData.investment_experience_level} onValueChange={(value) => handleInputChange("investment_experience_level", value)}>
                  <SelectTrigger className="h-12 text-lg">
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
                <Label className="text-lg font-medium">Marital Status</Label>
                <Select value={formData.marital_status} onValueChange={(value) => handleInputChange("marital_status", value)}>
                  <SelectTrigger className="h-12 text-lg">
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
                <Label htmlFor="number_of_dependents" className="text-lg font-medium">Number of Dependents</Label>
                <Input
                  id="number_of_dependents"
                  type="number"
                  value={formData.number_of_dependents}
                  onChange={(e) => handleInputChange("number_of_dependents", parseInt(e.target.value))}
                  className="text-lg h-12"
                  min="0"
                  required
                />
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4 pt-6">
              <Button
                type="button"
                variant="outline"
                onClick={onCancel}
                className="flex-1 h-12 text-lg"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={loading}
                className="flex-1 h-12 text-lg"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-5 h-5 mr-2 animate-spin" />
                    Creating Account...
                  </>
                ) : (
                  "Create Account"
                )}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
