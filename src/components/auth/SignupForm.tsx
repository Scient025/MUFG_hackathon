import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { SignupData, dataService } from "@/services/dataService";
import { Loader2, CheckCircle } from "lucide-react";
import { Send, Bot } from "lucide-react";

interface SignupFormProps {
  onSignupSuccess: (userId: string) => void;
  onCancel: () => void;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  message: string;
  timestamp: Date;
}

export function SignupForm({ onSignupSuccess, onCancel }: SignupFormProps) {
  // Gemini API constants
  const GEMINI_API_KEY = 'AIzaSyDBVO_5h4L6j2-M1q-1PecgC42sFQYWv0w';
  const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;

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
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'bot',
      message: "Hi! I'm here to help you fill out your superannuation profile. You can tell me things like 'I'm 35 years old' or 'My annual income is $75,000' and I'll update the form for you automatically.",
      timestamp: new Date()
    }
  ]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

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

  const parseUserInputWithLLM = async (input: string): Promise<Partial<SignupData>> => {
    const systemPrompt = `You are a form field extraction assistant. Analyze the user's message and extract relevant information for a superannuation signup form.

Available fields and their possible values:
- name: string (full name)
- age: number (18-100)
- gender: "Male" | "Female" | "Other"
- country: "Australia" | "New Zealand" | "United Kingdom" | "United States"
- employment_status: "Full-time" | "Part-time" | "Unemployed" | "Self-employed" | "Retired"
- annual_income: number (in dollars)
- current_savings: number (in dollars)
- retirement_age_goal: number (age when they want to retire)
- risk_tolerance: "Low" | "Medium" | "High"
- contribution_amount: number (monthly contribution in dollars)
- contribution_frequency: "Monthly" | "Quarterly" | "Annually"
- employer_contribution: number (employer's monthly contribution in dollars)
- years_contributed: number (years they've been contributing)
- investment_type: "Conservative" | "Balanced" | "Growth" | "Aggressive"
- fund_name: string
- marital_status: "Single" | "Married" | "Divorced" | "Widowed"
- number_of_dependents: number (0 or more)
- education_level: "High School" | "Bachelor" | "Master" | "PhD" | "Other"
- health_status: "Excellent" | "Good" | "Fair" | "Poor"
- home_ownership_status: "Own" | "Renting" | "Other"
- investment_experience_level: "Beginner" | "Intermediate" | "Expert"
- financial_goals: "Retirement" | "Education" | "Home" | "Travel" | "Other"
- insurance_coverage: "None" | "Basic" | "Comprehensive"
- pension_type: "Superannuation" | "401k" | "IRA" | "Other"
- withdrawal_strategy: "Fixed" | "Variable" | "Minimum" | "Maximum"

Return ONLY a valid JSON object with the extracted fields. If you can't extract a value for a field, don't include it. 
Handle currency amounts by removing currency symbols and 'k' suffixes (e.g., "$50k" becomes 50000).

Example: {"name": "John Smith", "age": 35, "annual_income": 75000}`;

    const userPrompt = `Extract form field values from this user message: "${input}"`;

    try {
      const response = await fetch(GEMINI_API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: `${systemPrompt}\n\nUser message: ${userPrompt}`
            }]
          }],
          generationConfig: {
            temperature: 0.1,
            maxOutputTokens: 1000,
            topP: 0.8,
            topK: 10,
          }
        })
      });

      if (!response.ok) {
        console.error(`API request failed: ${response.status} ${response.statusText}`);
        throw new Error(`API request failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Gemini API response:', data);
      const generatedText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      
      if (!generatedText) {
        console.error('No response text from API:', data);
        throw new Error('No response from API');
      }

      console.log('Generated text:', generatedText);

      // Extract JSON from the response (in case there's extra text)
      const jsonMatch = generatedText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No valid JSON found in response');
      }

      const extractedData = JSON.parse(jsonMatch[0]);
      console.log('Extracted data:', extractedData);
      
      // Validate the extracted data types
      const validatedData: Partial<SignupData> = {};
      
      for (const [key, value] of Object.entries(extractedData)) {
        if (key in formData) {
          // Type validation based on the original formData structure
          const originalValue = formData[key as keyof SignupData];
          if (typeof originalValue === 'number' && !isNaN(Number(value))) {
            (validatedData as any)[key] = Number(value);
          } else if (typeof originalValue === 'string') {
            (validatedData as any)[key] = String(value);
          }
        }
      }

      console.log('Validated data:', validatedData);
      return validatedData;
    } catch (error) {
      console.error('LLM parsing error:', error);
      console.log('Falling back to basic parsing for input:', input);
      // Fallback to basic parsing if LLM fails
      return parseUserInputBasic(input);
    }
  };

  const parseUserInputBasic = (input: string): Partial<SignupData> => {
    const updates: Partial<SignupData> = {};
    const lowercaseInput = input.toLowerCase();

    // Name parsing
    const nameMatch = input.match(/(?:my name is|i'm|i am called|call me)\s+([a-zA-Z\s]+)/i);
    if (nameMatch) {
      updates.name = nameMatch[1].trim();
    }

    // Age parsing
    const ageMatch = input.match(/(?:i'm|i am|my age is)\s+(\d+)(?:\s+years?\s+old)?/i) || 
                     input.match(/(\d+)\s+years?\s+old/i);
    if (ageMatch) {
      const age = parseInt(ageMatch[1]);
      if (age >= 18 && age <= 100) {
        updates.age = age;
      }
    }

    // Income parsing
    const incomeMatch = input.match(/(?:income|earn|salary|make)(?:\s+is)?\s+(?:\$)?(\d+(?:,\d{3})*(?:k|000)?)/i);
    if (incomeMatch) {
      let income = incomeMatch[1].replace(/,/g, '');
      if (income.endsWith('k')) {
        income = income.slice(0, -1) + '000';
      }
      updates.annual_income = parseFloat(income);
    }

    // Savings parsing
    const savingsMatch = input.match(/(?:saved|savings|have|current savings are?)\s+(?:\$)?(\d+(?:,\d{3})*(?:k|000)?)/i) || input.match(/(?:my\s+)?(?:current\s+)?savings?\s+(?:are?|is)\s+(?:\$)?(\d+(?:,\d{3})*(?:k|000)?)/i);
    if (savingsMatch) {
      let savings = savingsMatch[1].replace(/,/g, '');
      if (savings.endsWith('k')) {
        savings = savings.slice(0, -1) + '000';
      }
      updates.current_savings = parseFloat(savings);
    }

    // Gender parsing
    if (lowercaseInput.includes('male') && !lowercaseInput.includes('female')) {
      updates.gender = 'Male';
    } else if (lowercaseInput.includes('female')) {
      updates.gender = 'Female';
    }

    // Country parsing
    if (lowercaseInput.includes('australia')) updates.country = 'Australia';
    if (lowercaseInput.includes('new zealand')) updates.country = 'New Zealand';
    if (lowercaseInput.includes('united kingdom') || lowercaseInput.includes('uk')) updates.country = 'United Kingdom';
    if (lowercaseInput.includes('united states') || lowercaseInput.includes('usa')) updates.country = 'United States';

    // Risk tolerance parsing
    if (lowercaseInput.includes('low risk') || lowercaseInput.includes('conservative')) {
      updates.risk_tolerance = 'Low';
    } else if (lowercaseInput.includes('high risk') || lowercaseInput.includes('aggressive')) {
      updates.risk_tolerance = 'High';
    } else if (lowercaseInput.includes('medium risk') || lowercaseInput.includes('moderate')) {
      updates.risk_tolerance = 'Medium';
    }

    // Marital status parsing
    if (lowercaseInput.includes('single')) updates.marital_status = 'Single';
    if (lowercaseInput.includes('married')) updates.marital_status = 'Married';
    if (lowercaseInput.includes('divorced')) updates.marital_status = 'Divorced';
    if (lowercaseInput.includes('widowed')) updates.marital_status = 'Widowed';

    return updates;
  };

  // Handle chat messages with LLM integration
  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      message: chatInput.trim(),
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatLoading(true);

    try {
      // Use LLM to parse the user input
      const updates = await parseUserInputWithLLM(chatInput);
      const updatedFields = Object.keys(updates);

      if (updatedFields.length > 0) {
        // Update the form data
        setFormData(prev => ({ ...prev, ...updates }));

        // Create response message with more details
        const fieldDescriptions = updatedFields.map(field => {
          const value = updates[field as keyof SignupData];
          return `${field.replace(/_/g, ' ')}: ${value}`;
        }).join(', ');

        const responseMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          message: `Great! I've updated ${updatedFields.length} field(s): ${fieldDescriptions}. What else would you like to update?`,
          timestamp: new Date()
        };
        
        setChatMessages(prev => [...prev, responseMessage]);
      } else {
        // No fields recognized
        const responseMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          message: "I couldn't identify any form fields to update from your message. Try being more specific, like 'I'm 30 years old' or 'My income is $60,000'.",
          timestamp: new Date()
        };
        
        setChatMessages(prev => [...prev, responseMessage]);
      }
    } catch (error) {
      console.error('Chat processing error:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        message: "Sorry, I encountered an error processing your message. Please try again or fill out the form manually.",
        timestamp: new Date()
      };
      setChatMessages(prev => [...prev, errorMessage]);
    } finally {
      setChatInput("");
      setChatLoading(false);
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
      <div className="flex gap-6 w-full max-w-6xl">
        {/* Chatbot Panel - Now permanently visible on the left */}
        <div className="w-96 h-fit bg-white rounded-lg shadow-2xl border flex flex-col">
          {/* Chat Header */}
          <div className="bg-blue-600 text-white px-4 py-3 rounded-t-lg flex items-center gap-2">
            <Bot className="w-5 h-5" />
            <span className="font-medium">Form Assistant</span>
          </div>
          
          {/* Chat Messages */}
          <div className="h-96 overflow-y-auto p-4 space-y-3">
            {chatMessages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-xs px-3 py-2 rounded-lg ${
                    message.type === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-800'
                  }`}
                >
                  <p className="text-sm">{message.message}</p>
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 px-3 py-2 rounded-lg">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Chat Input */}
          <form onSubmit={handleChatSubmit} className="border-t p-3">
            <div className="flex gap-2">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Tell me about yourself..."
                className="flex-1 px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={chatLoading}
              />
              <button
                type="submit"
                disabled={chatLoading || !chatInput.trim()}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white p-2 rounded-md transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </form>
        </div>

        {/* Form Card - Now on the right side */}
        <Card className="flex-1 max-w-2xl">
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
    </div>
  );
}