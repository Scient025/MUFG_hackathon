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

interface ConversationStep {
  id: string;
  question: string;
  field: keyof SignupData;
  type: 'text' | 'number' | 'select';
  options?: string[];
  validation?: (value: any) => boolean;
  parser?: (input: string) => any;
}

export function SignupForm({ onSignupSuccess, onCancel }: SignupFormProps) {
  // Gemini API constants
  const GEMINI_API_KEY = 'AIzaSyDBVO_5h4L6j2-M1q-1PecgC42sFQYWv0w';
  const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`;

  // Define conversation steps
  const conversationSteps: ConversationStep[] = [
    {
      id: 'name',
      question: "Let's start with the basics. What's your full name?",
      field: 'name',
      type: 'text',
      validation: (value: string) => value.length > 0
    },
    {
      id: 'age',
      question: "How old are you?",
      field: 'age',
      type: 'number',
      validation: (value: number) => value >= 18 && value <= 100,
      parser: (input: string) => {
        const ageMatch = input.match(/(\d+)/);
        return ageMatch ? parseInt(ageMatch[1]) : null;
      }
    },
    {
      id: 'gender',
      question: "What's your gender?",
      field: 'gender',
      type: 'select',
      options: ['Male', 'Female', 'Other']
    },
    {
      id: 'country',
      question: "Which country are you from?",
      field: 'country',
      type: 'select',
      options: ['Australia', 'New Zealand', 'United Kingdom', 'United States']
    },
    {
      id: 'annual_income',
      question: "What's your annual income? (You can say something like '$75,000' or '75k')",
      field: 'annual_income',
      type: 'number',
      validation: (value: number) => value >= 0,
      parser: (input: string) => {
        const match = input.match(/(?:\$)?(\d+(?:,\d{3})*(?:k|000)?)/i);
        if (match) {
          let amount = match[1].replace(/,/g, '');
          if (amount.toLowerCase().endsWith('k')) {
            amount = amount.slice(0, -1) + '000';
          }
          return parseFloat(amount);
        }
        return null;
      }
    },
    {
      id: 'current_savings',
      question: "How much do you currently have saved? (Include all your savings and investments)",
      field: 'current_savings',
      type: 'number',
      validation: (value: number) => value >= 0,
      parser: (input: string) => {
        const match = input.match(/(?:\$)?(\d+(?:,\d{3})*(?:k|000)?)/i);
        if (match) {
          let amount = match[1].replace(/,/g, '');
          if (amount.toLowerCase().endsWith('k')) {
            amount = amount.slice(0, -1) + '000';
          }
          return parseFloat(amount);
        }
        return null;
      }
    },
    {
      id: 'employment_status',
      question: "What's your current employment status?",
      field: 'employment_status',
      type: 'select',
      options: ['Full-time', 'Part-time', 'Unemployed', 'Self-employed', 'Retired']
    },
    {
      id: 'retirement_age_goal',
      question: "At what age would you like to retire?",
      field: 'retirement_age_goal',
      type: 'number',
      validation: (value: number) => value >= 50 && value <= 80,
      parser: (input: string) => {
        const ageMatch = input.match(/(\d+)/);
        return ageMatch ? parseInt(ageMatch[1]) : null;
      }
    },
    {
      id: 'contribution_amount',
      question: "How much would you like to contribute monthly to your superannuation?",
      field: 'contribution_amount',
      type: 'number',
      validation: (value: number) => value >= 0,
      parser: (input: string) => {
        const match = input.match(/(?:\$)?(\d+(?:,\d{3})*(?:k|000)?)/i);
        if (match) {
          let amount = match[1].replace(/,/g, '');
          if (amount.toLowerCase().endsWith('k')) {
            amount = amount.slice(0, -1) + '000';
          }
          return parseFloat(amount);
        }
        return null;
      }
    },
    {
      id: 'employer_contribution',
      question: "How much does your employer contribute monthly to your super?",
      field: 'employer_contribution',
      type: 'number',
      validation: (value: number) => value >= 0,
      parser: (input: string) => {
        const match = input.match(/(?:\$)?(\d+(?:,\d{3})*(?:k|000)?)/i);
        if (match) {
          let amount = match[1].replace(/,/g, '');
          if (amount.toLowerCase().endsWith('k')) {
            amount = amount.slice(0, -1) + '000';
          }
          return parseFloat(amount);
        }
        return null;
      }
    },
    {
      id: 'risk_tolerance',
      question: "What's your risk tolerance for investments? Are you conservative (low risk), balanced (medium risk), or aggressive (high risk)?",
      field: 'risk_tolerance',
      type: 'select',
      options: ['Low', 'Medium', 'High']
    },
    {
      id: 'marital_status',
      question: "What's your marital status?",
      field: 'marital_status',
      type: 'select',
      options: ['Single', 'Married', 'Divorced', 'Widowed']
    },
    {
      id: 'number_of_dependents',
      question: "How many dependents do you have? (children or others you financially support)",
      field: 'number_of_dependents',
      type: 'number',
      validation: (value: number) => value >= 0,
      parser: (input: string) => {
        const match = input.match(/(\d+)/);
        return match ? parseInt(match[1]) : 0;
      }
    },
    {
      id: 'financial_goals',
      question: "What's your main financial goal?",
      field: 'financial_goals',
      type: 'select',
      options: ['Retirement', 'Education', 'Home', 'Travel', 'Other']
    },
    {
      id: 'investment_experience_level',
      question: "How would you describe your investment experience?",
      field: 'investment_experience_level',
      type: 'select',
      options: ['Beginner', 'Intermediate', 'Expert']
    }
  ];

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
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'bot',
      message: "Hi! I'm here to help you set up your superannuation profile. I'll ask you a few questions to get everything set up perfectly for you. Let's get started!",
      timestamp: new Date()
    },
    {
      id: '2',
      type: 'bot',
      message: conversationSteps[0].question,
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

  const parseUserInputWithLLM = async (input: string, step: ConversationStep): Promise<any> => {
    const systemPrompt = `You are helping to extract a specific piece of information from user input for a form field.

Field: ${step.field}
Type: ${step.type}
${step.options ? `Valid options: ${step.options.join(', ')}` : ''}

Instructions:
- Extract only the specific information requested for the "${step.field}" field
- Return ONLY a valid JSON object with a single field called "value"
- For currency amounts, remove currency symbols and convert "k" to thousands (e.g., "$50k" becomes 50000)
- For select fields, match the user input to one of the valid options (case insensitive, partial matching allowed)
- If you cannot extract a valid value, return {"value": null}

Examples:
- For name: {"value": "John Smith"}
- For age: {"value": 35}
- For income: {"value": 75000}
- For gender: {"value": "Male"}`;

    const userPrompt = `Extract the ${step.field} from this user message: "${input}"`;

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
            maxOutputTokens: 200,
            topP: 0.8,
            topK: 10,
          }
        })
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.statusText}`);
      }

      const data = await response.json();
      const generatedText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      
      if (!generatedText) {
        throw new Error('No response from API');
      }

      const jsonMatch = generatedText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No valid JSON found in response');
      }

      const result = JSON.parse(jsonMatch[0]);
      return result.value;
    } catch (error) {
      console.error('LLM parsing error:', error);
      // Fallback to step-specific parser
      if (step.parser) {
        return step.parser(input);
      }
      return null;
    }
  };

  const moveToNextStep = () => {
    if (currentStepIndex < conversationSteps.length - 1) {
      const nextIndex = currentStepIndex + 1;
      setCurrentStepIndex(nextIndex);
      
      const nextStep = conversationSteps[nextIndex];
      const botMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'bot',
        message: nextStep.question,
        timestamp: new Date()
      };
      
      setChatMessages(prev => [...prev, botMessage]);
    } else {
      // All questions completed
      const completionMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'bot',
        message: "Perfect! I've gathered all your information. You can review and modify any details in the form on the right, then click 'Create Account' to finish setting up your profile.",
        timestamp: new Date()
      };
      
      setChatMessages(prev => [...prev, completionMessage]);
    }
  };

  const handleChatSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim() || currentStepIndex >= conversationSteps.length) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      message: chatInput.trim(),
      timestamp: new Date()
    };

    setChatMessages(prev => [...prev, userMessage]);
    setChatLoading(true);

    try {
      const currentStep = conversationSteps[currentStepIndex];
      
      // Parse the user input for the current step
      const extractedValue = await parseUserInputWithLLM(chatInput, currentStep);
      
      if (extractedValue !== null && extractedValue !== undefined) {
        // Validate the extracted value
        if (currentStep.validation && !currentStep.validation(extractedValue)) {
          throw new Error('Invalid value');
        }
        
        // Update form data
        setFormData(prev => ({ ...prev, [currentStep.field]: extractedValue }));
        
        // Confirm the update
        const confirmationMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          message: `Got it! I've set your ${currentStep.field.replace(/_/g, ' ')} as: ${extractedValue}`,
          timestamp: new Date()
        };
        
        setChatMessages(prev => [...prev, confirmationMessage]);
        
        // Move to next step after a brief delay
        setTimeout(() => {
          moveToNextStep();
        }, 500);
        
      } else {
        // Could not parse the input
        const errorMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          message: `I couldn't understand your response. ${currentStep.question}${currentStep.options ? ` Please choose from: ${currentStep.options.join(', ')}` : ''}`,
          timestamp: new Date()
        };
        
        setChatMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Chat processing error:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        message: "I had trouble understanding that. Could you please try rephrasing your answer?",
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
        {/* Chatbot Panel */}
        <div className="w-96 h-fit bg-white rounded-lg shadow-2xl border flex flex-col">
          {/* Chat Header */}
          <div className="bg-blue-600 text-white px-4 py-3 rounded-t-lg flex items-center gap-2">
            <Bot className="w-5 h-5" />
            <span className="font-medium">
              Form Assistant ({currentStepIndex + 1}/{conversationSteps.length})
            </span>
          </div>
          
          {/* Progress Bar */}
          <div className="bg-gray-200 h-1">
            <div 
              className="bg-blue-600 h-1 transition-all duration-300"
              style={{ width: `${((currentStepIndex + 1) / conversationSteps.length) * 100}%` }}
            />
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
                placeholder={currentStepIndex >= conversationSteps.length ? "All questions completed!" : "Your answer..."}
                className="flex-1 px-3 py-2 border rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                disabled={chatLoading || currentStepIndex >= conversationSteps.length}
              />
              <button
                type="submit"
                disabled={chatLoading || !chatInput.trim() || currentStepIndex >= conversationSteps.length}
                className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white p-2 rounded-md transition-colors"
              >
                <Send className="w-4 h-4" />
              </button>
            </div>
          </form>
        </div>

        {/* Form Card */}
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