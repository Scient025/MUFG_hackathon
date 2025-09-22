import { useState, useMemo, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from '@/components/ui/slider';
import { SignupData, dataService } from "@/services/dataService";
import { Loader2, CheckCircle, Calculator, ArrowLeft, Mic, MicOff, Volume2, VolumeX } from "lucide-react";
import { Send, Bot, TrendingUp, DollarSign, Target, AlertCircle } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { useToast } from "@/hooks/use-toast";

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
      id: 'username',
      question: "What's your email address? This will be used as your username.",
      field: 'username',
      type: 'text',
      validation: (value: string) => /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)
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
    },
    {
      id: 'years_contributed',
      question: "How many years have you been contributing to your superannuation?",
      field: 'years_contributed',
      type: 'number',
      validation: (value: number) => value >= 0 && value <= 50,
      parser: (input: string) => {
        const match = input.match(/(\d+)/);
        return match ? parseInt(match[1]) : null;
      }
    },
    {
      id: 'investment_type',
      question: "What type of investment strategy do you prefer?",
      field: 'investment_type',
      type: 'select',
      options: ['Stocks', 'Bonds', 'Mutual Fund', 'ETF', 'Real Estate', 'Balanced']
    },
    {
      id: 'fund_name',
      question: "What's the name of your preferred superannuation fund?",
      field: 'fund_name',
      type: 'select',
      options: [
        'Jackson-Ryan',
        'Mclaughlin, Lopez and Hunt', 
        'Hernandez-Allen',
        'Price-Robles',
        'Kelly, Gonzales and Wagner',
        'Oconnell, Welch and Lopez',
        'White Group',
        'Johnson, Brown and Griffin',
        'Lane-Jennings',
        'Jenkins Inc',
        'Nash-Guerrero',
        'Smith, Mitchell and Burch',
        'Zhang-Scott',
        'Brown, Thompson and Davies',
        'Hensley, Hale and Morales',
        'Mullins, Dixon and Richardson',
        'Cox, Compton and Rhodes',
        'Hayes Group',
        'Simon Inc',
        'Howell and Sons',
        'Other (Specify)'
      ]
    },
    {
      id: 'debt_level',
      question: "What's your current debt level?",
      field: 'debt_level',
      type: 'select',
      options: ['Low', 'Medium', 'High']
    },
    {
      id: 'savings_rate',
      question: "What percentage of your income do you typically save? (e.g., '15%' or '15')",
      field: 'savings_rate',
      type: 'number',
      validation: (value: number) => value >= 0 && value <= 100,
      parser: (input: string) => {
        const match = input.match(/(\d+(?:\.\d+)?)/);
        return match ? parseFloat(match[1]) : null;
      }
    },
    {
      id: 'monthly_expenses',
      question: "What are your average monthly expenses? (You can say something like '$2,500' or '2.5k')",
      field: 'monthly_expenses',
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
      id: 'password',
      question: "Finally, create a password for your account (at least 6 characters):",
      field: 'password',
      type: 'text',
      validation: (value: string) => value.length >= 6
    }
  ];

  const [formData, setFormData] = useState<SignupData>({
    name: "",
    username: "", 
    password: "",
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
    years_contributed: 19, // Median from CSV: 18.5 rounded up
    investment_type: "Stocks", // Most common from CSV
    fund_name: "Jackson-Ryan", // Most common fund name from CSV
    marital_status: "Single",
    number_of_dependents: 0,
    education_level: "PhD", // Most common from CSV
    health_status: "Good", // Most common from CSV
    home_ownership_status: "Rent", // Most common from CSV
    investment_experience_level: "Beginner",
    financial_goals: "Retirement",
    insurance_coverage: "Yes", // Most common from CSV
    pension_type: "Defined Contribution", // Most common from CSV
    withdrawal_strategy: "Bucket", // Most common from CSV
    // Additional fields used by ML models
    debt_level: "Low",
    savings_rate: 0.15, // 15% savings rate
    portfolio_diversity_score: 0.6, // Moderate diversity
    monthly_expenses: 2500, // Estimated based on income
    transaction_amount: 0,
    transaction_pattern_score: 0.5,
    anomaly_score: 0.1,
    suspicious_flag: "No"
  });

  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [showCalculator, setShowCalculator] = useState(false);
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

  // Azure Speech Services state
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [voices, setVoices] = useState<any[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('en-AU-NatashaNeural');
  const [currentPlayingMessage, setCurrentPlayingMessage] = useState<string | null>(null);
  const [showAdvancedFields, setShowAdvancedFields] = useState(false);
  const [voiceGuidedMode, setVoiceGuidedMode] = useState(false);

  // Refs
  const chatContainerRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // Calculator inputs derived from form data
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

  const handleInputChange = (field: keyof SignupData, value: string | number) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  // Function to scroll chat to bottom
  const scrollChatToBottom = () => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  };

  // Load available voices on component mount
  useEffect(() => {
    loadVoices();
  }, []);

  // Auto-scroll chat to bottom when new messages are added
  useEffect(() => {
    // Small delay to ensure DOM has updated
    setTimeout(() => {
      scrollChatToBottom();
    }, 100);
  }, [chatMessages]);
  const { toast } = useToast();
  const loadVoices = async () => {
    try {
      const response = await fetch(`${(import.meta.env.VITE_API_URL || '/api').replace(/\/+$/, '')}/voices`);
      const data = await response.json();
      if (data.enabled) {
        setVoices(data.voices);
      } else {
        setSpeechEnabled(false);
        toast({
          title: "Speech Services Disabled",
          description: "Azure Speech Services are not configured. Text-only mode enabled.",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Error loading voices:', error);
      setSpeechEnabled(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      
      // Try to use WAV format, fallback to webm if not supported
      let mimeType = 'audio/wav';
      if (!MediaRecorder.isTypeSupported('audio/wav')) {
        mimeType = 'audio/webm';
      }
      
      const mediaRecorder = new MediaRecorder(stream, { mimeType });
      mediaRecorderRef.current = mediaRecorder;
      
      const audioChunks: Blob[] = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: mimeType });
        await processAudioInput(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);
      
      toast({
        title: "Recording Started",
        description: "Speak your answer now...",
      });
    } catch (error) {
      console.error('Error starting recording:', error);
      toast({
        title: "Recording Failed",
        description: "Could not access microphone. Please check permissions.",
        variant: "destructive",
      });
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const processAudioInput = async (audioBlob: Blob) => {
    try {
      const arrayBuffer = await audioBlob.arrayBuffer();
      const uint8Array = new Uint8Array(arrayBuffer);
      const hexString = Array.from(uint8Array).map(b => b.toString(16).padStart(2, '0')).join('');
      
      const response = await fetch(`${(import.meta.env.VITE_API_URL || '/api').replace(/\/+$/, '')}/speech-to-text`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          audio_data: hexString,
          language: 'en-AU'
        }),
      });
      
      const result = await response.json();
      
      if (result.success) {
        // Process the recognized text with the current step
        if (voiceGuidedMode && currentStepIndex < conversationSteps.length) {
          const currentStep = conversationSteps[currentStepIndex];
          const extractedValue = await parseUserInputWithLLM(result.text, currentStep);
          
          if (extractedValue !== null && extractedValue !== undefined) {
            if (currentStep.validation && !currentStep.validation(extractedValue)) {
              throw new Error('Invalid value');
            }
            
            setFormData(prev => ({ ...prev, [currentStep.field]: extractedValue }));
            
            const confirmationMessage: ChatMessage = {
              id: (Date.now() + 1).toString(),
              type: 'bot',
              message: `Got it! I've set your ${currentStep.field.replace(/_/g, ' ')} as: ${extractedValue}`,
              timestamp: new Date()
            };
            
            setChatMessages(prev => [...prev, confirmationMessage]);
            
            if (speechEnabled) {
              textToSpeech(confirmationMessage.message, confirmationMessage.id);
            }
            
            setTimeout(() => {
              moveToNextStep();
            }, 1000);
          } else {
            const errorMessage: ChatMessage = {
              id: (Date.now() + 1).toString(),
              type: 'bot',
              message: `I couldn't understand your response. ${currentStep.question}${currentStep.options ? ` Please choose from: ${currentStep.options.join(', ')}` : ''}`,
              timestamp: new Date()
            };
            
            setChatMessages(prev => [...prev, errorMessage]);
            
            if (speechEnabled) {
              textToSpeech(errorMessage.message, errorMessage.id);
            }
          }
        } else {
          // Regular chat mode
          setChatInput(result.text);
        }
        
        toast({
          title: "Speech Recognized",
          description: `"${result.text}"`,
        });
      } else {
        toast({
          title: "Speech Recognition Failed",
          description: "Could not understand your speech. Please try again.",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Error processing audio:', error);
      toast({
        title: "Error",
        description: "Failed to process speech input.",
        variant: "destructive",
      });
    }
  };

  const stopAudioPlayback = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setIsPlaying(false);
      setCurrentPlayingMessage(null);
    }
  };

  const textToSpeech = async (text: string, messageId?: string) => {
    if (!speechEnabled) return;
    
    // Stop any currently playing audio
    stopAudioPlayback();
    
    try {
      const response = await fetch(`${(import.meta.env.VITE_API_URL || '/api').replace(/\/+$/, '')}/text-to-speech`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text,
          voice_name: selectedVoice
        }),
      });
      
      const result = await response.json();
      
      if (result.success) {
        const audioData = new Uint8Array(result.audio_data.match(/.{1,2}/g)!.map(byte => parseInt(byte, 16)));
        const audioBlob = new Blob([audioData], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        if (audioRef.current) {
          audioRef.current.src = audioUrl;
          audioRef.current.play();
          setIsPlaying(true);
          setCurrentPlayingMessage(messageId || null);
          
          audioRef.current.onended = () => {
            setIsPlaying(false);
            setCurrentPlayingMessage(null);
            URL.revokeObjectURL(audioUrl);
          };
        }
      }
    } catch (error) {
      console.error('Error with text-to-speech:', error);
    }
  };

  const startVoiceGuidedMode = () => {
    setVoiceGuidedMode(true);
    setCurrentStepIndex(0);
    setChatMessages([
      {
        id: '1',
        type: 'bot',
        message: "Hi! I'm here to help you set up your superannuation profile with voice assistance. I'll ask you questions one by one, and you can speak your answers. Let's get started!",
        timestamp: new Date()
      },
      {
        id: '2',
        type: 'bot',
        message: conversationSteps[0].question,
        timestamp: new Date()
      }
    ]);
    
    // Speak the first question
    if (speechEnabled) {
      textToSpeech(conversationSteps[0].question, '2');
    }
  };

  const stopVoiceGuidedMode = () => {
    setVoiceGuidedMode(false);
    stopAudioPlayback();
    if (isRecording) {
      stopRecording();
    }
  };

  const handleCalculatorInputChange = (key: keyof CalculatorInputs, value: number) => {
    setCalculatorInputs(prev => ({ ...prev, [key]: value }));
  };

  // Update calculator inputs when form data changes
  const updateCalculatorFromForm = () => {
    setCalculatorInputs(prev => ({
      ...prev,
      currentAge: formData.age,
      retirementAge: formData.retirement_age_goal,
      currentBalance: formData.current_savings,
      monthlyContribution: formData.contribution_amount,
      employerContribution: formData.employer_contribution,
      annualReturnRate: formData.risk_tolerance === 'High' ? 9 : formData.risk_tolerance === 'Low' ? 6 : 7.5,
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      const result = await dataService.signupUser(formData);
      
      // Store user data in localStorage for the retirement calculator to access
      localStorage.setItem(`user_${result.userId}`, JSON.stringify(formData));
      
      setSuccess(true);
      setTimeout(() => {
        // Call onSignupSuccess with the userId to navigate to dashboard
        onSignupSuccess(result.userId);
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
      console.log('Calling Gemini API for field:', step.field, 'with input:', input);
      
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
        console.error('Gemini API request failed:', response.status, response.statusText);
        throw new Error(`API request failed: ${response.statusText}`);
      }

      const data = await response.json();
      console.log('Gemini API response:', data);
      
      const generatedText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      
      if (!generatedText) {
        console.error('No generated text from Gemini API');
        throw new Error('No response from API');
      }

      const jsonMatch = generatedText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        console.error('No JSON found in Gemini response:', generatedText);
        throw new Error('No valid JSON found in response');
      }

      const result = JSON.parse(jsonMatch[0]);
      console.log('Parsed result from Gemini:', result);
      return result.value;
    } catch (error) {
      console.error('LLM parsing error for field', step.field, ':', error);
      console.log('Falling back to parser function');
      if (step.parser) {
        const fallbackResult = step.parser(input);
        console.log('Fallback parser result:', fallbackResult);
        return fallbackResult;
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
      
      // Speak the next question in voice guided mode
      if (voiceGuidedMode && speechEnabled) {
        textToSpeech(nextStep.question, botMessage.id);
      }
    } else {
      const completionMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'bot',
        message: "Perfect! I've gathered all your information. You can review and modify any details in the form on the right, then click 'Create Account' to finish setting up your profile and see your retirement projections!",
        timestamp: new Date()
      };
      
      setChatMessages(prev => [...prev, completionMessage]);
      
      // Speak the completion message in voice guided mode
      if (voiceGuidedMode && speechEnabled) {
        textToSpeech(completionMessage.message, completionMessage.id);
      }
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
      
      const extractedValue = await parseUserInputWithLLM(chatInput, currentStep);
      
      if (extractedValue !== null && extractedValue !== undefined) {
        if (currentStep.validation && !currentStep.validation(extractedValue)) {
          throw new Error('Invalid value');
        }
        
        setFormData(prev => ({ ...prev, [currentStep.field]: extractedValue }));
        
        const confirmationMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          message: `Got it! I've set your ${currentStep.field.replace(/_/g, ' ')} as: ${extractedValue}`,
          timestamp: new Date()
        };
        
        setChatMessages(prev => [...prev, confirmationMessage]);
        
        setTimeout(() => {
          moveToNextStep();
        }, 500);
        
      } else {
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

  // Success screen
  if (success) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <Card className="w-full max-w-md">
          <CardContent className="pt-6">
            <div className="text-center">
              <CheckCircle className="w-16 h-16 text-green-500 mx-auto mb-4" />
              <h2 className="text-2xl font-bold text-green-700 mb-2">Signup Successful!</h2>
              <p className="text-gray-600">Welcome to SuperWise! Loading your retirement calculator...</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Calculator screen
  if (showCalculator) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
        <div className="max-w-7xl mx-auto space-y-6">
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 mb-2">Welcome to SuperWise, {formData.name}!</h1>
              <p className="text-xl text-gray-600">Here's your personalized retirement projection</p>
            </div>
            <Button 
              onClick={() => onSignupSuccess('user_123')}
              className="flex items-center gap-2"
            >
              Continue to Dashboard
            </Button>
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
                        <p><strong>Risk Tolerance:</strong> {formData.risk_tolerance} - Your projected return rate is {calculatorInputs.annualReturnRate}%</p>
                        <p><strong>Investment Experience:</strong> {formData.investment_experience_level} - Consider reviewing investment options</p>
                        <p><strong>Time Horizon:</strong> {calculatorInputs.retirementAge - calculatorInputs.currentAge} years - Time is your biggest advantage</p>
                        {formData.number_of_dependents > 0 && (
                          <p><strong>Dependents:</strong> {formData.number_of_dependents} - Consider life insurance and education planning</p>
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

  // Main signup form
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
          
          {/* Speech Controls */}
          {speechEnabled && (
            <div className="bg-green-600 text-white px-4 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Volume2 className="w-4 h-4" />
                <span className="text-sm font-medium">Speech Controls</span>
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  className="text-xs bg-green-700 text-white border-0 rounded px-2 py-1"
                  aria-label="Select voice for text-to-speech"
                  disabled={!speechEnabled}
                >
                  {Object.entries(voices).map(([lang, voiceList]: [string, any]) => (
                    <optgroup key={lang} label={lang.toUpperCase()}>
                      {Array.isArray(voiceList) && voiceList.map((voice: any) => (
                        <option key={voice.name} value={voice.name}>
                          {voice.display}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                </select>
                <Button
                  onClick={isRecording ? stopRecording : startRecording}
                  variant={isRecording ? "destructive" : "default"}
                  size="sm"
                  className="h-6 px-2"
                  disabled={!speechEnabled}
                >
                  {isRecording ? <MicOff className="w-3 h-3" /> : <Mic className="w-3 h-3" />}
                </Button>
                {isPlaying && (
                  <Button
                    onClick={stopAudioPlayback}
                    variant="destructive"
                    size="sm"
                    className="h-6 px-2"
                  >
                    <VolumeX className="w-3 h-3" />
                  </Button>
                )}
                <Button
                  onClick={() => setSpeechEnabled(!speechEnabled)}
                  variant="outline"
                  size="sm"
                  className="h-6 px-2 text-white border-white hover:bg-white hover:text-green-600"
                >
                  {speechEnabled ? <VolumeX className="w-3 h-3" /> : <Volume2 className="w-3 h-3" />}
                </Button>
              </div>
            </div>
          )}
          
          {/* Voice Guided Mode Controls */}
          <div className="bg-gray-50 px-4 py-2 border-b">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Voice Guided Mode</span>
              {!voiceGuidedMode ? (
                <Button
                  onClick={startVoiceGuidedMode}
                  size="sm"
                  className="h-6 px-3 text-xs"
                  disabled={!speechEnabled}
                >
                  Start Voice Guide
                </Button>
              ) : (
                <Button
                  onClick={stopVoiceGuidedMode}
                  variant="destructive"
                  size="sm"
                  className="h-6 px-3 text-xs"
                >
                  Stop Voice Guide
                </Button>
              )}
            </div>
            {voiceGuidedMode && (
              <p className="text-xs text-gray-600 mt-1">
                Questions will be read aloud. Speak your answers after each question.
              </p>
            )}
          </div>
          
          {/* Progress Bar */}
          <div className="bg-gray-200 h-1">
            <div 
              className="bg-blue-600 h-1 transition-all duration-300"
              style={{ width: `${((currentStepIndex + 1) / conversationSteps.length) * 100}%` } as React.CSSProperties}
            />
          </div>
          
          {/* Chat Messages */}
          <div ref={chatContainerRef} className="h-96 overflow-y-auto p-4 space-y-3">
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
                  {message.type === 'bot' && speechEnabled && (
                    <div className="mt-1 flex justify-end">
                      <button
                        onClick={() => textToSpeech(message.message, message.id)}
                        className="p-1 hover:bg-gray-200 rounded-full transition-colors"
                        title="Read this message"
                        disabled={isPlaying && currentPlayingMessage === message.id}
                      >
                        {isPlaying && currentPlayingMessage === message.id ? (
                          <VolumeX className="w-3 h-3 text-red-500" />
                        ) : (
                          <Volume2 className="w-3 h-3 text-gray-500 hover:text-gray-700" />
                        )}
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {chatLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 px-3 py-2 rounded-lg">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' } as React.CSSProperties}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' } as React.CSSProperties}></div>
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
                title="Send message"
                aria-label="Send message"
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
              {/* Essential Fields */}
              <div className="space-y-6">
                <h3 className="text-xl font-semibold text-gray-800 border-b pb-2">Essential Information</h3>
                
                {/* Personal Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="name" className="text-lg font-medium">Full Name *</Label>
                    <Input
                      id="name"
                      value={formData.name}
                      onChange={(e) => handleInputChange("name", e.target.value)}
                      className="text-lg h-12"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="username" className="text-lg font-medium">Email (Username) *</Label>
                    <Input
                      id="username"
                      type="email"
                      value={formData.username}
                      onChange={(e) => handleInputChange("username", e.target.value)}
                      className="text-lg h-12"
                      required
                    />
                  </div>
                  <div>
                    <Label htmlFor="age" className="text-lg font-medium">Age *</Label>
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
                  <div>
                    <Label className="text-lg font-medium">Gender *</Label>
                    <Select value={formData.gender} onValueChange={(value) => handleInputChange("gender", value)}>
                      <SelectTrigger className="h-12 text-lg bg-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-white">
                        <SelectItem value="Male">Male</SelectItem>
                        <SelectItem value="Female">Female</SelectItem>
                        <SelectItem value="Other">Other</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label className="text-lg font-medium">Country *</Label>
                    <Select value={formData.country} onValueChange={(value) => handleInputChange("country", value)}>
                      <SelectTrigger className="h-12 text-lg bg-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-white">
                        <SelectItem value="Australia">Australia</SelectItem>
                        <SelectItem value="New Zealand">New Zealand</SelectItem>
                        <SelectItem value="United Kingdom">United Kingdom</SelectItem>
                        <SelectItem value="United States">United States</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label htmlFor="annual_income" className="text-lg font-medium">Annual Income ($) *</Label>
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
                </div>

                {/* Financial Information */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="current_savings" className="text-lg font-medium">Current Savings ($) *</Label>
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
                  <div>
                    <Label htmlFor="contribution_amount" className="text-lg font-medium">Monthly Contribution ($) *</Label>
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
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="employer_contribution" className="text-lg font-medium">Employer Contribution ($) *</Label>
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
                  <div>
                    <Label className="text-lg font-medium">Risk Tolerance *</Label>
                    <Select value={formData.risk_tolerance} onValueChange={(value) => handleInputChange("risk_tolerance", value)}>
                      <SelectTrigger className="h-12 text-lg bg-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-white">
                        <SelectItem value="Low">Low Risk</SelectItem>
                        <SelectItem value="Medium">Medium Risk</SelectItem>
                        <SelectItem value="High">High Risk</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label className="text-lg font-medium">Investment Experience *</Label>
                    <Select value={formData.investment_experience_level} onValueChange={(value) => handleInputChange("investment_experience_level", value)}>
                      <SelectTrigger className="h-12 text-lg bg-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-white">
                        <SelectItem value="Beginner">Beginner</SelectItem>
                        <SelectItem value="Intermediate">Intermediate</SelectItem>
                        <SelectItem value="Expert">Expert</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <Label className="text-lg font-medium">Marital Status *</Label>
                    <Select value={formData.marital_status} onValueChange={(value) => handleInputChange("marital_status", value)}>
                      <SelectTrigger className="h-12 text-lg bg-white">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent className="bg-white">
                        <SelectItem value="Single">Single</SelectItem>
                        <SelectItem value="Married">Married</SelectItem>
                        <SelectItem value="Divorced">Divorced</SelectItem>
                        <SelectItem value="Widowed">Widowed</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label htmlFor="number_of_dependents" className="text-lg font-medium">Number of Dependents *</Label>
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
                  <div>
                    <Label htmlFor="password" className="text-lg font-medium">Password *</Label>
                    <Input
                      id="password"
                      type="password"
                      value={formData.password}
                      onChange={(e) => handleInputChange("password", e.target.value)}
                      className="text-lg h-12"
                      placeholder="Create a secure password (min 6 characters)"
                      required
                      minLength={6}
                    />
                  </div>
                </div>
              </div>

              {/* See More Button */}
              <div className="flex justify-center">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowAdvancedFields(!showAdvancedFields)}
                  className="px-8 py-3 text-lg"
                >
                  {showAdvancedFields ? 'Show Less' : 'See More Fields (Optional)'}
                </Button>
              </div>

              {/* Advanced Fields */}
              {showAdvancedFields && (
                <div className="space-y-6 border-t pt-6">
                  <h3 className="text-xl font-semibold text-gray-800 border-b pb-2">Additional Information (Optional)</h3>
                  <p className="text-sm text-gray-600 mb-4">
                    These fields help improve our ML models and provide more personalized recommendations. All fields are optional.
                  </p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="retirement_age_goal" className="text-lg font-medium">Retirement Age Goal</Label>
                      <Input
                        id="retirement_age_goal"
                        type="number"
                        value={formData.retirement_age_goal}
                        onChange={(e) => handleInputChange("retirement_age_goal", parseInt(e.target.value))}
                        className="text-lg h-12"
                        min="50"
                        max="80"
                      />
                    </div>
                    <div>
                      <Label htmlFor="years_contributed" className="text-lg font-medium">Years Contributed</Label>
                      <Input
                        id="years_contributed"
                        type="number"
                        value={formData.years_contributed}
                        onChange={(e) => handleInputChange("years_contributed", parseInt(e.target.value))}
                        className="text-lg h-12"
                        min="0"
                        max="50"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-lg font-medium">Investment Type</Label>
                      <Select value={formData.investment_type} onValueChange={(value) => handleInputChange("investment_type", value)}>
                        <SelectTrigger className="h-12 text-lg bg-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white">
                          <SelectItem value="Stocks">Stocks</SelectItem>
                          <SelectItem value="Bonds">Bonds</SelectItem>
                          <SelectItem value="Mutual Fund">Mutual Fund</SelectItem>
                          <SelectItem value="ETF">ETF</SelectItem>
                          <SelectItem value="Real Estate">Real Estate</SelectItem>
                          <SelectItem value="Balanced">Balanced</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="text-lg font-medium">Financial Goals</Label>
                      <Select value={formData.financial_goals} onValueChange={(value) => handleInputChange("financial_goals", value)}>
                        <SelectTrigger className="h-12 text-lg bg-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white">
                          <SelectItem value="Retirement">Retirement</SelectItem>
                          <SelectItem value="Education">Education</SelectItem>
                          <SelectItem value="Home">Home</SelectItem>
                          <SelectItem value="Travel">Travel</SelectItem>
                          <SelectItem value="Other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-lg font-medium">Employment Status</Label>
                      <Select value={formData.employment_status} onValueChange={(value) => handleInputChange("employment_status", value)}>
                        <SelectTrigger className="h-12 text-lg bg-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white">
                          <SelectItem value="Full-time">Full-time</SelectItem>
                          <SelectItem value="Part-time">Part-time</SelectItem>
                          <SelectItem value="Unemployed">Unemployed</SelectItem>
                          <SelectItem value="Self-employed">Self-employed</SelectItem>
                          <SelectItem value="Retired">Retired</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="text-lg font-medium">Education Level</Label>
                      <Select value={formData.education_level} onValueChange={(value) => handleInputChange("education_level", value)}>
                        <SelectTrigger className="h-12 text-lg bg-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white">
                          <SelectItem value="High School">High School</SelectItem>
                          <SelectItem value="Bachelor">Bachelor</SelectItem>
                          <SelectItem value="Master">Master</SelectItem>
                          <SelectItem value="PhD">PhD</SelectItem>
                          <SelectItem value="Other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-lg font-medium">Health Status</Label>
                      <Select value={formData.health_status} onValueChange={(value) => handleInputChange("health_status", value)}>
                        <SelectTrigger className="h-12 text-lg bg-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white">
                          <SelectItem value="Excellent">Excellent</SelectItem>
                          <SelectItem value="Good">Good</SelectItem>
                          <SelectItem value="Fair">Fair</SelectItem>
                          <SelectItem value="Poor">Poor</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="text-lg font-medium">Home Ownership Status</Label>
                      <Select value={formData.home_ownership_status} onValueChange={(value) => handleInputChange("home_ownership_status", value)}>
                        <SelectTrigger className="h-12 text-lg bg-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white">
                          <SelectItem value="Own">Own</SelectItem>
                          <SelectItem value="Rent">Rent</SelectItem>
                          <SelectItem value="Other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-lg font-medium">Insurance Coverage</Label>
                      <Select value={formData.insurance_coverage} onValueChange={(value) => handleInputChange("insurance_coverage", value)}>
                        <SelectTrigger className="h-12 text-lg bg-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white">
                          <SelectItem value="Yes">Yes</SelectItem>
                          <SelectItem value="No">No</SelectItem>
                          <SelectItem value="Partial">Partial</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="text-lg font-medium">Pension Type</Label>
                      <Select value={formData.pension_type} onValueChange={(value) => handleInputChange("pension_type", value)}>
                        <SelectTrigger className="h-12 text-lg bg-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white">
                          <SelectItem value="Defined Contribution">Defined Contribution</SelectItem>
                          <SelectItem value="Defined Benefit">Defined Benefit</SelectItem>
                          <SelectItem value="Hybrid">Hybrid</SelectItem>
                          <SelectItem value="Other">Other</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label className="text-lg font-medium">Debt Level</Label>
                      <Select value={formData.debt_level} onValueChange={(value) => handleInputChange("debt_level", value)}>
                        <SelectTrigger className="h-12 text-lg bg-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent className="bg-white">
                          <SelectItem value="Low">Low</SelectItem>
                          <SelectItem value="Medium">Medium</SelectItem>
                          <SelectItem value="High">High</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label htmlFor="savings_rate" className="text-lg font-medium">Savings Rate (%)</Label>
                      <Input
                        id="savings_rate"
                        type="number"
                        value={formData.savings_rate}
                        onChange={(e) => handleInputChange("savings_rate", parseFloat(e.target.value))}
                        className="text-lg h-12"
                        min="0"
                        max="100"
                        step="0.1"
                      />
                    </div>
                    <div>
                      <Label htmlFor="monthly_expenses" className="text-lg font-medium">Monthly Expenses ($)</Label>
                      <Input
                        id="monthly_expenses"
                        type="number"
                        value={formData.monthly_expenses}
                        onChange={(e) => handleInputChange("monthly_expenses", parseFloat(e.target.value))}
                        className="text-lg h-12"
                        min="0"
                      />
                    </div>
                  </div>
                </div>
              )}

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
                    "Create Account & See Projections"
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
      
      {/* Hidden audio element for TTS */}
      <audio ref={audioRef} />
    </div>
  );
}