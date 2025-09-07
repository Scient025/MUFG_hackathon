import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageCircle, Send, Bot, User, Loader2 } from "lucide-react";
import { useState } from "react";

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
}

interface ChatbotPageProps {
  user: any;
}

export function ChatbotPage({ user, onGoalChange }: any) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'bot',
      content: `Hello ${user.name}! I'm your AI superannuation advisor. I can help you understand your retirement projections, analyze your portfolio, and answer questions about your financial goals. What would you like to know?`,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sampleQuestions = [
    "What if I increase my monthly contribution by $200?",
    "How safe is my portfolio if markets dip 20%?",
    "Should I consider changing my risk profile?",
    "What are the tax benefits of my superannuation?",
    "How does my portfolio compare to others my age?",
    "What's the best withdrawal strategy for retirement?"
  ];

  const simulateAIResponse = (userMessage: string): string => {
    const lowerMessage = userMessage.toLowerCase();
    
    // Simple keyword-based responses (in a real app, this would call ML models)
    if (lowerMessage.includes('contribution') || lowerMessage.includes('increase')) {
      return `Great question! If you increase your monthly contribution by $200, your projected retirement balance would grow from $${user.projectedPensionAmount.toLocaleString()} to approximately $${(user.projectedPensionAmount + 200 * 12 * (65 - user.age)).toLocaleString()}. This extra $2,400 annually could significantly boost your retirement income. Based on your current age of ${user.age}, this additional contribution would compound over ${65 - user.age} years.`;
    }
    
    if (lowerMessage.includes('market') || lowerMessage.includes('dip') || lowerMessage.includes('crash')) {
      return `Market volatility is a valid concern. With your current ${user.riskProfile} risk profile, a 20% market dip would reduce your portfolio by approximately $${(user.currentSavings * 0.2).toLocaleString()}. However, your diversified portfolio across ${user.investmentType.join(', ')} helps mitigate risk. Historically, markets recover over time, and your long-term strategy should weather short-term volatility. Consider your time horizon - you have ${65 - user.age} years until retirement.`;
    }
    
    if (lowerMessage.includes('risk') || lowerMessage.includes('profile')) {
      return `Your current risk profile is ${user.riskProfile}. This means you're ${user.riskProfile === 'Low' ? 'prioritizing capital preservation with lower volatility' : user.riskProfile === 'Medium' ? 'balancing growth and stability' : 'focusing on growth with higher volatility'}. Given your age of ${user.age}, this seems appropriate. If you're ${user.age < 50 ? 'younger' : 'older'}, you might consider ${user.age < 50 ? 'increasing' : 'reducing'} risk for ${user.age < 50 ? 'higher growth potential' : 'more stability'}.`;
    }
    
    if (lowerMessage.includes('tax') || lowerMessage.includes('benefit')) {
      return `Your superannuation offers excellent tax benefits! ${user.taxBenefitsEligibility ? 'You are eligible for concessional contributions taxed at only 15% (vs your marginal tax rate).' : 'While you may not be eligible for all tax benefits, your superannuation still offers tax advantages.'} Investment earnings are taxed at a maximum of 15%, and withdrawals after age 60 are generally tax-free. This makes superannuation one of the most tax-effective ways to save for retirement.`;
    }
    
    if (lowerMessage.includes('compare') || lowerMessage.includes('peer') || lowerMessage.includes('others')) {
      return `Compared to others your age, you're doing well! Your current balance of $${user.currentSavings.toLocaleString()} puts you in a strong position. You're contributing more than most people in your age group, and your projected payout of $${user.expectedAnnualPayout.toLocaleString()} annually would provide a comfortable retirement. Your investment strategy with ${user.investmentType.join(' and ')} is diversified and appropriate for your risk profile.`;
    }
    
    if (lowerMessage.includes('withdrawal') || lowerMessage.includes('strategy')) {
      return `For retirement withdrawals, consider these strategies: 1) Fixed withdrawal strategy - withdraw a fixed amount annually (e.g., 4% of your balance), 2) Dynamic strategy - adjust based on market conditions, 3) Bucket strategy - separate assets by time horizon. Given your ${user.riskProfile} risk profile and expected balance of $${user.projectedPensionAmount.toLocaleString()}, a dynamic strategy might work well. Start with 4-5% annual withdrawals and adjust based on market performance.`;
    }
    
    // Default response
    return `That's an interesting question about your superannuation! Based on your profile, I can see you're ${user.age} years old with a ${user.riskProfile} risk tolerance and a current balance of $${user.currentSavings.toLocaleString()}. Your projected retirement balance is $${user.projectedPensionAmount.toLocaleString()}. For more specific advice, I'd recommend consulting with a qualified financial advisor who can provide personalized recommendations based on your complete financial situation.`;
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    // Simulate AI processing time
    setTimeout(() => {
      const botResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: simulateAIResponse(inputValue),
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botResponse]);
      setIsLoading(false);
    }, 1500);
  };

  const handleSampleQuestion = (question: string) => {
    setInputValue(question);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="space-y-8">
      {/* Chat Interface */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground flex items-center gap-3">
            <MessageCircle className="w-7 h-7" />
            AI Superannuation Advisor
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 border border-border rounded-xl overflow-hidden">
            <ScrollArea className="h-full">
              <div className="p-4 space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {message.type === 'bot' && (
                      <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                        <Bot className="w-4 h-4 text-primary-foreground" />
                      </div>
                    )}
                    
                    <div className={`max-w-[80%] ${message.type === 'user' ? 'order-first' : ''}`}>
                      <div
                        className={`p-4 rounded-xl text-lg ${
                          message.type === 'user'
                            ? 'bg-primary text-primary-foreground ml-auto'
                            : 'bg-muted'
                        }`}
                      >
                        {message.content}
                      </div>
                      <div className={`text-xs text-muted-foreground mt-1 ${
                        message.type === 'user' ? 'text-right' : 'text-left'
                      }`}>
                        {formatTime(message.timestamp)}
                      </div>
                    </div>
                    
                    {message.type === 'user' && (
                      <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center flex-shrink-0">
                        <User className="w-4 h-4" />
                      </div>
                    )}
                  </div>
                ))}
                
                {isLoading && (
                  <div className="flex gap-3 justify-start">
                    <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                      <Bot className="w-4 h-4 text-primary-foreground" />
                    </div>
                    <div className="p-4 bg-muted rounded-xl">
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span className="text-lg">AI is thinking...</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </ScrollArea>
          </div>
          
          <div className="mt-4 flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask me anything about your superannuation..."
              className="flex-1 h-12 text-lg"
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            />
            <Button 
              onClick={handleSendMessage} 
              disabled={!inputValue.trim() || isLoading}
              className="h-12 px-6"
            >
              <Send className="w-5 h-5" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Sample Questions */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-xl font-bold text-card-foreground">
            Try These Sample Questions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-3">
            {sampleQuestions.map((question, index) => (
              <Button
                key={index}
                variant="outline"
                onClick={() => handleSampleQuestion(question)}
                className="h-auto p-4 text-left justify-start text-lg hover:bg-muted"
              >
                <MessageCircle className="w-4 h-4 mr-2 flex-shrink-0" />
                {question}
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* AI Capabilities */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-xl font-bold text-card-foreground">
            What I Can Help You With
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div className="p-4 bg-muted/30 rounded-xl">
              <h4 className="font-semibold text-lg mb-2">Retirement Projections</h4>
              <p className="text-muted-foreground">
                Calculate how different contribution amounts affect your retirement balance
              </p>
            </div>
            
            <div className="p-4 bg-muted/30 rounded-xl">
              <h4 className="font-semibold text-lg mb-2">Risk Analysis</h4>
              <p className="text-muted-foreground">
                Understand how market changes might impact your portfolio
              </p>
            </div>
            
            <div className="p-4 bg-muted/30 rounded-xl">
              <h4 className="font-semibold text-lg mb-2">Tax Optimization</h4>
              <p className="text-muted-foreground">
                Explain tax benefits and strategies for your superannuation
              </p>
            </div>
            
            <div className="p-4 bg-muted/30 rounded-xl">
              <h4 className="font-semibold text-lg mb-2">Portfolio Comparison</h4>
              <p className="text-muted-foreground">
                Compare your strategy with others in your age and risk group
              </p>
            </div>
            
            <div className="p-4 bg-muted/30 rounded-xl">
              <h4 className="font-semibold text-lg mb-2">Withdrawal Strategies</h4>
              <p className="text-muted-foreground">
                Recommend the best approach for accessing your superannuation
              </p>
            </div>
            
            <div className="p-4 bg-muted/30 rounded-xl">
              <h4 className="font-semibold text-lg mb-2">Goal Planning</h4>
              <p className="text-muted-foreground">
                Help you plan and prioritize your financial goals
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
