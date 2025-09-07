import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageCircle, Send, Bot, User, Loader2 } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { dataService } from "@/services/dataService";

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
}

interface ChatbotPageProps {
  user: any;
}

export function ChatbotPage({ user }: ChatbotPageProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'bot',
      content: `Hello! I'm your AI superannuation advisor. I can help you understand your retirement projections, analyze your portfolio, and answer questions about your financial goals. What would you like to know?`,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages, isLoading]);

  const sampleQuestions = [
    "What is my risk category?",
    "What if I increase my monthly contribution by $200?",
    "How much will I retire with?",
    "How do I compare to others my age?",
    "Am I on track for retirement?",
    "Should I consider changing my risk profile?"
  ];

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

    try {
      // Call the real AI API
      const response = await dataService.sendChatMessage(user.User_ID, inputValue);
      
      const botResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response || "I'm sorry, I couldn't process your request at the moment. Please try again.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botResponse]);
    } catch (error) {
      const botResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: "I'm sorry, I'm having trouble connecting to the AI service. Please make sure the backend is running.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botResponse]);
    } finally {
      setIsLoading(false);
    }
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
            <ScrollArea ref={scrollAreaRef} className="h-full">
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