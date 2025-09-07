import { MessageCircle, X, Send, Bot, User, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState, useEffect, useRef } from "react";
import { dataService } from "@/services/dataService";

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
}

interface FloatingChatButtonProps {
  user: any;
}

export function FloatingChatButton({ user }: FloatingChatButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      type: 'bot',
      content: `Hi ${user?.User_ID || 'there'}! I'm your AI superannuation advisor. How can I help with your superannuation questions?`,
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatAreaRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (chatAreaRef.current) {
      chatAreaRef.current.scrollTop = chatAreaRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !user?.User_ID) return;

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

  const handleQuickQuestion = (question: string) => {
    setInputValue(question);
  };

  return (
    <>
      {/* Floating Chat Button */}
      <div className="fixed bottom-6 right-6 z-50">
        <Button
          onClick={() => setIsOpen(!isOpen)}
          className="w-16 h-16 rounded-full shadow-lg hover:shadow-xl transition-all duration-200 bg-gradient-to-r from-primary to-primary/80 text-primary-foreground hover:from-primary/90 hover:to-primary/70 hover:scale-110"
        >
          {isOpen ? (
            <X className="w-6 h-6" />
          ) : (
            <MessageCircle className="w-6 h-6" />
          )}
        </Button>
        
        {/* Floating label when not open */}
        {!isOpen && (
          <div className="absolute bottom-full right-0 mb-2 px-3 py-1 bg-neutral text-neutral-foreground text-sm rounded-lg shadow-lg opacity-0 hover:opacity-100 transition-opacity duration-200 whitespace-nowrap">
            Ask AI Advisor
          </div>
        )}
      </div>

      {/* Chat Panel */}
      {isOpen && (
        <div className="fixed bottom-24 right-6 w-96 h-[500px] bg-card border border-card-border rounded-xl shadow-xl z-40 flex flex-col">
          {/* Header */}
          <div className="p-4 border-b border-card-border">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center">
                <MessageCircle className="w-4 h-4" />
              </div>
              <div>
                <h3 className="font-semibold text-card-foreground">AI Advisor</h3>
                <p className="text-sm text-muted-foreground">How can I help with your super?</p>
              </div>
            </div>
          </div>

          {/* Chat Area */}
          <div ref={chatAreaRef} className="flex-1 p-4 overflow-y-auto">
            <div className="space-y-3">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex gap-2 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {message.type === 'bot' && (
                    <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                      <Bot className="w-3 h-3" />
                    </div>
                  )}
                  
                  <div className={`max-w-[80%] ${message.type === 'user' ? 'order-first' : ''}`}>
                    <div
                      className={`p-3 rounded-xl text-sm ${
                        message.type === 'user'
                          ? 'bg-primary text-primary-foreground ml-auto'
                          : 'bg-muted/50'
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                  
                  {message.type === 'user' && (
                    <div className="w-6 h-6 bg-muted rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                      <User className="w-3 h-3" />
                    </div>
                  )}
                </div>
              ))}
              
              {isLoading && (
                <div className="flex gap-2 justify-start">
                  <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                    <Bot className="w-3 h-3" />
                  </div>
                  <div className="p-3 bg-muted/50 rounded-xl">
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      <span className="text-sm">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-card-border">
            <div className="flex gap-2">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Type your question..."
                className="flex-1 px-3 py-2 border border-input rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                disabled={isLoading}
              />
              <Button 
                size="sm" 
                className="px-3"
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading}
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
            
            {/* Quick Questions */}
            <div className="flex flex-wrap gap-2 mt-3">
              <button 
                className="text-xs px-2 py-1 bg-muted hover:bg-muted/80 rounded-full transition-colors"
                onClick={() => handleQuickQuestion("How to increase super?")}
              >
                How to increase super?
              </button>
              <button 
                className="text-xs px-2 py-1 bg-muted hover:bg-muted/80 rounded-full transition-colors"
                onClick={() => handleQuickQuestion("What is my risk category?")}
              >
                Risk category
              </button>
              <button 
                className="text-xs px-2 py-1 bg-muted hover:bg-muted/80 rounded-full transition-colors"
                onClick={() => handleQuickQuestion("Tax benefits")}
              >
                Tax benefits
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 z-30" 
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}