import { MessageCircle, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState } from "react";

export function FloatingChatButton() {
  const [isOpen, setIsOpen] = useState(false);

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
        <div className="fixed bottom-24 right-6 w-80 h-96 bg-card border border-card-border rounded-xl shadow-xl z-40 flex flex-col">
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
          <div className="flex-1 p-4 overflow-y-auto">
            <div className="space-y-4">
              {/* AI Welcome Message */}
              <div className="flex gap-3">
                <div className="w-6 h-6 bg-primary text-primary-foreground rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                  <MessageCircle className="w-3 h-3" />
                </div>
                <div className="bg-muted/50 rounded-xl p-3 max-w-64">
                  <p className="text-sm">Hi Margaret! I'm here to help with your superannuation questions. You can ask me about:</p>
                  <ul className="text-xs mt-2 space-y-1 text-muted-foreground">
                    <li>• Your retirement projections</li>
                    <li>• Investment strategies</li>
                    <li>• Contribution options</li>
                    <li>• Government benefits</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Input Area */}
          <div className="p-4 border-t border-card-border">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Type your question..."
                className="flex-1 px-3 py-2 border border-input rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-ring"
              />
              <Button size="sm" className="px-3">
                <MessageCircle className="w-4 h-4" />
              </Button>
            </div>
            
            {/* Quick Questions */}
            <div className="flex flex-wrap gap-2 mt-3">
              <button className="text-xs px-2 py-1 bg-muted hover:bg-muted/80 rounded-full transition-colors">
                How to increase super?
              </button>
              <button className="text-xs px-2 py-1 bg-muted hover:bg-muted/80 rounded-full transition-colors">
                Retirement timeline
              </button>
              <button className="text-xs px-2 py-1 bg-muted hover:bg-muted/80 rounded-full transition-colors">
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