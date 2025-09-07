import React from "react";
import { MessageCircle, X, Move, Maximize2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useState, useRef, useCallback } from "react";

export function FloatingChatButton() {
  const [isOpen, setIsOpen] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 320, height: 384 });
  const [position, setPosition] = useState({ bottom: 96, right: 24 });
  const [isResizing, setIsResizing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const chatPanelRef = useRef<HTMLDivElement>(null);
  const resizeStartRef = useRef({ x: 0, y: 0, width: 0, height: 0 });
  const dragStartRef = useRef({ x: 0, y: 0, bottom: 0, right: 0 });

  // Resize functionality
  const handleResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsResizing(true);
    const rect = chatPanelRef.current?.getBoundingClientRect();
    if (rect) {
      resizeStartRef.current = {
        x: e.clientX,
        y: e.clientY,
        width: dimensions.width,
        height: dimensions.height
      };
    }
  }, [dimensions]);

  const handleResize = useCallback((e: MouseEvent) => {
    if (!isResizing) return;
    
    const deltaX = e.clientX - resizeStartRef.current.x;
    const deltaY = resizeStartRef.current.y - e.clientY; // Inverted for bottom-right resize
    
    const newWidth = Math.max(280, Math.min(600, resizeStartRef.current.width + deltaX));
    const newHeight = Math.max(300, Math.min(700, resizeStartRef.current.height + deltaY));
    
    setDimensions({ width: newWidth, height: newHeight });
  }, [isResizing]);

  const handleResizeEnd = useCallback(() => {
    setIsResizing(false);
  }, []);

  // Drag functionality for repositioning
  const handleDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      bottom: position.bottom,
      right: position.right
    };
  }, [position]);

  const handleDrag = useCallback((e: MouseEvent) => {
    if (!isDragging) return;
    
    const deltaX = dragStartRef.current.x - e.clientX;
    const deltaY = e.clientY - dragStartRef.current.y;
    
    const newRight = Math.max(0, Math.min(window.innerWidth - dimensions.width, dragStartRef.current.right + deltaX));
    const newBottom = Math.max(0, Math.min(window.innerHeight - dimensions.height, dragStartRef.current.bottom + deltaY));
    
    setPosition({ right: newRight, bottom: newBottom });
  }, [isDragging, dimensions]);

  const handleDragEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Mouse event listeners
  React.useEffect(() => {
    if (isResizing) {
      document.addEventListener('mousemove', handleResize);
      document.addEventListener('mouseup', handleResizeEnd);
      return () => {
        document.removeEventListener('mousemove', handleResize);
        document.removeEventListener('mouseup', handleResizeEnd);
      };
    }
  }, [isResizing, handleResize, handleResizeEnd]);

  React.useEffect(() => {
    if (isDragging) {
      document.addEventListener('mousemove', handleDrag);
      document.addEventListener('mouseup', handleDragEnd);
      return () => {
        document.removeEventListener('mousemove', handleDrag);
        document.removeEventListener('mouseup', handleDragEnd);
      };
    }
  }, [isDragging, handleDrag, handleDragEnd]);

  return (
    <>
      {/* Floating Chat Button */}
      <div 
        className="fixed z-50"
        style={{ 
          bottom: `${position.bottom - 64}px`, 
          right: `${position.right}px` 
        }}
      >
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
          <div className="absolute bottom-full right-0 mb-2 px-3 py-1 bg-neutral text-neutral-foreground text-sm rounded-lg shadow-lg opacity-0 hover:opacity-100 transition-opacity duration-200 whitespace-nowrap pointer-events-none">
            Ask AI Advisor
          </div>
        )}
      </div>

      {/* Resizable Chat Panel */}
      {isOpen && (
        <div 
          ref={chatPanelRef}
          className="fixed bg-card border border-card-border rounded-xl shadow-xl z-40 flex flex-col"
          style={{ 
            bottom: `${position.bottom}px`, 
            right: `${position.right}px`,
            width: `${dimensions.width}px`,
            height: `${dimensions.height}px`
          }}
        >
          {/* Draggable Header */}
          <div 
            className="p-4 border-b border-card-border cursor-move select-none"
            onMouseDown={handleDragStart}
          >
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-primary text-primary-foreground rounded-full flex items-center justify-center">
                <MessageCircle className="w-4 h-4" />
              </div>
              <div className="flex-1">
                <h3 className="font-semibold text-card-foreground">AI Advisor</h3>
                <p className="text-sm text-muted-foreground">How can I help with your super?</p>
              </div>
              <div className="flex items-center gap-1">
                <Move className="w-4 h-4 text-muted-foreground" />
                <span className="text-xs text-muted-foreground">Drag to move</span>
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
          
          {/* Resize Handle */}
          <div 
            className="absolute bottom-0 right-0 w-4 h-4 cursor-se-resize bg-muted/50 hover:bg-muted transition-colors"
            onMouseDown={handleResizeStart}
            style={{
              background: 'linear-gradient(-45deg, transparent 30%, currentColor 30%, currentColor 40%, transparent 40%, transparent 60%, currentColor 60%, currentColor 70%, transparent 70%)'
            }}
          />
          
          {/* Side resize handles */}
          <div 
            className="absolute bottom-0 right-4 left-4 h-1 cursor-s-resize hover:bg-primary/20 transition-colors"
            onMouseDown={handleResizeStart}
          />
          <div 
            className="absolute top-4 bottom-4 right-0 w-1 cursor-e-resize hover:bg-primary/20 transition-colors"
            onMouseDown={handleResizeStart}
          />
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