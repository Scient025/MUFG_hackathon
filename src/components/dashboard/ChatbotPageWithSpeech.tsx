import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageCircle, Send, Bot, User, Loader2, Mic, MicOff, Volume2, VolumeX } from "lucide-react";
import { useState, useEffect, useRef } from "react";
import { dataService } from "@/services/dataService";
import { useToast } from "@/hooks/use-toast";

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  audioUrl?: string; // For TTS audio
}

type TablePart = {
  type: 'table';
  headers: string[];
  rows: string[][];
} | {
  type: 'text';
  content: string;
};

interface ChatbotPageProps {
  user: any; // UserProfile from Supabase
}

export function ChatbotPageWithSpeech({ user }: ChatbotPageProps) {
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
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [voices, setVoices] = useState<any[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('en-AU-NatashaNeural');
  const [currentPlayingMessage, setCurrentPlayingMessage] = useState<string | null>(null);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const { toast } = useToast();

  // Load available voices on component mount
  useEffect(() => {
    loadVoices();
  }, []);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]');
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [messages, isLoading]);

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

    // Function to parse and render markdown tables
  const parseMarkdownTable = (text: string) => {
    const tableRegex = /\|(.+)\|\s*\n\|[-\s|:]+\|\s*\n((?:\|.+\|\s*\n?)+)/g;
    const parts: TablePart[] = [];
    let lastIndex = 0;
    let match;

    while ((match = tableRegex.exec(text)) !== null) {
      // Add text before the table
      if (match.index > lastIndex) {
        const beforeText = text.slice(lastIndex, match.index).trim();
        if (beforeText) {
          parts.push({ type: 'text', content: beforeText });
        }
      }

      // Parse table headers
      const headerRow = match[1];
      const headers = headerRow.split('|').map(h => h.trim()).filter(h => h);

      // Parse table rows
      const bodyText = match[2];
      const rows = bodyText.trim().split('\n').map(row => {
        return row.split('|').map(cell => cell.trim()).filter(cell => cell);
      });

      parts.push({
        type: 'table',
        headers,
        rows: rows.filter(row => row.length === headers.length)
      });

      lastIndex = match.index + match[0].length;
    }

    // Add remaining text after the last table
    if (lastIndex < text.length) {
      const remainingText = text.slice(lastIndex).trim();
      if (remainingText) {
        parts.push({ type: 'text', content: remainingText });
      }
    }

    // If no tables found, return the original text as a single text part
    if (parts.length === 0) {
      parts.push({ type: 'text', content: text });
    }

    return parts;
  };

  // Function to render formatted text with proper sections
  const renderFormattedText = (text: string) => {
    const lines = text.split('\n').filter(line => line.trim());
    const sectionHeaders = ['Summary', 'Strengths', 'Areas for Improvement', 'Recommendations', 
                          'Next Steps', 'Encouragement Statement', 'Key Metrics', 'Financial Snapshot'];
    
    return lines.map((line, lineIdx) => {
      const isHeading = sectionHeaders.some(header => line.trim().startsWith(header));
      const isBullet = line.trim().startsWith('-');
      const isNumbered = line.trim().match(/^\d+\./);
      
      return (
        <div 
          key={lineIdx} 
          className={`
            ${isHeading ? 'font-semibold mt-3 mb-2 text-card-foreground text-base' : ''}
            ${isBullet || isNumbered ? 'ml-4 mb-1' : 'mb-2'}
          `}
        >
          {line.trim()}
        </div>
      );
    });
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      
      const audioChunks: Blob[] = [];
      
      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };
      
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        await processAudioInput(audioBlob);
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      setIsRecording(true);
      
      toast({
        title: "Recording Started",
        description: "Speak your question now...",
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
        setInputValue(result.text);
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
        // Convert hex string back to audio data
        const audioData = new Uint8Array(result.audio_data.match(/.{1,2}/g)!.map(byte => parseInt(byte, 16)));
        const audioBlob = new Blob([audioData], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Play the audio
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

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputValue;
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the real AI API
      const userId = (user && (user.User_ID || user.id || user.userId)) as string;
      const response = await dataService.sendChatMessage(userId, currentInput);
      
      const botResponse: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: response || "I'm sorry, I couldn't process your request at the moment. Please try again.",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botResponse]);
      
      // Convert bot response to speech
      if (speechEnabled) {
        await textToSpeech(botResponse.content, botResponse.id);
      }
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
      {/* Speech Controls - Always visible when speech is enabled */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-xl font-bold text-card-foreground flex items-center gap-3">
            <Volume2 className="w-6 h-6" />
            Speech Controls
            {!speechEnabled && <span className="text-sm font-normal text-muted-foreground">(Disabled)</span>}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Voice:</label>
              <select
                value={selectedVoice}
                onChange={(e) => setSelectedVoice(e.target.value)}
                className="px-3 py-1 border rounded-md"
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
            </div>
            
            <Button
              onClick={isRecording ? stopRecording : startRecording}
              variant={isRecording ? "destructive" : "default"}
              className="flex items-center gap-2"
              disabled={!speechEnabled}
            >
              {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
              {isRecording ? "Stop Recording" : "Start Recording"}
            </Button>
            
            {isPlaying && (
              <Button
                onClick={stopAudioPlayback}
                variant="destructive"
                className="flex items-center gap-2"
              >
                <VolumeX className="w-4 h-4" />
                Stop Audio
              </Button>
            )}
            
            <Button
              onClick={() => setSpeechEnabled(!speechEnabled)}
              variant="outline"
              className="flex items-center gap-2"
            >
              {speechEnabled ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
              {speechEnabled ? "Disable Speech" : "Enable Speech"}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Chat Interface */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground flex items-center gap-3">
            <MessageCircle className="w-7 h-7" />
            AI Superannuation Advisor {speechEnabled && <span className="text-sm font-normal text-muted-foreground">(with Speech)</span>}
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
                        {message.type === 'bot' ? (
                          <div className="space-y-3">
                            {parseMarkdownTable(message.content).map((part, idx) => {
                              if (part.type === 'table') {
                                return (
                                  <div key={idx} className="my-4">
                                    <div className="overflow-x-auto">
                                      <table className="w-full border-collapse border border-gray-300 bg-white rounded-lg shadow-sm">
                                        <thead>
                                          <tr className="bg-gray-50">
                                            {part.headers.map((header: string, headerIdx: number) => (
                                              <th
                                                key={headerIdx}
                                                className="border border-gray-300 px-3 py-2 text-left font-semibold text-gray-700 text-sm"
                                              >
                                                {header}
                                              </th>
                                            ))}
                                          </tr>
                                        </thead>
                                        <tbody>
                                          {part.rows.map((row: string[], rowIdx: number) => (
                                            <tr key={rowIdx} className={rowIdx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                                              {row.map((cell: string, cellIdx: number) => (
                                                <td
                                                  key={cellIdx}
                                                  className="border border-gray-300 px-3 py-2 text-gray-800 text-sm"
                                                >
                                                  {cell}
                                                </td>
                                              ))}
                                            </tr>
                                          ))}
                                        </tbody>
                                      </table>
                                    </div>
                                  </div>
                                );
                              } else {
                                return (
                                  <div key={idx}>
                                    {renderFormattedText(part.content)}
                                  </div>
                                );
                              }
                            })}
                          </div>
                        ) : (
                          message.content
                        )}
                      </div>
                      <div className={`text-xs text-muted-foreground mt-1 flex items-center gap-2 ${
                        message.type === 'user' ? 'text-right justify-end' : 'text-left justify-start'
                      }`}>
                        <span>{formatTime(message.timestamp)}</span>
                        {message.type === 'bot' && speechEnabled && (
                          <button
                            onClick={() => textToSpeech(message.content, message.id)}
                            className="p-1 hover:bg-muted rounded-full transition-colors"
                            title="Read this message"
                            disabled={isPlaying && currentPlayingMessage === message.id}
                          >
                            {isPlaying && currentPlayingMessage === message.id ? (
                              <VolumeX className="w-3 h-3 text-red-500" />
                            ) : (
                              <Volume2 className="w-3 h-3 text-muted-foreground hover:text-foreground" />
                            )}
                          </button>
                        )}
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
            {[
              "What is my risk category?",
              "What if I increase my monthly contribution by $200?",
              "How much will I retire with?",
              "How do I compare to others my age?",
              "Am I on track for retirement?",
              "Should I consider changing my risk profile?"
            ].map((question, index) => (
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

      {/* Hidden audio element for TTS */}
      <audio ref={audioRef} />
    </div>
  );
}
