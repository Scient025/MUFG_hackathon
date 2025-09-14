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

export function SignupFormWithSpeech({ onSignupSuccess, onCancel }: SignupFormProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  const [voices, setVoices] = useState<any[]>([]);
  const [selectedVoice, setSelectedVoice] = useState('en-AU-NatashaNeural');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const { toast } = useToast();

  // Load available voices on component mount
  useEffect(() => {
    loadVoices();
  }, []);

  const loadVoices = async () => {
    try {
      const response = await fetch('http://localhost:8000/voices');
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
      
      const response = await fetch('http://localhost:8000/speech-to-text', {
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
        // This would integrate with the existing chat logic
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

  const textToSpeech = async (text: string) => {
    if (!speechEnabled) return;
    
    try {
      const response = await fetch('http://localhost:8000/text-to-speech', {
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
          
          audioRef.current.onended = () => {
            setIsPlaying(false);
            URL.revokeObjectURL(audioUrl);
          };
        }
      }
    } catch (error) {
      console.error('Error with text-to-speech:', error);
    }
  };

  // Rest of the component would be similar to the original SignupForm
  // but with speech controls integrated into the chat interface
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="flex gap-6 w-full max-w-6xl">
        {/* Enhanced Chatbot Panel with Speech */}
        <div className="w-96 h-fit bg-white rounded-lg shadow-2xl border flex flex-col">
          {/* Speech Controls Header */}
          {speechEnabled && (
            <div className="bg-green-600 text-white px-4 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Volume2 className="w-4 h-4" />
                <span className="text-sm font-medium">Speech Assistant</span>
              </div>
              <div className="flex items-center gap-2">
                <select
                  value={selectedVoice}
                  onChange={(e) => setSelectedVoice(e.target.value)}
                  className="text-xs bg-green-700 text-white border-0 rounded px-2 py-1"
                  aria-label="Select voice for text-to-speech"
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
                >
                  {isRecording ? <MicOff className="w-3 h-3" /> : <Mic className="w-3 h-3" />}
                </Button>
              </div>
            </div>
          )}
          
          {/* Rest of the chat interface would go here */}
          <div className="p-4 text-center text-gray-500">
            <Bot className="w-8 h-8 mx-auto mb-2" />
            <p>Enhanced Profile Creation Assistant with Speech</p>
            <p className="text-sm">Use voice or text to complete your profile</p>
          </div>
        </div>

        {/* Form Card */}
        <Card className="flex-1 max-w-2xl">
          <CardHeader>
            <CardTitle className="text-3xl font-bold text-center">Join SuperWise</CardTitle>
            <CardDescription className="text-center text-lg">
              Create your personalized superannuation profile with voice assistance
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-center py-8">
              <p className="text-gray-600 mb-4">
                This is a simplified version. The full implementation would include:
              </p>
              <ul className="text-left text-sm text-gray-600 space-y-2">
                <li>• Voice input for all form fields</li>
                <li>• Text-to-speech for questions and confirmations</li>
                <li>• Voice-guided form completion</li>
                <li>• Speech recognition for natural language input</li>
              </ul>
              <div className="mt-6 flex gap-4 justify-center">
                <Button onClick={onCancel} variant="outline">
                  Cancel
                </Button>
                <Button onClick={() => onSignupSuccess('demo_user')}>
                  Continue to Dashboard
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
      
      {/* Hidden audio element for TTS */}
      <audio ref={audioRef} />
    </div>
  );
}
