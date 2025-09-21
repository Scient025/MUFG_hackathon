import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UserProfile } from "@/services/dataService";
import { Mail, Loader2 } from "lucide-react";
import { useState } from "react";
import { useToast } from "@/hooks/use-toast";

interface DashboardHeaderProps {
  user: UserProfile;
  goalProgress: any;
}

export function DashboardHeader({ user, goalProgress }: DashboardHeaderProps) {
  const [isSendingEmail, setIsSendingEmail] = useState(false);
  const [newsSource, setNewsSource] = useState("newsapi");
  const { toast } = useToast();

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low': return 'bg-green-100 text-green-800 border-green-200';
      case 'Medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'High': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const handleSendEmail = async () => {
    setIsSendingEmail(true);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || '/api'}/trigger-email/${newsSource}`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const result = await response.json();
      
      if (result.success) {
        const sourceName = newsSource === "gemini" ? "Gemini AI" : "NewsAPI";
        toast({
          title: "Email Sent! 📧",
          description: `Financial update email sent successfully using ${sourceName}.`,
        });
      } else {
        toast({
          title: "Email Failed",
          description: "Failed to send email. Please try again.",
          variant: "destructive",
        });
      }
    } catch (error) {
      console.error('Error sending email:', error);
      toast({
        title: "Error",
        description: "Failed to send email. Please check your connection.",
        variant: "destructive",
      });
    } finally {
      setIsSendingEmail(false);
    }
  };

  return (
    <div className="dashboard-card mb-8">
      <div className="flex flex-col lg:flex-row items-start lg:items-center gap-6">
        <div className="flex items-center gap-6">
          <Avatar className="w-20 h-20 lg:w-24 lg:h-24">
            <AvatarImage src="" alt={user.User_ID || user.id || 'User'} />
            <AvatarFallback className="text-2xl lg:text-3xl font-bold bg-primary text-primary-foreground">
              {user.User_ID ? user.User_ID.slice(-2) : (user.id ? user.id.slice(-2) : 'U')}
            </AvatarFallback>
          </Avatar>
          <div>
            <h1 className="text-3xl lg:text-4xl font-bold text-card-foreground mb-3">
              Welcome back, {user.Name || user.name || user.User_ID || user.id || 'User'}
            </h1>
            <div className="flex flex-wrap items-center gap-3">
              <Badge variant="outline" className="text-lg py-2 px-4 font-medium">
                Age {user.Age || user.age || 'N/A'}
              </Badge>
              <Badge variant="outline" className="text-lg py-2 px-4 font-medium">
                {user.Marital_Status || user.marital_status || 'N/A'}
              </Badge>
              <Badge variant="outline" className="text-lg py-2 px-4 font-medium">
                {user.Number_of_Dependents || user.number_of_dependents || 0} {(user.Number_of_Dependents || user.number_of_dependents || 0) === 1 ? 'Dependent' : 'Dependents'}
              </Badge>
              <Badge 
                variant="outline" 
                className={`text-lg py-2 px-4 font-medium ${getRiskColor(user.Risk_Tolerance || user.risk_tolerance || 'Medium')}`}
              >
                {user.Risk_Tolerance || user.risk_tolerance || 'Medium'} Risk
              </Badge>
            </div>
          </div>
        </div>
        
        <div className="lg:ml-auto text-center lg:text-right">
          <div className="text-lg text-muted-foreground mb-2 font-medium">Retirement Goal Progress</div>
          <div className="text-4xl lg:text-5xl font-bold text-success mb-3">
            {Math.round(goalProgress?.percentage || 0)}%
          </div>
          <div className="text-xl text-success font-semibold mb-4">
            You're on track! Keep it up! 🎯
          </div>
          
          {/* Email Trigger Section */}
          <div className="space-y-3">
            <div className="text-sm text-muted-foreground">News Source:</div>
            <Select value={newsSource} onValueChange={setNewsSource}>
              <SelectTrigger className="w-48">
                <SelectValue placeholder="Select news source" />
              </SelectTrigger>
              <SelectContent className="bg-white border border-gray-300">
                <SelectItem value="newsapi">📰 NewsAPI (Real-time)</SelectItem>
                <SelectItem value="gemini">🤖 Gemini AI (Generated)</SelectItem>
              </SelectContent>
            </Select>
            
            <Button
              onClick={handleSendEmail}
              disabled={isSendingEmail}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-medium transition-colors w-full"
            >
              {isSendingEmail ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Sending...
                </>
              ) : (
                <>
                  <Mail className="w-4 h-4 mr-2" />
                  Get Financial Update
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}