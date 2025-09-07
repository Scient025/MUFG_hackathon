import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { UserProfile } from "@/services/dataService";

interface DashboardHeaderProps {
  user: UserProfile;
  goalProgress: number;
}

export function DashboardHeader({ user, goalProgress }: DashboardHeaderProps) {
  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'Low': return 'bg-green-100 text-green-800 border-green-200';
      case 'Medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'High': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className="dashboard-card mb-8">
      <div className="flex flex-col lg:flex-row items-start lg:items-center gap-6">
        <div className="flex items-center gap-6">
          <Avatar className="w-20 h-20 lg:w-24 lg:h-24">
            <AvatarImage src="" alt={user.name} />
            <AvatarFallback className="text-2xl lg:text-3xl font-bold bg-primary text-primary-foreground">
              {user.name.split(' ').map(n => n[0]).join('')}
            </AvatarFallback>
          </Avatar>
          <div>
            <h1 className="text-3xl lg:text-4xl font-bold text-card-foreground mb-3">
              Welcome back, {user.name}
            </h1>
            <div className="flex flex-wrap items-center gap-3">
              <Badge variant="outline" className="text-lg py-2 px-4 font-medium">
                Age {user.age}
              </Badge>
              <Badge variant="outline" className="text-lg py-2 px-4 font-medium">
                {user.maritalStatus}
              </Badge>
              <Badge variant="outline" className="text-lg py-2 px-4 font-medium">
                {user.dependents} {user.dependents === 1 ? 'Dependent' : 'Dependents'}
              </Badge>
              <Badge 
                variant="outline" 
                className={`text-lg py-2 px-4 font-medium ${getRiskColor(user.riskProfile)}`}
              >
                {user.riskProfile} Risk
              </Badge>
            </div>
          </div>
        </div>
        
        <div className="lg:ml-auto text-center lg:text-right">
          <div className="text-lg text-muted-foreground mb-2 font-medium">Retirement Goal Progress</div>
          <div className="text-4xl lg:text-5xl font-bold text-success mb-3">{goalProgress}%</div>
          <div className="text-xl text-success font-semibold">
            You're on track! Keep it up! 🎯
          </div>
        </div>
      </div>
    </div>
  );
}