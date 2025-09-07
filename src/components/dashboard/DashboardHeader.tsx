import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";

interface DashboardHeaderProps {
  user: {
    name: string;
    age: number;
    avatar?: string;
    maritalStatus: string;
    dependents: number;
    riskProfile: string;
    goalProgress: number;
  };
}

export function DashboardHeader({ user }: DashboardHeaderProps) {
  return (
    <div className="dashboard-card mb-8">
      <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
        <div className="flex items-center gap-4">
          <Avatar className="w-16 h-16 md:w-20 md:h-20">
            <AvatarImage src={user.avatar} alt={user.name} />
            <AvatarFallback className="text-2xl font-semibold bg-primary text-primary-foreground">
              {user.name.split(' ').map(n => n[0]).join('')}
            </AvatarFallback>
          </Avatar>
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-card-foreground mb-2">
              Welcome back, {user.name}
            </h1>
            <div className="flex flex-wrap items-center gap-3">
              <Badge variant="outline" className="text-base py-1 px-3">
                Age {user.age}
              </Badge>
              <Badge variant="outline" className="text-base py-1 px-3">
                {user.maritalStatus}
              </Badge>
              <Badge variant="outline" className="text-base py-1 px-3">
                {user.dependents} {user.dependents === 1 ? 'Dependent' : 'Dependents'}
              </Badge>
              <Badge 
                variant="outline" 
                className={`text-base py-1 px-3 ${
                  user.riskProfile === 'Low' ? 'status-good' :
                  user.riskProfile === 'Medium' ? 'status-warning' : 'status-risk'
                }`}
              >
                {user.riskProfile} Risk
              </Badge>
            </div>
          </div>
        </div>
        
        <div className="ml-auto text-right">
          <div className="text-sm text-muted-foreground mb-1">Retirement Goal Progress</div>
          <div className="text-3xl font-bold text-success mb-2">{user.goalProgress}%</div>
          <div className="text-lg text-success font-medium">
            You're on track! Keep it up! 🎯
          </div>
        </div>
      </div>
    </div>
  );
}