import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { UserProfile } from "@/services/dataService";

interface UserSelectionProps {
  users: UserProfile[];
  selectedUserId: string;
  onUserChange: (userId: string) => void;
}

export function UserSelection({ users, selectedUserId, onUserChange }: UserSelectionProps) {
  const selectedUser = users.find(user => user.User_ID === selectedUserId);
  
  return (
    <div className="dashboard-card mb-6">
      <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4">
        <div className="flex-1">
          <h2 className="text-xl font-semibold text-card-foreground mb-2">
            Switch User Profile
          </h2>
          <p className="text-muted-foreground text-sm">
            Select a different user to view their personalized dashboard
          </p>
        </div>
        
        <div className="w-full sm:w-80">
          <Select value={selectedUserId} onValueChange={onUserChange}>
            <SelectTrigger className="h-12 text-base">
              <SelectValue placeholder="Select a user profile" />
            </SelectTrigger>
            <SelectContent>
              {users.map((user) => (
                <SelectItem key={user.User_ID} value={user.User_ID} className="text-base py-3">
                  <div className="flex flex-col">
                    <span className="font-medium">{user.name}</span>
                    <span className="text-sm text-muted-foreground">
                      Age {user.age} • {user.riskProfile} Risk • ${user.currentSavings.toLocaleString()}
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      
      {selectedUser && (
        <div className="mt-4 p-4 bg-muted/30 rounded-xl">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-primary rounded-full"></div>
            <span className="font-medium text-card-foreground">
              Currently viewing: {selectedUser.name}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
