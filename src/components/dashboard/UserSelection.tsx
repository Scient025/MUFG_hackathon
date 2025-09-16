import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { User } from "@/services/dataService";
import { useState, useEffect } from "react";

interface UserSelectionProps {
  selectedUserId: string;
  onUserChange: (userId: string) => void;
  availableUsers: User[];
}

export function UserSelection({ selectedUserId, onUserChange, availableUsers }: UserSelectionProps) {
  const [selectedUser, setSelectedUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (selectedUserId && availableUsers.length > 0) {
      const user = availableUsers.find(u => u.User_ID === selectedUserId);
      setSelectedUser(user || null);
    }
  }, [selectedUserId, availableUsers]);
  
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
        <SelectContent className="bg-white border-2 border-gray-200">
          {availableUsers.map((user) => (
            <SelectItem key={user.User_ID} value={user.User_ID} className="text-base py-3 bg-white hover:bg-gray-50">
              <div className="flex flex-col">
                <span className="font-medium text-gray-900">{user.User_ID}</span>
                <span className="text-sm text-gray-600">
                  {user.Name || 'No Name'} - Age {user.Age}, {user.Risk_Tolerance} Risk
                </span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
          </Select>
        </div>
      </div>
      
      {loading && (
        <div className="mt-4 p-4 bg-muted/30 rounded-xl">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-primary rounded-full animate-pulse"></div>
            <span className="font-medium text-card-foreground">
              Loading user profile...
            </span>
          </div>
        </div>
      )}
      
      {selectedUser && !loading && (
        <div className="mt-4 p-4 bg-muted/30 rounded-xl">
          <div className="flex items-center gap-3">
            <div className="w-3 h-3 bg-primary rounded-full"></div>
            <span className="font-medium text-card-foreground">
              Currently viewing: {selectedUser.Name || selectedUser.User_ID} - Age {selectedUser.Age}, {selectedUser.Risk_Tolerance} Risk
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
