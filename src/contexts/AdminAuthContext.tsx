import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { UserProfile } from '@/lib/supabase';

interface AdminAuthContextType {
  adminUser: UserProfile | null;
  loginAsUser: (user: UserProfile) => void;
  logout: () => void;
  isAdminMode: boolean;
}

const AdminAuthContext = createContext<AdminAuthContextType | undefined>(undefined);

export function AdminAuthProvider({ children }: { children: ReactNode }) {
  const [adminUser, setAdminUser] = useState<UserProfile | null>(null);
  const [isAdminMode, setIsAdminMode] = useState(false);

  const loginAsUser = (user: UserProfile) => {
    setAdminUser(user);
    setIsAdminMode(true);
    // Store in sessionStorage for persistence
    sessionStorage.setItem('adminUser', JSON.stringify(user));
    sessionStorage.setItem('isAdminMode', 'true');
  };

  const logout = () => {
    setAdminUser(null);
    setIsAdminMode(false);
    sessionStorage.removeItem('adminUser');
    sessionStorage.removeItem('isAdminMode');
  };

  // Initialize from sessionStorage on mount
  useEffect(() => {
    const storedUser = sessionStorage.getItem('adminUser');
    const storedMode = sessionStorage.getItem('isAdminMode');
    
    if (storedUser && storedMode === 'true') {
      try {
        setAdminUser(JSON.parse(storedUser));
        setIsAdminMode(true);
      } catch (error) {
        console.error('Error parsing stored admin user:', error);
        sessionStorage.removeItem('adminUser');
        sessionStorage.removeItem('isAdminMode');
      }
    }
  }, []);

  const value = {
    adminUser,
    loginAsUser,
    logout,
    isAdminMode,
  };

  return (
    <AdminAuthContext.Provider value={value}>
      {children}
    </AdminAuthContext.Provider>
  );
}

export function useAdminAuth() {
  const context = useContext(AdminAuthContext);
  if (context === undefined) {
    throw new Error('useAdminAuth must be used within an AdminAuthProvider');
  }
  return context;
}
