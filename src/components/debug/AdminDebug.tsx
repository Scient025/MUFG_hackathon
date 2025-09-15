import { useAdminAuth } from '@/contexts/AdminAuthContext';
import { useAuth } from '@/contexts/AuthContext';

export function AdminDebug() {
  const { adminUser, isAdminMode } = useAdminAuth();
  const { user } = useAuth();

  return (
    <div className="fixed bottom-4 right-4 bg-black bg-opacity-80 text-white p-4 rounded-lg text-xs">
      <div><strong>Debug Info:</strong></div>
      <div>Regular User: {user ? user.id : 'None'}</div>
      <div>Admin Mode: {isAdminMode ? 'Yes' : 'No'}</div>
      <div>Admin User: {adminUser ? adminUser.id : 'None'}</div>
      <div>Admin User Name: {adminUser ? adminUser.name : 'None'}</div>
    </div>
  );
}
