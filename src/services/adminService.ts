import { supabase } from '@/lib/supabase';
import bcrypt from 'bcryptjs';

interface AdminUser {
  id: string;
  username: string;
  password_hash: string;
}

export const adminService = {
  login: async (username: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const { data, error } = await supabase
        .from('admin')
        .select('*')
        .eq('username', username)
        .single();

      if (error || !data) {
        return { success: false, error: 'Invalid username or password' };
      }

      const storedPassword: string = data.password_hash || '';
      let isPasswordValid = false;
      if (storedPassword.startsWith('$2')) {
        isPasswordValid = await bcrypt.compare(password, storedPassword);
      } else {
        isPasswordValid = password === storedPassword;
      }

      if (!isPasswordValid) {
        return { success: false, error: 'Invalid username or password' };
      }

      localStorage.setItem('adminSession', JSON.stringify({
        isAdmin: true,
        username: data.username,
        loggedInAt: new Date().toISOString()
      }));

      return { success: true };
    } catch (error) {
      console.error('Admin login error:', error);
      return { success: false, error: 'An error occurred during login' };
    }
  },

  isAuthenticated: (): boolean => {
    if (typeof window === 'undefined') return false;
    
    const session = localStorage.getItem('adminSession');
    if (!session) return false;
    
    try {
      const { isAdmin } = JSON.parse(session);
      return !!isAdmin;
    } catch {
      return false;
    }
  },

  logout: (): void => {
    localStorage.removeItem('adminSession');
    window.location.href = '/';
  },

  createAdmin: async (username: string, password: string): Promise<{ success: boolean; error?: string }> => {
    try {
      const { data: existingAdmin } = await supabase
        .from('admin')
        .select('username')
        .eq('username', username)
        .single();

      if (existingAdmin) {
        return { success: false, error: 'Admin user already exists' };
      }

      const salt = await bcrypt.genSalt(10);
      const passwordHash = await bcrypt.hash(password, salt);

      const { error } = await supabase
        .from('admin')
        .insert([{ username, password_hash: passwordHash }]);

      if (error) {
        throw error;
      }

      return { success: true };
    } catch (error) {
      console.error('Error creating admin:', error);
      return { success: false, error: 'Failed to create admin user' };
    }
  }
};
