import { supabase, UserProfile } from "@/lib/supabase";

export class SupabaseService {
  // Get current authenticated user
  static async getCurrentUser() {
    const { data: { user } } = await supabase.auth.getUser();
    return user;
  }

  // Get user profile by ID
  static async getUserProfile(userId: string): Promise<UserProfile | null> {
    console.log('SupabaseService: Fetching user profile for userId:', userId);
    
    const { data, error } = await supabase
      .from('MUFG')
      .select('*')
      .eq('User_ID', userId)
      .single();

    if (error) {
      console.error('SupabaseService: Error fetching user profile:', error);
      return null;
    }

    console.log('SupabaseService: User profile found:', data);
    return data;
  }

  // Get all user profiles (for admin)
  static async getAllUserProfiles(): Promise<UserProfile[]> {
    const { data, error } = await supabase
      .from('MUFG')
      .select('*')
      .order('User_ID', { ascending: false });

    if (error) {
      console.error('Error fetching user profiles:', error);
      return [];
    }

    return data || [];
  }

  // Update user profile
  static async updateUserProfile(userId: string, updates: Partial<UserProfile>): Promise<boolean> {
    const { error } = await supabase
      .from('MUFG')
      .update(updates)
      .eq('User_ID', userId);

    if (error) {
      console.error('Error updating user profile:', error);
      return false;
    }

    return true;
  }

  // Sign out user
  static async signOut() {
    const { error } = await supabase.auth.signOut();
    if (error) {
      console.error('Error signing out:', error);
    }
    return !error;
  }

  // Check if user is authenticated
  static async isAuthenticated(): Promise<boolean> {
    const { data: { session } } = await supabase.auth.getSession();
    return !!session;
  }

  // Listen to auth state changes
  static onAuthStateChange(callback: (user: any) => void) {
    return supabase.auth.onAuthStateChange((event, session) => {
      callback(session?.user || null);
    });
  }
}
