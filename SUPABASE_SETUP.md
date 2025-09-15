# Supabase Setup Instructions

## Database Setup

1. **Run the SQL Script**
   - Go to your Supabase dashboard: https://imtmbgbktomztqtoyuvh.supabase.co
   - Navigate to the SQL Editor
   - Copy and paste the contents of `supabase_setup.sql`
   - Execute the script to create the necessary tables and policies

## Authentication Setup

1. **Enable Email Authentication**
   - In your Supabase dashboard, go to Authentication > Settings
   - Make sure "Enable email confirmations" is configured as needed
   - For development, you can disable email confirmations

2. **Configure Auth Settings**
   - Set up your site URL (e.g., `http://localhost:5173` for development)
   - Configure redirect URLs for your application

## Application Features

### User Authentication Flow
1. **Sign Up**: Users create an account with email/password
2. **Profile Setup**: After signup, users complete their financial profile
3. **Dashboard**: Users access their personalized dashboard with chatbot first
4. **Admin Panel**: Admins can view all user data

### Key Features Implemented
- ✅ Chatbot appears first in the UI
- ✅ Supabase authentication (login/signup)
- ✅ User profile data stored in Supabase
- ✅ Admin page to view all users
- ✅ User details displayed from Supabase
- ✅ Secure data access with Row Level Security

### Database Schema
The `user_profiles` table includes:
- Personal information (name, age, gender, etc.)
- Financial data (income, savings, contributions)
- Investment preferences (risk tolerance, experience level)
- Goals and planning information

### Security
- Row Level Security (RLS) enabled
- Users can only access their own data
- Admin access available for user management

## Running the Application

1. Install dependencies: `npm install`
2. Start the development server: `npm run dev`
3. Navigate to `http://localhost:5173`
4. The app will redirect to login page by default
5. Create an account or use admin access

## Admin Access
- Navigate to `/admin` to view all user profiles
- Search and filter users by various criteria
- View detailed user information

## Notes
- The chatbot integration with the backend API remains unchanged
- User data is now stored in Supabase instead of local storage
- Authentication state is managed globally with React Context
- All user interactions are secured with Supabase RLS policies
