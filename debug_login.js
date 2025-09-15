// Debug script to test login functionality
// Run this in your browser console on the login page

async function debugLogin() {
  try {
    // Import the dataService (assuming it's available globally or we can access it)
    console.log('Testing login with your credentials...');
    
    // Test the debug function first
    const result = await dataService.debugUserExists('U1757964409558zcdigw8rl');
    console.log('Debug result:', result);
    
    // Test the actual login
    const loginResult = await dataService.loginUser({
      userId: 'U1757964409558zcdigw8rl',
      password: 'test123'
    });
    
    console.log('Login result:', loginResult);
    
  } catch (error) {
    console.error('Debug error:', error);
  }
}

// Also test direct Supabase query
async function testDirectQuery() {
  try {
    const { supabase } = await import('/src/lib/supabase.ts');
    
    console.log('Testing direct Supabase query...');
    
    // Test 1: Find user by ID only
    const { data: userData, error: userError } = await supabase
      .from('MUFG')
      .select('User_ID, Name, Password')
      .eq('User_ID', 'U1757964409558zcdigw8rl');
    
    console.log('User query result:', { userData, userError });
    
    // Test 2: Find user by ID and password
    const { data: loginData, error: loginError } = await supabase
      .from('MUFG')
      .select('User_ID, Name, Password')
      .eq('User_ID', 'U1757964409558zcdigw8rl')
      .eq('Password', 'test123');
    
    console.log('Login query result:', { loginData, loginError });
    
  } catch (error) {
    console.error('Direct query error:', error);
  }
}

console.log('Debug functions loaded. Run debugLogin() or testDirectQuery() to test.');
