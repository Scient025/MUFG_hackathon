import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Eye, EyeOff, TrendingUp, Shield, Users, Calculator, Award, Target } from "lucide-react";
import { dataService } from "@/services/dataService";

export default function Landing() {
  const navigate = useNavigate();
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      console.log('Attempting login with name:', name, 'password:', password);
      
      const result = await dataService.loginUser({
        name,
        password
      });

      console.log('Login result:', result);

      if (result.success && result.userId) {
        // Store user info in localStorage for the session
        localStorage.setItem('currentUser', JSON.stringify({
          userId: result.userId,
          name: result.name
        }));
        
        console.log('Redirecting to dashboard with userId:', result.userId);
        
        // Redirect to dashboard with userId
        navigate("/dashboard", { state: { userId: result.userId } });
      } else {
        console.log('Login failed:', result.message);
        setError(result.message || "Login failed");
      }
    } catch (err) {
      console.error("Login error:", err);
      setError("An unexpected error occurred");
    } finally {
      setLoading(false);
    }
  };

  const features = [
    {
      icon: <Calculator className="h-8 w-8 text-blue-600" />,
      title: "Smart Calculations",
      description: "Get accurate projections of your superannuation balance and retirement income with our advanced calculators."
    },
    {
      icon: <Award className="h-8 w-8 text-green-600" />,
      title: "Expert Guidance",
      description: "Access professional advice tailored to your unique financial situation and retirement goals."
    },
    {
      icon: <Target className="h-8 w-8 text-purple-600" />,
      title: "Goal Planning",
      description: "Set and track your retirement goals with personalized strategies to maximize your super savings."
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="bg-gradient-to-r from-blue-600 to-indigo-600 p-2 rounded-lg">
                <TrendingUp className="h-6 w-6 text-white" />
              </div>
              <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                SuperWise
              </span>
            </div>
            <div className="flex items-center space-x-4">
              <Link to="/signup" className="text-sm text-gray-500 hover:text-blue-600 transition-colors">
                Sign Up
              </Link>
              <Link to="/admin" className="text-sm text-gray-500 hover:text-blue-600 transition-colors">
                Admin Access
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid lg:grid-cols-2 gap-8 items-start min-h-[calc(100vh-8rem)]">
          {/* Left Column - Content */}
          <div className="space-y-6 flex flex-col justify-center">
            {/* Hero Section */}
            <div className="space-y-4">
              <div className="inline-flex items-center space-x-2 bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm font-medium">
                <Shield className="h-4 w-4" />
                <span>Secure & Trusted</span>
              </div>
              
              <h1 className="text-3xl lg:text-4xl font-bold text-gray-900 leading-tight">
                Take Control of Your
                <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent block">
                  Superannuation Future
                </span>
              </h1>
              
              <p className="text-lg text-gray-600 leading-relaxed">
                Get personalized superannuation advice, smart calculators, and expert guidance 
                to maximize your retirement savings. Your financial future starts here.
              </p>
              

            </div>

            {/* Feature Cards */}
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-900">What We Provide</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                {features.map((feature, index) => (
                  <Card key={index} className="border-0 shadow-md hover:shadow-lg transition-shadow duration-200 bg-white/80 backdrop-blur-sm">
                    <CardContent className="p-4 text-center">
                      <div className="flex flex-col items-center space-y-2">
                        <div className="bg-gray-50 p-3 rounded-lg">
                          {feature.icon}
                        </div>
                        <h3 className="text-sm font-semibold text-gray-900">
                          {feature.title}
                        </h3>
                        <p className="text-xs text-gray-800 leading-relaxed">
                          {feature.description}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column - Login Form */}
          <div className="flex items-center justify-center">
            <Card className="w-full max-w-md shadow-xl border-0 bg-white/90 backdrop-blur-sm">
              <CardHeader className="text-center">
                <CardTitle className="text-3xl font-bold">Welcome Back</CardTitle>
                <CardDescription className="text-lg">
                  Sign in to your SuperWise account
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {error && (
                    <Alert variant="destructive">
                      <AlertDescription>{error}</AlertDescription>
                    </Alert>
                  )}
                  
                  <div className="space-y-2">
                    <Label htmlFor="name">Name</Label>
                    <Input
                      id="name"
                      type="text"
                      value={name}
                      onChange={(e) => setName(e.target.value)}
                      placeholder="Enter your name"
                      className="h-12"
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="password">Password</Label>
                    <div className="relative">
                      <Input
                        id="password"
                        type={showPassword ? "text" : "password"}
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        placeholder="Enter your password"
                        className="h-12 pr-10"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-12 px-3 py-2 hover:bg-transparent"
                        onClick={() => setShowPassword(!showPassword)}
                      >
                        {showPassword ? (
                          <EyeOff className="h-4 w-4" />
                        ) : (
                          <Eye className="h-4 w-4" />
                        )}
                      </Button>
                    </div>
                  </div>
                  
                  <Button
                    onClick={handleLogin}
                    className="w-full h-12"
                    disabled={loading}
                  >
                    {loading ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Signing In...
                      </>
                    ) : (
                      "Sign In"
                    )}
                  </Button>
                </div>
                
                <div className="mt-6 text-center">
                  <p className="text-sm text-muted-foreground">
                    Don't have an account?{" "}
                    <Link to="/signup" className="text-primary hover:underline">
                      Sign up
                    </Link>
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}