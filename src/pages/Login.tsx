import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Eye, EyeOff, Mail, Lock, Shield, Users, TrendingUp, Calculator, Award, Target } from "lucide-react";
import { dataService } from "@/services/dataService";

export default function Landing() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    username: "",
    password: ""
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    const checkAuth = () => {
      const userSession = localStorage.getItem('userSession');
      if (userSession) {
        try {
          const session = JSON.parse(userSession);
          if (session?.userId) {
            navigate("/dashboard");
          }
        } catch (err) {
          console.error("Error parsing user session:", err);
          localStorage.removeItem('userSession');
        }
      }
      setIsMounted(true);
    };
    
    checkAuth();
  }, [navigate]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    if (!formData.username || !formData.password) {
      setError("Please enter both username and password");
      setLoading(false);
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(formData.username)) {
      setError("Please enter a valid email address");
      setLoading(false);
      return;
    }

    try {
      console.log('Attempting login with username:', formData.username);
      
      const result = await dataService.loginUser({
        username: formData.username,
        password: formData.password
      });

      console.log('Login result:', result);

      if (result.success && result.userId) {
        const session = {
          userId: result.userId,
          username: result.username,
          name: result.name,
          loggedInAt: new Date().toISOString()
        };
        
        localStorage.setItem('userSession', JSON.stringify(session));
        
        console.log('Login successful, redirecting to dashboard');
        navigate("/dashboard");
      } else {
        setError(result.message || "Login failed. Please check your credentials and try again.");
      }
    } catch (err: any) {
      console.error("Login error:", err);
      setError(err.message || "An error occurred during login. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleAdminAccess = (e: React.MouseEvent) => {
    e.preventDefault();
    navigate("/admin");
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

  if (!isMounted) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100">
      {/* Header */}
      <header className="bg-white/80 backdrop-blur-sm shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Shield className="h-8 w-8 text-blue-600" />
              <span className="ml-2 text-xl font-bold text-gray-900">SuperWise</span>
            </div>
            <nav className="hidden md:flex space-x-8">
              <a href="#features" className="text-gray-600 hover:text-blue-600">Features</a>
              <a href="#how-it-works" className="text-gray-600 hover:text-blue-600">How It Works</a>
              <a href="#testimonials" className="text-gray-600 hover:text-blue-600">Testimonials</a>
              <a href="#contact" className="text-gray-600 hover:text-blue-600">Contact</a>
            </nav>
            <div className="flex items-center space-x-4">
              <Link to="/signup" className="hidden md:inline-block text-gray-700 hover:text-blue-600">
                Sign Up
              </Link>
              <Button variant="outline" onClick={() => window.location.href = '/admin'} className="hidden md:flex items-center">
                <Shield className="h-4 w-4 mr-2" />
                Admin
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Left Column - Hero Text */}
          <div className="flex flex-col justify-center">
            <h1 className="text-4xl font-bold text-gray-900 mb-6">
              Take Control of Your <span className="text-blue-600">Financial Future</span>
            </h1>
            <p className="text-lg text-gray-600 mb-8">
              SuperWise helps you plan for retirement with confidence. Our AI-powered platform provides personalized advice to help you make the most of your superannuation.
            </p>
            
            <div className="space-y-6 mb-8">
              <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 bg-blue-100 p-2 rounded-lg">
                  <TrendingUp className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">Smart Projections</h3>
                  <p className="text-gray-600">Get accurate retirement projections based on your unique financial situation.</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 bg-green-100 p-2 rounded-lg">
                  <Shield className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">Risk Analysis</h3>
                  <p className="text-gray-600">Understand your risk tolerance and optimize your investment strategy.</p>
                </div>
              </div>
              
              <div className="flex items-start space-x-4">
                <div className="flex-shrink-0 bg-purple-100 p-2 rounded-lg">
                  <Users className="h-6 w-6 text-purple-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">Peer Comparison</h3>
                  <p className="text-gray-600">See how your retirement savings compare to others in your demographic.</p>
                </div>
              </div>
            </div>

            {/* Feature Cards */}
            <div className="space-y-4">
              <h2 className="text-xl font-semibold text-gray-900">What We Provide</h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                {features.map((feature, index) => (
                  <Card key={index} className="border-0 shadow-md hover:shadow-lg transition-shadow duration-200 bg-white/80">
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
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-bold text-gray-900">Welcome Back</h2>
              <p className="text-gray-600">Sign in to your account</p>
            </div>
            
            {error && (
              <Alert variant="destructive" className="mb-6">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}
            
            <form onSubmit={handleLogin} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="username">Email</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
                  <Input
                    id="username"
                    name="username"
                    type="email"
                    placeholder="Enter your email"
                    className="pl-10"
                    value={formData.username}
                    onChange={handleInputChange}
                    required
                    disabled={loading}
                  />
                </div>
              </div>
              
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label htmlFor="password">Password</Label>
                  <Link to="/forgot-password" className="text-sm text-blue-600 hover:underline">
                    Forgot password?
                  </Link>
                </div>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
                  <Input
                    id="password"
                    name="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    className="pl-10 pr-10"
                    value={formData.password}
                    onChange={handleInputChange}
                    required
                    disabled={loading}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                    tabIndex={-1}
                    aria-label={showPassword ? "Hide password" : "Show password"}
                    title={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? (
                      <EyeOff className="h-5 w-5" />
                    ) : (
                      <Eye className="h-5 w-5" />
                    )}
                  </button>
                </div>
              </div>
              
              <Button type="submit" className="w-full" disabled={loading}>
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Signing in...
                  </>
                ) : (
                  'Sign In'
                )}
              </Button>
            </form>
            
            <div className="mt-6 text-center text-sm text-gray-600">
              Don't have an account?{' '}
              <Link to="/signup" className="font-medium text-blue-600 hover:underline">
                Sign up
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}