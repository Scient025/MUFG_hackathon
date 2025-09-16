import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Loader2, Search, User, Mail, Calendar, DollarSign, LogOut } from "lucide-react";
import { supabase } from "@/lib/supabase";
import { useNavigate } from "react-router-dom";
import { adminService } from "@/services/adminService";
import { AdminLoginModal } from "@/components/auth/AdminLoginModal";
import { UserProfile } from "@/services/dataService";

interface ExtendedUserProfile extends UserProfile {
  username?: string;
  created_at?: string;
  updated_at?: string;
}

export default function Admin() {
  const navigate = useNavigate();
  const [users, setUsers] = useState<ExtendedUserProfile[]>([]);
  const [filteredUsers, setFilteredUsers] = useState<ExtendedUserProfile[]>([]);
  const [loading, setLoading] = useState(true);
  const [showLoginModal, setShowLoginModal] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedUser, setSelectedUser] = useState<ExtendedUserProfile | null>(null);
  const [filterBy, setFilterBy] = useState("all");

  useEffect(() => {
    const checkAuth = () => {
      const authStatus = adminService.isAuthenticated();
      setIsAuthenticated(authStatus);
      
      if (!authStatus) {
        setShowLoginModal(true);
        setLoading(false);
      } else {
        loadUsers();
      }
    };

    checkAuth();
  }, []);

  useEffect(() => {
    filterUsers();
  }, [users, searchTerm, filterBy]);

  const loadUsers = async () => {
    try {
      setLoading(true);
      const response = await fetch('/api/supabase/users');
      const result = await response.json();
      
      if (result.success) {
        const normalized: ExtendedUserProfile[] = (result.data || []).map((u: any) => {
          const annualIncome = Number(u.Annual_Income ?? u.annual_income ?? 0) || 0;
          const currentSavings = Number(u.Current_Savings ?? u.current_savings ?? 0) || 0;
          const age = Number(u.Age ?? u.age ?? 0) || 0;
          const rawCreatedAt = u.created_at ?? u.createdAt ?? null;
          const createdAt = (rawCreatedAt === 'N/A' || !rawCreatedAt) ? null : rawCreatedAt;

          return {
            User_ID: u.User_ID ?? u.id,
            Name: u.Name ?? u.name ?? 'N/A',
            Age: age,
            Annual_Income: annualIncome,
            Current_Savings: currentSavings,
            Risk_Tolerance: u.Risk_Tolerance ?? u.risk_tolerance ?? 'Not specified',
            created_at: createdAt,
            username: u.username ?? u.email ?? 'N/A',
            ...u,
          } as ExtendedUserProfile;
        });

        setUsers(normalized);
      } else {
        setError("Failed to load users from backend");
      }
    } catch (err) {
      setError("Failed to load users - make sure backend is running");
    } finally {
      setLoading(false);
    }
  };

  const filterUsers = () => {
    let filtered = users;

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(user =>
        (user.Name?.toLowerCase().includes(searchTerm.toLowerCase()) || '') ||
        ((user as ExtendedUserProfile).username?.toLowerCase().includes(searchTerm.toLowerCase()) || '')
      );
    }

    // Filter by criteria
    if (filterBy !== "all") {
      switch (filterBy) {
        case "high_income":
          filtered = filtered.filter(user => user.Annual_Income >= 100000);
          break;
        case "high_savings":
          filtered = filtered.filter(user => user.Current_Savings >= 50000);
          break;
        case "high_risk":
          filtered = filtered.filter(user => user.Risk_Tolerance === "High");
          break;
        case "married":
          filtered = filtered.filter(user => user.Marital_Status === "Married");
          break;
        case "with_dependents":
          filtered = filtered.filter(user => (user.Number_of_Dependents || 0) > 0);
          break;
      }
    }

    setFilteredUsers(filtered);
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-AU', {
      style: 'currency',
      currency: 'AUD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }).format(amount);
  };

  const formatDate = (dateString: string) => {
    if (!dateString) return 'N/A';
    const d = new Date(dateString);
    if (isNaN(d.getTime())) return 'N/A';
    return d.toLocaleDateString();
  };

  const handleLogout = () => {
    adminService.logout();
    navigate('/');
  };

  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
    setShowLoginModal(false);
    loadUsers();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Loading user data...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <AdminLoginModal 
        onClose={() => navigate('/')} 
        onSuccess={handleLoginSuccess} 
      />
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">Admin Dashboard</h1>
          <p className="text-xl text-gray-600">Manage and view user profiles</p>
          <div className="mt-4 flex justify-center gap-4">
            <Button 
              onClick={() => window.location.href = '/user-manager'}
              className="flex items-center gap-2"
              variant="outline"
            >
              <User className="w-4 h-4" />
              User Manager
            </Button>
            <Button 
              onClick={handleLogout}
              variant="outline"
              className="flex items-center gap-2"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </Button>
          </div>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card>
            <CardContent className="p-6 text-center">
              <User className="h-8 w-8 mx-auto mb-2 text-blue-600" />
              <div className="text-2xl font-bold">{users.length}</div>
              <div className="text-sm text-gray-600">Total Users</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <DollarSign className="h-8 w-8 mx-auto mb-2 text-green-600" />
              <div className="text-2xl font-bold">
                {formatCurrency(users.reduce((sum, user) => sum + user.Annual_Income, 0))}
              </div>
              <div className="text-sm text-gray-600">Total Annual Income</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <DollarSign className="h-8 w-8 mx-auto mb-2 text-purple-600" />
              <div className="text-2xl font-bold">
                {formatCurrency(users.reduce((sum, user) => sum + user.Current_Savings, 0))}
              </div>
              <div className="text-sm text-gray-600">Total Savings</div>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="p-6 text-center">
              <Calendar className="h-8 w-8 mx-auto mb-2 text-orange-600" />
              <div className="text-2xl font-bold">
                {users.length > 0 ? Math.round(users.reduce((sum, user) => sum + user.Age, 0) / users.length) : 0}
              </div>
              <div className="text-sm text-gray-600">Average Age</div>
            </CardContent>
          </Card>
        </div>

        {/* Filters */}
        <Card>
          <CardHeader>
            <CardTitle>User Management</CardTitle>
            <CardDescription>Search and filter user profiles</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex gap-4 mb-4">
              <div className="flex-1">
                <Label htmlFor="search">Search Users</Label>
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="search"
                    placeholder="Search by name or email..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="pl-10 h-12"
                  />
                </div>
              </div>
              <div className="w-64">
                <Label>Filter By</Label>
                <Select value={filterBy} onValueChange={setFilterBy}>
                  <SelectTrigger className="h-12">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Users</SelectItem>
                    <SelectItem value="high_income">High Income (≥$100k)</SelectItem>
                    <SelectItem value="high_savings">High Savings (≥$50k)</SelectItem>
                    <SelectItem value="high_risk">High Risk Tolerance</SelectItem>
                    <SelectItem value="married">Married</SelectItem>
                    <SelectItem value="with_dependents">With Dependents</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Users Table */}
            <div className="border rounded-lg">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Email</TableHead>
                    <TableHead>Age</TableHead>
                    <TableHead>Income</TableHead>
                    <TableHead>Savings</TableHead>
                    <TableHead>Risk</TableHead>
                    <TableHead>Joined</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredUsers.map((user) => (
                    <TableRow key={user.User_ID}>
                      <TableCell className="font-medium">{user.Name || 'N/A'}</TableCell>
                      <TableCell>{(user as ExtendedUserProfile).username || 'N/A'}</TableCell>
                      <TableCell>{user.Age}</TableCell>
                      <TableCell>{formatCurrency(user.Annual_Income)}</TableCell>
                      <TableCell>{formatCurrency(user.Current_Savings)}</TableCell>
                      <TableCell>
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          user.Risk_Tolerance === 'High' ? 'bg-red-100 text-red-800' :
                          user.Risk_Tolerance === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-green-100 text-green-800'
                        }`}>
                          {user.Risk_Tolerance || 'Not specified'}
                        </span>
                      </TableCell>
                      <TableCell>{user.created_at ? formatDate(user.created_at) : 'N/A'}</TableCell>
                      <TableCell>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedUser(user)}
                        >
                          View Details
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>

            {filteredUsers.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No users found matching your criteria.
              </div>
            )}
          </CardContent>
        </Card>

        {/* User Details Modal */}
        {selectedUser && (
          <Card>
            <CardHeader>
              <CardTitle>User Details: {selectedUser.Name || 'User'}</CardTitle>
              <CardDescription>Complete profile information</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Personal Information</h3>
                  <div className="space-y-2">
                    <div><strong>Name:</strong> {selectedUser.Name || 'N/A'}</div>
                    <div><strong>Email:</strong> {(selectedUser as ExtendedUserProfile).username || 'N/A'}</div>
                    <div><strong>Age:</strong> {selectedUser.Age || 'N/A'}</div>
                    <div><strong>Gender:</strong> {selectedUser.Gender || 'N/A'}</div>
                    <div><strong>Country:</strong> {selectedUser.Country || 'N/A'}</div>
                    <div><strong>Marital Status:</strong> {selectedUser.Marital_Status || 'N/A'}</div>
                    <div><strong>Dependents:</strong> {selectedUser.Number_of_Dependents || '0'}</div>
                  </div>
                </div>

                <div className="space-y-4">
                  <h3 className="text-lg font-semibold">Financial Information</h3>
                  <div className="space-y-2">
                    <div><strong>Annual Income:</strong> {formatCurrency(selectedUser.Annual_Income)}</div>
                    <div><strong>Current Savings:</strong> {formatCurrency(selectedUser.Current_Savings)}</div>
                    <div><strong>Monthly Contribution:</strong> {formatCurrency(selectedUser.Contribution_Amount || 0)}</div>
                    <div><strong>Employer Contribution:</strong> {formatCurrency(selectedUser.Employer_Contribution || 0)}</div>
                    <div><strong>Retirement Goal Age:</strong> {selectedUser.Retirement_Age_Goal || 'N/A'}</div>
                    <div><strong>Risk Tolerance:</strong> {selectedUser.Risk_Tolerance || 'Not specified'}</div>
                    <div><strong>Investment Experience:</strong> {selectedUser.Investment_Experience_Level || 'N/A'}</div>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex justify-end">
                <Button onClick={() => setSelectedUser(null)}>
                  Close
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
