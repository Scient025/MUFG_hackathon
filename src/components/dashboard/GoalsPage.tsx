import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogDescription } from "@/components/ui/dialog";
import { Plus, Target, Calendar, DollarSign, Trash2 } from "lucide-react";
import { useState } from "react";

interface Goal {
  id: string;
  title: string;
  description: string;
  targetAmount: number;
  targetDate: string;
  currentAmount: number;
  priority: 'High' | 'Medium' | 'Low';
  type: 'Short-term' | 'Long-term';
  category: 'Retirement' | 'Non-retirement';
}

interface GoalsPageProps {
  user: any;
  onGoalChange: (goals: Goal[]) => void;
}

export function GoalsPage({ user, onGoalChange }: GoalsPageProps) {
  const [goals, setGoals] = useState<Goal[]>([
    {
      id: '1',
      title: 'Retirement by 65',
      description: 'Primary retirement goal',
      targetAmount: user.Projected_Pension_Amount || 0,
      targetDate: '2031-12-31',
      currentAmount: user.Current_Savings || 0,
      priority: 'High',
      type: 'Long-term',
      category: 'Retirement'
    }
  ]);

  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [newGoal, setNewGoal] = useState<Partial<Goal>>({
    title: '',
    description: '',
    targetAmount: 0,
    targetDate: '',
    currentAmount: 0,
    priority: 'Medium',
    type: 'Short-term',
    category: 'Non-retirement'
  });

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('en-AU', {
      style: 'currency',
      currency: 'AUD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(amount);
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'High': return 'bg-red-100 text-red-800 border-red-200';
      case 'Medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'Low': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getCategoryColor = (category: string) => {
    return category === 'Retirement' ? 'bg-blue-100 text-blue-800 border-blue-200' : 'bg-orange-100 text-orange-800 border-orange-200';
  };

  const getTypeColor = (type: string) => {
    return type === 'Long-term' ? 'bg-blue-100 text-blue-800 border-blue-200' : 'bg-purple-100 text-purple-800 border-purple-200';
  };

  const calculateRetirementProgress = () => {
    const currentSavings = user.Current_Savings || 0;
    const retirementTarget = user.Projected_Pension_Amount || 0;
    
    // Calculate non-retirement commitments
    const nonRetirementCommitments = goals
      .filter(g => g.category === 'Non-retirement')
      .reduce((sum, goal) => sum + (goal.targetAmount - goal.currentAmount), 0);
    
    // Calculate available funds for retirement after non-retirement commitments
    const availableForRetirement = Math.max(0, currentSavings - nonRetirementCommitments);
    
    // Calculate progress based on available funds vs retirement target
    const progress = retirementTarget > 0 ? (availableForRetirement / retirementTarget) * 100 : 0;
    
    return {
      progress: Math.round(Math.max(0, progress)),
      availableForRetirement,
      nonRetirementCommitments,
      retirementTarget
    };
  };

  const retirementProgress = calculateRetirementProgress();

  const calculateProgress = (current: number, target: number) => {
    return Math.min(Math.round((current / target) * 100), 100);
  };

  const addGoal = () => {
    console.log('addGoal called with newGoal:', newGoal);
    console.log('Validation check:', {
      hasTitle: !!newGoal.title,
      hasTargetAmount: !!newGoal.targetAmount,
      hasTargetDate: !!newGoal.targetDate
    });
    
    // Clear previous validation error
    setValidationError(null);
    
    if (newGoal.title && newGoal.targetAmount && newGoal.targetDate) {
      const goal: Goal = {
        id: Date.now().toString(),
        title: newGoal.title,
        description: newGoal.description || '',
        targetAmount: newGoal.targetAmount,
        targetDate: newGoal.targetDate,
        currentAmount: newGoal.currentAmount || 0,
        priority: newGoal.priority || 'Medium',
        type: newGoal.type || 'Short-term',
        category: newGoal.category || 'Non-retirement'
      };
      
      console.log('Creating goal:', goal);
      
      const updatedGoals = [...goals, goal];
      setGoals(updatedGoals);
      onGoalChange(updatedGoals);
      
      setNewGoal({
        title: '',
        description: '',
        targetAmount: 0,
        targetDate: '',
        currentAmount: 0,
        priority: 'Medium',
        type: 'Short-term',
        category: 'Non-retirement'
      });
      setIsDialogOpen(false);
      console.log('Goal added successfully');
    } else {
      console.log('Validation failed - missing required fields');
      setValidationError('Please fill in all required fields: Title, Target Amount, and Target Date');
    }
  };

  const deleteGoal = (goalId: string) => {
    const updatedGoals = goals.filter(goal => goal.id !== goalId);
    setGoals(updatedGoals);
    onGoalChange(updatedGoals);
  };

  const updateGoalAmount = (goalId: string, amount: number) => {
    const updatedGoals = goals.map(goal => 
      goal.id === goalId ? { ...goal, currentAmount: amount } : goal
    );
    setGoals(updatedGoals);
    onGoalChange(updatedGoals);
  };

  return (
    <div className="space-y-8">
      {/* Goals Overview */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground flex items-center gap-3">
            <Target className="w-7 h-7" />
            Your Financial Goals
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-3 gap-6 mb-6">
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-card-foreground">{goals.length}</div>
              <div className="text-muted-foreground text-lg">Total Goals</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-green-600">
                {goals.filter(g => calculateProgress(g.currentAmount, g.targetAmount) >= 100).length}
              </div>
              <div className="text-muted-foreground text-lg">Completed</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-blue-600">
                {goals.filter(g => g.priority === 'High').length}
              </div>
              <div className="text-muted-foreground text-lg">High Priority</div>
            </div>
          </div>

          <Dialog open={isDialogOpen} onOpenChange={(open) => {
            setIsDialogOpen(open);
            if (open) {
              setValidationError(null); // Clear validation error when opening dialog
            }
          }}>
            <DialogTrigger asChild>
              <Button className="w-full h-12 text-lg font-semibold">
                <Plus className="w-5 h-5 mr-2" />
                Add New Goal
              </Button>
            </DialogTrigger>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle className="text-2xl">Add New Financial Goal</DialogTitle>
                <DialogDescription>
                  Create a new financial goal to track your progress and stay motivated.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <label className="text-lg font-medium mb-2 block">Goal Title</label>
                  <Input
                    value={newGoal.title}
                    onChange={(e) => setNewGoal({ ...newGoal, title: e.target.value })}
                    placeholder="e.g., House Deposit"
                    className="text-lg h-12"
                  />
                </div>
                <div>
                  <label className="text-lg font-medium mb-2 block">Description</label>
                  <Textarea
                    value={newGoal.description}
                    onChange={(e) => setNewGoal({ ...newGoal, description: e.target.value })}
                    placeholder="Describe your goal..."
                    className="text-lg"
                  />
                </div>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-lg font-medium mb-2 block">Target Amount</label>
                    <Input
                      type="number"
                      value={newGoal.targetAmount}
                      onChange={(e) => setNewGoal({ ...newGoal, targetAmount: Number(e.target.value) })}
                      placeholder="50000"
                      className="text-lg h-12"
                    />
                  </div>
                  <div>
                    <label className="text-lg font-medium mb-2 block">Current Amount</label>
                    <Input
                      type="number"
                      value={newGoal.currentAmount}
                      onChange={(e) => setNewGoal({ ...newGoal, currentAmount: Number(e.target.value) })}
                      placeholder="10000"
                      className="text-lg h-12"
                    />
                  </div>
                </div>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-lg font-medium mb-2 block">Target Date</label>
                    <Input
                      type="date"
                      value={newGoal.targetDate}
                      onChange={(e) => setNewGoal({ ...newGoal, targetDate: e.target.value })}
                      className="text-lg h-12"
                    />
                  </div>
                  <div>
                    <label className="text-lg font-medium mb-2 block">Priority</label>
                    <select
                      value={newGoal.priority}
                      onChange={(e) => setNewGoal({ ...newGoal, priority: e.target.value as any })}
                      className="w-full h-12 px-3 border border-input rounded-md text-lg"
                      aria-label="Select goal priority"
                    >
                      <option value="High">High</option>
                      <option value="Medium">Medium</option>
                      <option value="Low">Low</option>
                    </select>
                  </div>
                </div>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-lg font-medium mb-2 block">Goal Type</label>
                    <select
                      value={newGoal.type}
                      onChange={(e) => setNewGoal({ ...newGoal, type: e.target.value as any })}
                      className="w-full h-12 px-3 border border-input rounded-md text-lg"
                      aria-label="Select goal type"
                    >
                      <option value="Short-term">Short-term (1-3 years)</option>
                      <option value="Long-term">Long-term (5+ years)</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-lg font-medium mb-2 block">Category</label>
                    <select
                      value={newGoal.category}
                      onChange={(e) => setNewGoal({ ...newGoal, category: e.target.value as any })}
                      className="w-full h-12 px-3 border border-input rounded-md text-lg"
                      aria-label="Select goal category"
                    >
                      <option value="Non-retirement">Non-retirement Goal</option>
                      <option value="Retirement">Retirement Goal</option>
                    </select>
                  </div>
                </div>
                {validationError && (
                  <div className="bg-red-100 border border-red-300 rounded-lg p-3">
                    <div className="text-red-800 text-sm font-medium">{validationError}</div>
                  </div>
                )}
                <Button onClick={addGoal} className="w-full h-12 text-lg font-semibold">
                  Add Goal
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </CardContent>
      </Card>

      {/* Goals List */}
      <div className="grid lg:grid-cols-2 gap-6">
        {goals.map((goal) => {
          const progress = calculateProgress(goal.currentAmount, goal.targetAmount);
          const isCompleted = progress >= 100;
          
          return (
            <Card key={goal.id} className="dashboard-card">
              <CardHeader>
                <div className="flex justify-between items-start">
                  <div>
                    <CardTitle className="text-xl font-bold text-card-foreground mb-2">
                      {goal.title}
                    </CardTitle>
                    <p className="text-muted-foreground text-lg">{goal.description}</p>
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteGoal(goal.id)}
                    className="text-red-500 hover:text-red-700"
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-lg font-medium">Progress</span>
                    <span className="text-xl font-bold text-primary">{progress}%</span>
                  </div>
                  
                  <div className="w-full bg-muted rounded-full h-4">
                    <div 
                      className={`h-4 rounded-full transition-all duration-300 ${
                        isCompleted ? 'bg-green-500' : 'bg-primary'
                      }`}
                      style={{ width: `${progress}%` }}
                    ></div>
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4 text-center">
                    <div>
                      <div className="text-2xl font-bold text-card-foreground">
                        {formatCurrency(goal.currentAmount)}
                      </div>
                      <div className="text-muted-foreground text-sm">Current</div>
                    </div>
                    <div>
                      <div className="text-2xl font-bold text-card-foreground">
                        {formatCurrency(goal.targetAmount)}
                      </div>
                      <div className="text-muted-foreground text-sm">Target</div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 text-sm">
                    <Calendar className="w-4 h-4 text-muted-foreground" />
                    <span className="text-muted-foreground">Target: {new Date(goal.targetDate).toLocaleDateString()}</span>
                  </div>
                  
                  <div className="flex gap-2">
                    <div className={`px-3 py-1 rounded-full text-sm font-medium border ${getPriorityColor(goal.priority)}`}>
                      {goal.priority} Priority
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium border ${getTypeColor(goal.type)}`}>
                      {goal.type}
                    </div>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium border ${getCategoryColor(goal.category)}`}>
                      {goal.category}
                    </div>
                  </div>
                  
                  {!isCompleted && (
                    <div className="pt-2">
                      <label className="text-sm font-medium mb-1 block">Update Current Amount</label>
                      <div className="flex gap-2">
                        <Input
                          type="number"
                          value={goal.currentAmount}
                          onChange={(e) => updateGoalAmount(goal.id, Number(e.target.value))}
                          className="flex-1"
                        />
                        <Button size="sm" className="px-4">
                          Update
                        </Button>
                      </div>
                    </div>
                  )}
                  
                  {isCompleted && (
                    <div className="p-3 bg-green-100 border border-green-200 rounded-lg">
                      <div className="flex items-center gap-2 text-green-800 font-medium">
                        <Target className="w-4 h-4" />
                        Goal Completed! 🎉
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Goal Impact on Retirement */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground">
            Goal Impact on Retirement
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="p-6 bg-muted/30 rounded-xl">
            <h4 className="text-lg font-semibold mb-3">How your goals affect retirement projections:</h4>
            <div className="space-y-3 text-lg">
              <div className="flex justify-between">
                <span>Non-retirement goal commitments:</span>
                <span className="font-semibold">{formatCurrency(retirementProgress.nonRetirementCommitments)}</span>
              </div>
              <div className="flex justify-between">
                <span>Retirement target amount:</span>
                <span className="font-semibold">{formatCurrency(retirementProgress.retirementTarget)}</span>
              </div>
              <div className="flex justify-between">
                <span>Current retirement savings:</span>
                <span className="font-semibold">{formatCurrency(user.Current_Savings || 0)}</span>
              </div>
              <div className="flex justify-between">
                <span>Available for retirement:</span>
                <span className="font-semibold">{formatCurrency(retirementProgress.availableForRetirement)}</span>
              </div>
              <div className="flex justify-between">
                <span>Retirement goal progress:</span>
                <span className={`font-semibold ${retirementProgress.progress >= 100 ? 'text-green-600' : retirementProgress.progress >= 50 ? 'text-yellow-600' : 'text-red-600'}`}>
                  {retirementProgress.progress}%
                </span>
              </div>
              
              {/* Progress Bar */}
              <div className="mt-4">
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div 
                    className={`h-3 rounded-full transition-all duration-300 ${
                      retirementProgress.progress >= 100 ? 'bg-green-500' : 
                      retirementProgress.progress >= 50 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min(retirementProgress.progress, 100)}%` }}
                  ></div>
                </div>
                <div className="text-sm text-muted-foreground mt-2">
                  {retirementProgress.progress >= 100 ? '🎉 Retirement goal achieved!' : 
                   retirementProgress.progress >= 50 ? '📈 Good progress on retirement savings' : 
                   '⚠️ Consider reducing non-retirement commitments to improve retirement progress'}
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
