import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { GraduationCap, Shield, DollarSign, Info, CheckCircle, XCircle } from "lucide-react";
import { useState } from "react";

interface EducationPageProps {
  user: any;
}

export function EducationPage({ user }: EducationPageProps) {
  const [selectedCard, setSelectedCard] = useState<string | null>(null);

  const educationCards = [
    {
      id: 'tax-benefits',
      title: 'Tax Benefits',
      icon: DollarSign,
      status: user.taxBenefitsEligibility,
      description: 'Understanding superannuation tax advantages',
      details: {
        eligible: user.taxBenefitsEligibility,
        benefits: [
          'Concessional contributions taxed at 15% (vs your marginal tax rate)',
          'Non-concessional contributions are tax-free',
          'Investment earnings taxed at maximum 15%',
          'Tax-free withdrawals after age 60'
        ],
        explanation: 'Superannuation offers significant tax advantages. Your contributions are taxed at a lower rate than your regular income, and investment earnings within super are taxed at a maximum of 15%. After age 60, withdrawals are generally tax-free.'
      }
    },
    {
      id: 'government-pension',
      title: 'Government Pension Eligibility',
      icon: Shield,
      status: user.governmentPensionEligibility,
      description: 'Age Pension and government support',
      details: {
        eligible: user.governmentPensionEligibility,
        benefits: [
          'Age Pension provides income support',
          'Asset and income tests apply',
          'Maximum single rate: $1,096.70 per fortnight',
          'Maximum couple rate: $1,653.40 per fortnight'
        ],
        explanation: 'The Age Pension is a government payment for older Australians. Eligibility depends on age, residency, assets, and income. It provides a safety net for retirement income, but the amount depends on your other assets and income.'
      }
    },
    {
      id: 'private-pension',
      title: 'Private Pension Eligibility',
      icon: GraduationCap,
      status: user.privatePensionEligibility,
      description: 'Private pension and superannuation options',
      details: {
        eligible: user.privatePensionEligibility,
        benefits: [
          'Account-based pensions from super',
          'Transition to retirement pensions',
          'Flexible withdrawal options',
          'Tax-effective income streams'
        ],
        explanation: 'Private pensions allow you to convert your superannuation into a regular income stream. They offer flexibility in withdrawal amounts and can provide tax advantages, especially for those over 60.'
      }
    },
    {
      id: 'withdrawal-strategy',
      title: 'Withdrawal Strategy',
      icon: Info,
      status: true,
      description: 'Optimizing your superannuation withdrawals',
      details: {
        eligible: true,
        benefits: [
          'Fixed withdrawal strategy',
          'Dynamic withdrawal strategy',
          'Bucket strategy for different time horizons',
          'Tax optimization strategies'
        ],
        explanation: 'Withdrawal strategies help you manage your superannuation income in retirement. Fixed strategies provide predictable income, while dynamic strategies adjust based on market conditions. The bucket strategy separates assets by time horizon.'
      }
    }
  ];

  const getStatusIcon = (status: boolean) => {
    return status ? (
      <CheckCircle className="w-6 h-6 text-green-600" />
    ) : (
      <XCircle className="w-6 h-6 text-red-600" />
    );
  };

  const getStatusBadge = (status: boolean) => {
    return status ? (
      <Badge className="bg-green-100 text-green-800 border-green-200">
        Eligible
      </Badge>
    ) : (
      <Badge className="bg-red-100 text-red-800 border-red-200">
        Not Eligible
      </Badge>
    );
  };

  return (
    <div className="space-y-8">
      {/* Education Overview */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground flex items-center gap-3">
            <GraduationCap className="w-7 h-7" />
            Superannuation Education & Benefits
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-lg text-muted-foreground mb-6">
            Learn about your superannuation benefits, tax advantages, and retirement options. 
            Click "Explain This" on any card for detailed explanations in plain language.
          </p>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-card-foreground">
                {educationCards.filter(card => card.status).length}
              </div>
              <div className="text-muted-foreground text-lg">Benefits Available</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-green-600">
                {user.taxBenefitsEligibility ? 'Yes' : 'No'}
              </div>
              <div className="text-muted-foreground text-lg">Tax Benefits</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-blue-600">
                {user.governmentPensionEligibility ? 'Yes' : 'No'}
              </div>
              <div className="text-muted-foreground text-lg">Age Pension</div>
            </div>
            <div className="text-center p-4 bg-muted/30 rounded-xl">
              <div className="text-3xl font-bold text-purple-600">
                {user.privatePensionEligibility ? 'Yes' : 'No'}
              </div>
              <div className="text-muted-foreground text-lg">Private Pension</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Education Cards */}
      <div className="grid lg:grid-cols-2 gap-6">
        {educationCards.map((card) => {
          const Icon = card.icon;
          
          return (
            <Card key={card.id} className="dashboard-card">
              <CardHeader>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-3">
                    <div className={`p-3 rounded-xl ${card.status ? 'bg-green-100' : 'bg-red-100'}`}>
                      <Icon className={`w-6 h-6 ${card.status ? 'text-green-600' : 'text-red-600'}`} />
                    </div>
                    <div>
                      <CardTitle className="text-xl font-bold text-card-foreground">
                        {card.title}
                      </CardTitle>
                      <p className="text-muted-foreground text-lg">{card.description}</p>
                    </div>
                  </div>
                  {getStatusIcon(card.status)}
                </div>
                <div className="flex justify-between items-center">
                  {getStatusBadge(card.status)}
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-semibold text-lg mb-2">Key Benefits:</h4>
                    <ul className="space-y-2">
                      {card.details.benefits.map((benefit, index) => (
                        <li key={index} className="flex items-start gap-2 text-muted-foreground">
                          <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                          <span className="text-lg">{benefit}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                  
                  <Dialog>
                    <DialogTrigger asChild>
                      <Button 
                        className="w-full h-12 text-lg font-semibold"
                        onClick={() => setSelectedCard(card.id)}
                      >
                        <Info className="w-5 h-5 mr-2" />
                        Explain This
                      </Button>
                    </DialogTrigger>
                    <DialogContent className="max-w-3xl">
                      <DialogHeader>
                        <DialogTitle className="text-2xl flex items-center gap-3">
                          <Icon className="w-6 h-6" />
                          {card.title} - Detailed Explanation
                        </DialogTitle>
                      </DialogHeader>
                      <div className="space-y-6">
                        <div className="p-4 bg-muted/30 rounded-xl">
                          <h4 className="font-semibold text-lg mb-2">Your Eligibility Status:</h4>
                          <div className="flex items-center gap-2">
                            {getStatusIcon(card.status)}
                            <span className="text-lg font-medium">
                              {card.status ? 'You are eligible for this benefit' : 'You are not currently eligible'}
                            </span>
                          </div>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-lg mb-3">Plain Language Explanation:</h4>
                          <p className="text-lg text-muted-foreground leading-relaxed">
                            {card.details.explanation}
                          </p>
                        </div>
                        
                        <div>
                          <h4 className="font-semibold text-lg mb-3">What This Means for You:</h4>
                          <ul className="space-y-2">
                            {card.details.benefits.map((benefit, index) => (
                              <li key={index} className="flex items-start gap-2 text-lg">
                                <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                                <span>{benefit}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        {!card.status && (
                          <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-xl">
                            <h4 className="font-semibold text-lg mb-2 text-yellow-800">
                              Not Currently Eligible?
                            </h4>
                            <p className="text-yellow-700">
                              Don't worry! Eligibility can change based on your age, income, and circumstances. 
                              Consider speaking with a financial advisor about your options.
                            </p>
                          </div>
                        )}
                      </div>
                    </DialogContent>
                  </Dialog>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Additional Resources */}
      <Card className="dashboard-card">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-card-foreground">
            Additional Resources
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid md:grid-cols-2 gap-6">
            <div className="p-4 bg-card border border-card-border rounded-xl hover:shadow-md transition-shadow cursor-pointer">
              <h4 className="font-semibold text-lg mb-2">Superannuation Basics</h4>
              <p className="text-muted-foreground mb-3">
                Learn the fundamentals of how superannuation works in Australia
              </p>
              <span className="text-primary font-medium text-lg">5 min read →</span>
            </div>
            
            <div className="p-4 bg-card border border-card-border rounded-xl hover:shadow-md transition-shadow cursor-pointer">
              <h4 className="font-semibold text-lg mb-2">Tax Optimization Strategies</h4>
              <p className="text-muted-foreground mb-3">
                Maximize your tax benefits with smart superannuation strategies
              </p>
              <span className="text-primary font-medium text-lg">8 min read →</span>
            </div>
            
            <div className="p-4 bg-card border border-card-border rounded-xl hover:shadow-md transition-shadow cursor-pointer">
              <h4 className="font-semibold text-lg mb-2">Retirement Planning Checklist</h4>
              <p className="text-muted-foreground mb-3">
                Essential steps to prepare for a comfortable retirement
              </p>
              <span className="text-primary font-medium text-lg">12 min read →</span>
            </div>
            
            <div className="p-4 bg-card border border-card-border rounded-xl hover:shadow-md transition-shadow cursor-pointer">
              <h4 className="font-semibold text-lg mb-2">Withdrawal Strategies Guide</h4>
              <p className="text-muted-foreground mb-3">
                Choose the right strategy for accessing your superannuation
              </p>
              <span className="text-primary font-medium text-lg">10 min read →</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
