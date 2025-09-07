import { BarChart3, Target, BookOpen, DollarSign, MessageCircle } from "lucide-react";
import { Button } from "@/components/ui/button";

export function QuickActions() {
  const actions = [
    {
      title: "Portfolio",
      subtitle: "View investments",
      icon: BarChart3,
      color: "bg-primary text-primary-foreground"
    },
    {
      title: "Goals",
      subtitle: "Set targets",
      icon: Target,
      color: "bg-success text-success-foreground"
    },
    {
      title: "Education",
      subtitle: "Learn & guides",
      icon: BookOpen,
      color: "bg-warning text-warning-foreground"
    },
    {
      title: "Benefits",
      subtitle: "Entitlements",
      icon: DollarSign,
      color: "bg-neutral text-neutral-foreground"
    }
  ];

  return (
    <div className="dashboard-card">
      <h3 className="text-2xl font-semibold text-card-foreground mb-6">Quick Actions</h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {actions.map((action) => {
          const Icon = action.icon;
          return (
            <Button
              key={action.title}
              variant="outline"
              className="h-auto p-6 flex flex-col items-center gap-3 hover:scale-105 transition-transform duration-200"
            >
              <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${action.color}`}>
                <Icon className="w-6 h-6" />
              </div>
              <div className="text-center">
                <div className="font-semibold text-base">{action.title}</div>
                <div className="text-sm text-muted-foreground">{action.subtitle}</div>
              </div>
            </Button>
          );
        })}
      </div>

      <div className="border-t border-card-border pt-6">
        <Button className="w-full bg-gradient-to-r from-primary to-primary/80 text-primary-foreground hover:from-primary/90 hover:to-primary/70 py-4 text-lg font-medium rounded-xl">
          <MessageCircle className="w-5 h-5 mr-2" />
          Ask AI Advisor
        </Button>
        <div className="text-center text-muted-foreground mt-3">
          Get instant answers to your superannuation questions
        </div>
      </div>
    </div>
  );
}