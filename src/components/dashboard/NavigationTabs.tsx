import { useState } from "react";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BarChart3, PieChart, Target, BookOpen } from "lucide-react";

interface NavigationTabsProps {
  activeTab: string;
  onTabChange: (tab: string) => void;
}

export function NavigationTabs({ activeTab, onTabChange }: NavigationTabsProps) {
  const tabs = [
    {
      id: "performance",
      label: "Performance",
      icon: BarChart3
    },
    {
      id: "allocation",
      label: "Allocation", 
      icon: PieChart
    },
    {
      id: "goals",
      label: "Goals",
      icon: Target
    },
    {
      id: "learn",
      label: "Learn",
      icon: BookOpen
    }
  ];

  return (
    <div className="dashboard-card mb-8">
      <Tabs value={activeTab} onValueChange={onTabChange} className="w-full">
        <TabsList className="grid w-full grid-cols-4 h-auto p-2 bg-muted/50">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <TabsTrigger
                key={tab.id}
                value={tab.id}
                className="flex flex-col items-center gap-2 py-4 px-6 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground text-lg"
              >
                <Icon className="w-5 h-5" />
                <span className="font-medium">{tab.label}</span>
              </TabsTrigger>
            );
          })}
        </TabsList>
      </Tabs>
    </div>
  );
}