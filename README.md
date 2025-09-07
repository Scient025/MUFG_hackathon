# AI-Powered Superannuation Advisor Dashboard

A clean, card-based, accessible dashboard UI for an AI-powered superannuation advisor targeted at older users. Features high-contrast design, large fonts, and simple navigation with comprehensive financial planning tools.

## 🚀 Key Features

### 📊 Dashboard Overview
- **User Selection**: Dropdown with 10 sample User_IDs from the dataset for testing different user profiles
- **Summary Card**: Shows retirement amount, monthly increase needed, and peer comparison
- **Key Metrics**: Current balance, % to goal, estimated monthly income at 65, contributions
- **Peer Comparison**: Investment strategy comparison with age/risk groups

### 📈 Portfolio Management
- **Asset Allocation Cards**: Visual breakdown by asset type (Stocks, ETFs, Managed Funds, etc.)
- **Donut Chart**: Interactive asset allocation with hover tooltips
- **Growth Projection**: Line chart showing investment growth with milestones
- **Investment Types**: Display of user's current investment types and funds

### 🎯 Goals Planning
- **Milestone Cards**: Display user goals with progress tracking
- **Dynamic Goal Addition**: Add short-term and long-term goals
- **Progress Tracking**: Visual progress bars and completion status
- **Retirement Impact**: Shows how goals affect retirement projections

### 📚 Education & Benefits
- **Tax Benefits**: Explanation of superannuation tax advantages
- **Government Pension**: Age Pension eligibility and benefits
- **Private Pension**: Private pension options and strategies
- **Withdrawal Strategies**: Fixed, Dynamic, and Bucket strategies
- **"Explain This" Buttons**: Chatbot integration for plain language explanations

### ⚖️ Risk Management
- **Risk Tolerance Slider**: Low/Medium/High risk selection
- **Asset Allocation Input**: Interactive sliders for ideal allocation
- **Comparison Tool**: Compare user allocation with recommended allocation
- **Risk Assessment**: Comprehensive risk analysis and recommendations

### 🤖 AI Chatbot
- **Floating Action Button**: Easy access to AI advisor
- **Sample Questions**: Pre-built questions for common scenarios
- **ML Model Integration**: Simulated responses based on user data
- **Contextual Advice**: Personalized recommendations based on user profile


## 🛠️ Technical Implementation

### Data Service
- **Mock Data**: 10 sample users with realistic financial profiles
- **Peer Comparison**: Age group and risk group comparisons
- **Projection Calculations**: Retirement amount and contribution calculations
- **Investment Types**: Support for various Australian investment categories

### Components Structure
```
src/
├── components/dashboard/
│   ├── UserSelection.tsx          # User profile switcher
│   ├── DashboardHeader.tsx        # User info and progress
│   ├── SummaryCard.tsx            # Retirement projection
│   ├── MetricsGrid.tsx            # Key financial metrics
│   ├── NavigationTabs.tsx          # Main navigation
│   ├── PortfolioPage.tsx          # Asset allocation & growth
│   ├── GoalsPage.tsx              # Financial goals management
│   ├── EducationPage.tsx          # Benefits & education
│   ├── RiskPage.tsx               # Risk assessment
│   ├── ChatbotPage.tsx            # AI advisor interface
│   └── FloatingChatButton.tsx     # Quick chat access
├── services/
│   └── dataService.ts            # Data management & calculations
└── pages/
    └── Dashboard.tsx              # Main dashboard component
```

### Key Technologies
- **React 18**: Modern React with hooks
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Radix UI**: Accessible component primitives
- **Recharts**: Data visualization
- **Lucide React**: Consistent iconography

## 📱 User Experience

### For Older Users
- **Large, Clear Text**: Easy-to-read fonts and high contrast
- **Simple Navigation**: Intuitive tab-based interface
- **Visual Feedback**: Clear status indicators and progress bars
- **Helpful Tooltips**: Contextual information and explanations
- **Consistent Layout**: Predictable interface patterns

### Financial Planning Features
- **Scenario Planning**: "What if" calculations for contributions
- **Risk Assessment**: Personalized risk tolerance evaluation
- **Goal Tracking**: Visual progress toward financial goals
- **Peer Comparison**: Benchmark against similar users
- **Educational Resources**: Built-in learning materials

## 🚀 Getting Started

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Start Development Server**
   ```bash
   npm run dev
   ```

3. **Open Browser**
   Navigate to `http://localhost:5173`

4. **Test Different Users**
   Use the user selection dropdown to switch between different user profiles

## 📊 Sample Data

The application includes 10 sample users with varying:
- Ages (35-64)
- Risk profiles (Low, Medium, High)
- Investment types (Stocks, ETFs, Managed Funds, Fixed Income, Cash)
- Financial goals and circumstances
- Tax and pension eligibility

## 🔮 Future Enhancements

- **Real Excel Integration**: Connect to actual dataset
- **ML Model Integration**: Connect to XGBoost and Logistic Regression models
- **Real-time Data**: Live market data integration
- **Advanced Analytics**: More sophisticated financial modeling
- **Mobile App**: Native mobile application
- **Voice Interface**: Voice-activated AI assistant

## 📄 License

This project is part of the MUFG Hackathon and is designed for demonstration purposes.

---

**Built with ❤️ for accessible financial planning**