# AI-Powered Superannuation Advisor Dashboard

A comprehensive AI-powered superannuation advisor with ML models, real-time predictions, and conversational AI. Features a clean, card-based, accessible dashboard UI targeted at older users with high-contrast design, large fonts, and simple navigation.

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


## 📁 Project Structure

```
├── backend/               # Python ML backend
│   ├── api.py            # FastAPI server
│   ├── train.py          # ML model training
│   ├── chat_router.py    # AI chatbot integration
│   ├── inference.py      # ML model inference
│   ├── requirements.txt  # Python dependencies
│   └── setup.bat/sh      # Backend setup scripts
├── src/
│   ├── components/
│   │   ├── dashboard/    # Dashboard components
│   │   └── ui/           # Reusable UI components
│   ├── pages/            # Main application pages
│   ├── services/         # Data service layer
│   └── lib/              # Utility functions
└── case1.csv             # Dataset
```

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

### Quick Setup (Automated)

**Windows:**
```bash
cd backend
setup.bat
```

**Linux/Mac:**
```bash
cd backend
chmod +x setup.sh
./setup.sh
```

### Manual Setup

1. **Install Python Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Install Node.js Dependencies**
   ```bash
   npm install
   ```

3. **Train ML Models**
   ```bash
   cd backend
   python train.py
   ```

4. **Set up Environment Variables**
   Create a `.env` file in the backend folder with your Hugging Face token:
   ```
   HF_TOKEN=your_huggingface_token_here
   ```

5. **Start the ML Backend**
   ```bash
   cd backend
   python api.py
   ```

6. **Start the Frontend**
   ```bash
   npm run dev
   ```

7. **Access the Application**
   - Frontend: http://localhost:8080
   - Backend API: http://localhost:8000

### Testing Different Users
Use the user selection dropdown to switch between different user profiles (U1000-U1009)

## 🤖 ML Models & AI Features

### Machine Learning Models
- **KMeans Clustering**: User segmentation based on demographics and financial behavior
- **Logistic Regression**: Risk tolerance prediction using financial and demographic features
- **XGBoost**: Investment recommendations and pension projections

### AI Chatbot Integration
- **Mistral LLM**: Conversational AI powered by meta-llama/Llama-3-8b-chat
- **Context-Aware Responses**: AI pulls user data from ML models and CSV dataset
- **Natural Language Processing**: Understands queries about risk, contributions, projections, and peer comparisons

### API Endpoints
- `/user/{user_id}` - Get user profile
- `/predict/{user_id}` - Pension projection + risk category
- `/summary/{user_id}` - Dashboard summary statistics
- `/peer_stats/{user_id}` - Peer comparison data
- `/simulate` - Run projections with adjusted contributions
- `/risk/{user_id}` - Risk tolerance prediction
- `/segment/{user_id}` - User segmentation
- `/recommendations/{user_id}` - Investment recommendations

## 📊 Dataset Integration

The application uses real data from `case1.csv` with 500+ users including:
- Demographics (Age, Gender, Country, Employment Status)
- Financial Data (Income, Savings, Contributions, Investment Types)
- Risk Profiles and Investment Experience
- Pension Eligibility and Tax Benefits
- Transaction History and Portfolio Diversity

## 🔮 Future Enhancements

- **Real Excel Integration**: Connect to actual dataset
- **Real-time Data**: Live market data integration
- **Advanced Analytics**: More sophisticated financial modeling
- **Mobile App**: Native mobile application
- **Voice Interface**: Voice-activated AI assistant

## 📄 License

This project is part of the MUFG Hackathon and is designed for demonstration purposes.

---

