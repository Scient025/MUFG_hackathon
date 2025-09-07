import os
import json
import requests
from typing import Dict, Any, List
import asyncio
import aiohttp
import google.generativeai as genai
from inference import SuperannuationInference


# Try to load .env file, but don't fail if it doesn't exist
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

class SuperannuationChatRouter:
    def __init__(self, api_base_url: str = "http://localhost:8000", gemini_api_key: str = None):
        self.api_base_url = api_base_url
        self.inference_engine = SuperannuationInference()
        
        # Configure Gemini API
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or "AIzaSyDBVO_5h4L6j2-M1q-1PecgC42sFQYWv0w"
        
        if self.gemini_api_key and self.gemini_api_key != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print(f"✅ Gemini API configured successfully")
            except Exception as e:
                print(f"❌ Error configuring Gemini API: {e}")
                self.model = None
        else:
            print("⚠️ GEMINI_API_KEY not set. LLM features will not work.")
            self.model = None
        
        # Response length configuration (in tokens)
        self.response_lengths = {
            "risk": 300,        # Shorter for risk questions
            "contribution": 400, # Medium for contribution analysis
            "projection": 500,   # Longer for retirement planning
            "peer": 350,        # Medium for peer comparisons
            "general": 400       # Default length
        }
        
        # Prompt length control (concise vs detailed)
        self.concise_mode = True  # Set to False for detailed responses
    
    async def query_gemini(self, prompt: str, max_tokens: int = 500) -> str:
        """Query Gemini LLM via Google Generative AI API with configurable response length"""
        if not self.model:
            return "I'm sorry, but I need to be configured with a Gemini API key to provide AI responses."
        
        try:
            # Configure generation parameters for response length control
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,  # Control response length
                temperature=0.7,               # Creativity level
                top_p=0.8,                     # Nucleus sampling
                top_k=40                       # Top-k sampling
            )
            
            # Generate content using Gemini with length control
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config
            )
            
            if response and response.text:
                return response.text.strip()
            else:
                return "I couldn't generate a response. Please try again."
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return f"I'm currently unable to connect to the AI service. Error: {str(e)}"
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context from ML models"""
        try:
            # Get user profile
            user_profile = self.inference_engine.get_user_profile(user_id)
            
            # Get ML predictions
            risk_pred = self.inference_engine.predict_risk_tolerance(user_id)
            segment = self.inference_engine.get_user_segment(user_id)
            projection = self.inference_engine.predict_pension_projection(user_id)
            summary = self.inference_engine.get_summary_stats(user_id)
            
            return {
                "profile": user_profile,
                "risk_prediction": risk_pred,
                "segment": segment,
                "projection": projection,
                "summary": summary
            }
        except Exception as e:
            return {"error": str(e)}
    
    def parse_query_intent(self, message: str) -> str:
        """Parse user query to determine intent"""
        message_lower = message.lower()
        
        # Risk-related queries
        if any(word in message_lower for word in ["risk", "risky", "safe", "conservative", "aggressive"]):
            return "risk"
        
        # Contribution/saving queries
        if any(word in message_lower for word in ["contribute", "save", "invest", "increase", "decrease", "more", "less"]):
            return "contribution"
        
        # Retirement projection queries
        if any(word in message_lower for word in ["retire", "retirement", "pension", "projection", "amount", "will i have"]):
            return "projection"
        
        # Peer comparison queries
        if any(word in message_lower for word in ["peer", "compare", "others", "similar", "group"]):
            return "peer"
        
        # General advice queries
        if any(word in message_lower for word in ["advice", "recommend", "should", "what", "how"]):
            return "advice"
        
        return "general"
    
    async def handle_risk_query(self, user_id: str, message: str) -> str:
        """Handle risk-related queries"""
        context = await self.get_user_context(user_id)
        
        if "error" in context:
            return f"I couldn't retrieve your information: {context['error']}"
        
        risk_data = context["risk_prediction"]
        user_profile = context["profile"]
        summary = context["summary"]
        projection = context["projection"]
        
        # Calculate risk appropriateness based on age
        age = user_profile['Age']
        years_to_retirement = projection.get('years_to_retirement', 65 - age)
        
        # Determine age-appropriate risk level
        if age < 35:
            age_appropriate_risk = "High"
            risk_advice = "You have many years ahead, so you can afford higher risk for better growth potential."
        elif age < 50:
            age_appropriate_risk = "Medium"
            risk_advice = "You're in the middle of your career - balanced risk is ideal for steady growth."
        else:
            age_appropriate_risk = "Low"
            risk_advice = "You're approaching retirement - preserving capital becomes more important than growth."
        
        prompt = f"""
        You are an expert superannuation advisor with 20+ years of experience. Provide detailed, personalized advice about risk management.

        COMPREHENSIVE USER PROFILE:
        - Name: {user_profile.get('Name', user_id)}
        - Age: {age} years old
        - Years to Retirement: {years_to_retirement} years
        - Current Risk Tolerance: {risk_data['current_risk']}
        - Predicted Risk Tolerance: {risk_data['predicted_risk']}
        - Model Confidence: {risk_data['confidence']:.1f}%
        - Investment Experience: {user_profile.get('Investment_Experience_Level', 'Unknown')}
        - Current Investment Type: {user_profile.get('Investment_Type', 'Unknown')}
        - Annual Income: ${user_profile.get('Annual_Income', 0):,.0f}
        - Current Savings: ${user_profile.get('Current_Savings', 0):,.0f}
        - Monthly Contribution: ${user_profile.get('Contribution_Amount', 0):,.0f}
        - Marital Status: {user_profile.get('Marital_Status', 'Unknown')}
        - Dependents: {user_profile.get('Number_of_Dependents', 0)}
        - Home Ownership: {user_profile.get('Home_Ownership_Status', 'Unknown')}

        RISK ANALYSIS:
        - Age-Appropriate Risk Level: {age_appropriate_risk}
        - Risk Assessment: {risk_advice}
        - Current vs Predicted Risk Match: {'Yes' if risk_data['current_risk'] == risk_data['predicted_risk'] else 'No'}

        USER'S SPECIFIC QUESTION: "{message}"

        INSTRUCTIONS:
        1. Analyze their current risk profile in detail
        2. Compare it to age-appropriate risk levels
        3. Explain the implications of their risk tolerance
        4. Provide specific recommendations for their situation
        5. Include actionable next steps
        6. Address any concerns about changing risk profiles
        7. Use specific numbers and percentages from their profile
        8. Be encouraging but realistic
        9. Keep language conversational but professional

        Provide a comprehensive response that directly addresses their question with specific, actionable advice.
        """
        
        return await self.query_gemini(prompt, self.response_lengths["risk"])
    
    async def handle_contribution_query(self, user_id: str, message: str) -> str:
        """Handle contribution/saving queries"""
        context = await self.get_user_context(user_id)
        
        if "error" in context:
            return f"I couldn't retrieve your information: {context['error']}"
        
        user_profile = context["profile"]
        projection = context["projection"]
        summary = context["summary"]
        
        # Extract numbers from message for simulation
        import re
        numbers = re.findall(r'\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', message)
        extra_amount = 0
        if numbers:
            try:
                extra_amount = float(numbers[0].replace(',', ''))
            except:
                pass
        
        # Run simulation if amount found
        if extra_amount > 0:
            sim_projection = self.inference_engine.predict_pension_projection(user_id, extra_amount)
            improvement = sim_projection['improvement']
        else:
            improvement = 0
        
        # Calculate contribution analysis
        current_annual_contrib = user_profile.get('Contribution_Amount', 0) * 12
        annual_income = user_profile.get('Annual_Income', 0)
        contribution_rate = (current_annual_contrib / annual_income * 100) if annual_income > 0 else 0
        
        # Determine if contributions are adequate
        if contribution_rate >= 12:
            contrib_assessment = "Excellent - You're contributing above the recommended 9.5%"
        elif contribution_rate >= 9.5:
            contrib_assessment = "Good - You're meeting the minimum requirement"
        else:
            contrib_assessment = "Below recommended - Consider increasing contributions"
        
        prompt = f"""
        You are an expert superannuation advisor with deep knowledge of Australian superannuation laws and strategies. Provide detailed contribution analysis and recommendations.

        COMPREHENSIVE USER PROFILE:
        - Name: {user_profile.get('Name', user_id)}
        - Age: {user_profile.get('Age', 0)} years old
        - Annual Income: ${annual_income:,.0f}
        - Current Monthly Contribution: ${user_profile.get('Contribution_Amount', 0):,.0f}
        - Current Annual Contribution: ${current_annual_contrib:,.0f}
        - Employer Contribution: ${user_profile.get('Employer_Contribution', 0):,.0f}
        - Total Annual Contribution: ${user_profile.get('Total_Annual_Contribution', 0):,.0f}
        - Current Savings: ${user_profile.get('Current_Savings', 0):,.0f}
        - Years to Retirement: {projection.get('years_to_retirement', 0)}
        - Risk Tolerance: {user_profile.get('Risk_Tolerance', 'Unknown')}

        CONTRIBUTION ANALYSIS:
        - Current Contribution Rate: {contribution_rate:.1f}% of income
        - Assessment: {contrib_assessment}
        - Recommended Rate: 9.5% minimum, 12-15% optimal
        - Current Projected Pension: ${projection.get('adjusted_projection', 0):,.0f}
        - Monthly Income at Retirement: ${projection.get('monthly_income_at_retirement', 0):,.0f}

        SIMULATION RESULTS (if applicable):
        - Proposed Additional Monthly: ${extra_amount:,.0f}
        - Additional Annual Contribution: ${extra_amount * 12:,.0f}
        - Pension Improvement: ${improvement:,.0f}
        - New Projected Pension: ${projection.get('adjusted_projection', 0) + improvement:,.0f}

        USER'S SPECIFIC QUESTION: "{message}"

        INSTRUCTIONS:
        1. Analyze their current contribution strategy in detail
        2. Compare to industry benchmarks and recommendations
        3. Calculate the impact of any proposed changes
        4. Provide specific, actionable recommendations
        5. Consider their age, income, and retirement timeline
        6. Explain tax benefits and employer matching opportunities
        7. Address any concerns about affordability
        8. Include concrete next steps they can take
        9. Use specific numbers and calculations
        10. Be encouraging but realistic about expectations

        Provide a comprehensive response with specific calculations and actionable advice.
        """
        
        return await self.query_gemini(prompt, self.response_lengths["contribution"])
    
    async def handle_projection_query(self, user_id: str, message: str) -> str:
        """Handle retirement projection queries"""
        context = await self.get_user_context(user_id)
        
        if "error" in context:
            return f"I couldn't retrieve your information: {context['error']}"
        
        projection = context["projection"]
        summary = context["summary"]
        user_profile = context["profile"]
        
        # Calculate lifestyle analysis
        monthly_income = projection.get('monthly_income_at_retirement', 0)
        annual_income = user_profile.get('Annual_Income', 0)
        income_replacement_ratio = (monthly_income * 12) / annual_income if annual_income > 0 else 0
        
        # Determine lifestyle adequacy
        if income_replacement_ratio >= 0.7:
            lifestyle_assessment = "Excellent - You'll maintain a comfortable lifestyle"
        elif income_replacement_ratio >= 0.5:
            lifestyle_assessment = "Good - You'll have a decent retirement income"
        else:
            lifestyle_assessment = "Below recommended - Consider increasing savings"
        
        # Calculate years of coverage
        projected_pension = projection.get('adjusted_projection', 0)
        years_of_coverage = projected_pension / (monthly_income * 12) if monthly_income > 0 else 0
        
        prompt = f"""
        You are an expert retirement planning advisor with extensive knowledge of Australian superannuation and retirement strategies. Provide comprehensive retirement projection analysis.

        COMPREHENSIVE USER PROFILE:
        - Name: {user_profile.get('Name', user_id)}
        - Age: {user_profile.get('Age', 0)} years old
        - Retirement Age Goal: {user_profile.get('Retirement_Age_Goal', 65)} years
        - Years to Retirement: {projection.get('years_to_retirement', 0)} years
        - Current Annual Income: ${annual_income:,.0f}
        - Current Savings: ${summary.get('current_savings', 0):,.0f}
        - Monthly Contribution: ${user_profile.get('Contribution_Amount', 0):,.0f}
        - Risk Tolerance: {user_profile.get('Risk_Tolerance', 'Unknown')}
        - Investment Type: {user_profile.get('Investment_Type', 'Unknown')}

        RETIREMENT PROJECTION ANALYSIS:
        - Projected Pension Amount: ${projection.get('adjusted_projection', 0):,.0f}
        - Monthly Income at Retirement: ${monthly_income:,.0f}
        - Annual Retirement Income: ${monthly_income * 12:,.0f}
        - Income Replacement Ratio: {income_replacement_ratio:.1%}
        - Lifestyle Assessment: {lifestyle_assessment}
        - Years of Coverage: {years_of_coverage:.1f} years
        - Progress to Goal: {summary.get('percent_to_goal', 0):.1f}%

        RETIREMENT READINESS INDICATORS:
        - ASFA Comfortable Standard: $62,000/year for couples, $44,000/year for singles
        - ASFA Modest Standard: $43,000/year for couples, $31,000/year for singles
        - Recommended Replacement Ratio: 70-80% of pre-retirement income

        USER'S SPECIFIC QUESTION: "{message}"

        INSTRUCTIONS:
        1. Analyze their retirement readiness in detail
        2. Compare projections to industry standards (ASFA benchmarks)
        3. Calculate income replacement ratios and lifestyle implications
        4. Assess whether they're on track for their retirement goals
        5. Identify potential gaps and improvement opportunities
        6. Provide specific strategies to improve their retirement outlook
        7. Consider inflation and longevity factors
        8. Address concerns about retirement timing and lifestyle
        9. Include concrete action steps they can take
        10. Use specific numbers and realistic scenarios
        11. Be encouraging but honest about their situation

        Provide a comprehensive retirement analysis with specific recommendations and actionable steps.
        """
        
        return await self.query_gemini(prompt, self.response_lengths["projection"])
    
    async def handle_peer_query(self, user_id: str, message: str) -> str:
        """Handle peer comparison queries"""
        context = await self.get_user_context(user_id)
        
        if "error" in context:
            return f"I couldn't retrieve your information: {context['error']}"
        
        segment = context["segment"]
        peer_stats = segment["peer_stats"]
        user_profile = context["profile"]
        summary = context["summary"]
        
        # Calculate performance metrics
        user_savings = user_profile.get('Current_Savings', 0)
        user_income = user_profile.get('Annual_Income', 0)
        user_contrib = user_profile.get('Contribution_Amount', 0) * 12
        
        avg_savings = peer_stats.get('avg_savings', 0)
        avg_income = peer_stats.get('avg_income', 0)
        avg_contrib = peer_stats.get('avg_contribution', 0)
        
        # Calculate relative performance
        savings_vs_peers = (user_savings / avg_savings * 100) if avg_savings > 0 else 0
        contrib_vs_peers = (user_contrib / avg_contrib * 100) if avg_contrib > 0 else 0
        
        # Determine performance level
        if savings_vs_peers >= 120:
            performance_level = "Above Average"
        elif savings_vs_peers >= 80:
            performance_level = "Average"
        else:
            performance_level = "Below Average"
        
        prompt = f"""
        You are an expert financial advisor specializing in peer benchmarking and comparative analysis. Provide detailed peer comparison insights.

        COMPREHENSIVE USER PROFILE:
        - Name: {user_profile.get('Name', user_id)}
        - Age: {user_profile.get('Age', 0)} years old
        - Annual Income: ${user_income:,.0f}
        - Current Savings: ${user_savings:,.0f}
        - Annual Contribution: ${user_contrib:,.0f}
        - Investment Type: {user_profile.get('Investment_Type', 'Unknown')}
        - Risk Tolerance: {user_profile.get('Risk_Tolerance', 'Unknown')}
        - Marital Status: {user_profile.get('Marital_Status', 'Unknown')}
        - Dependents: {user_profile.get('Number_of_Dependents', 0)}

        PEER GROUP ANALYSIS:
        - Total Peers in Group: {peer_stats.get('total_peers', 0)}
        - Average Age: {peer_stats.get('avg_age', 0):.1f} years
        - Average Income: ${avg_income:,.0f}
        - Average Savings: ${avg_savings:,.0f}
        - Average Annual Contribution: ${avg_contrib:,.0f}
        - Common Investment Types: {peer_stats.get('common_investment_types', 'Unknown')}
        - Risk Distribution: {peer_stats.get('risk_distribution', 'Unknown')}

        PERFORMANCE METRICS:
        - Your Savings vs Peers: {savings_vs_peers:.1f}% of peer average
        - Your Contributions vs Peers: {contrib_vs_peers:.1f}% of peer average
        - Overall Performance Level: {performance_level}
        - Contribution Percentile: {peer_stats.get('contribution_percentile', 0):.1f}%

        COMPARATIVE INSIGHTS:
        - Income Comparison: {'Above' if user_income > avg_income else 'Below'} peer average by ${abs(user_income - avg_income):,.0f}
        - Savings Comparison: {'Above' if user_savings > avg_savings else 'Below'} peer average by ${abs(user_savings - avg_savings):,.0f}
        - Contribution Comparison: {'Above' if user_contrib > avg_contrib else 'Below'} peer average by ${abs(user_contrib - avg_contrib):,.0f}

        USER'S SPECIFIC QUESTION: "{message}"

        INSTRUCTIONS:
        1. Analyze their performance relative to their peer group
        2. Highlight strengths and areas for improvement
        3. Explain what the peer data means for their situation
        4. Provide specific recommendations based on peer insights
        5. Compare their strategies to what's working for similar people
        6. Address any concerns about their relative performance
        7. Suggest actionable steps to improve their position
        8. Use specific numbers and percentages
        9. Be encouraging while being realistic
        10. Focus on actionable insights they can implement

        Provide a comprehensive peer comparison analysis with specific insights and recommendations.
        """
        
        return await self.query_gemini(prompt, self.response_lengths["peer"])
    
    async def handle_general_query(self, user_id: str, message: str) -> str:
        """Handle general queries"""
        context = await self.get_user_context(user_id)
        
        if "error" in context:
            return f"I couldn't retrieve your information: {context['error']}"
        
        summary = context["summary"]
        user_profile = context["profile"]
        projection = context["projection"]
        
        # Calculate key insights
        age = user_profile.get('Age', 0)
        years_to_retirement = projection.get('years_to_retirement', 65 - age)
        current_savings = summary.get('current_savings', 0)
        projected_pension = projection.get('adjusted_projection', 0)
        monthly_contrib = user_profile.get('Contribution_Amount', 0)
        
        # Determine overall financial health
        if projected_pension >= 500000 and years_to_retirement > 10:
            overall_assessment = "Excellent - You're well on track for a comfortable retirement"
        elif projected_pension >= 300000:
            overall_assessment = "Good - You're making solid progress toward retirement"
        else:
            overall_assessment = "Needs attention - Consider increasing contributions or adjusting strategy"
        
        prompt = f"""
        You are an expert superannuation advisor with comprehensive knowledge of Australian retirement planning, tax strategies, and investment principles. Provide detailed, personalized advice.

        COMPREHENSIVE USER PROFILE:
        - Name: {user_profile.get('Name', user_id)}
        - Age: {age} years old
        - Years to Retirement: {years_to_retirement} years
        - Annual Income: ${user_profile.get('Annual_Income', 0):,.0f}
        - Current Savings: ${current_savings:,.0f}
        - Monthly Contribution: ${monthly_contrib:,.0f}
        - Projected Pension: ${projected_pension:,.0f}
        - Risk Tolerance: {user_profile.get('Risk_Tolerance', 'Unknown')}
        - Investment Type: {user_profile.get('Investment_Type', 'Unknown')}
        - Marital Status: {user_profile.get('Marital_Status', 'Unknown')}
        - Dependents: {user_profile.get('Number_of_Dependents', 0)}
        - Home Ownership: {user_profile.get('Home_Ownership_Status', 'Unknown')}

        FINANCIAL HEALTH OVERVIEW:
        - Overall Assessment: {overall_assessment}
        - Progress to Goal: {summary.get('percent_to_goal', 0):.1f}%
        - Monthly Income at Retirement: ${projection.get('monthly_income_at_retirement', 0):,.0f}
        - Investment Experience: {user_profile.get('Investment_Experience_Level', 'Unknown')}

        USER'S SPECIFIC QUESTION: "{message}"

        INSTRUCTIONS:
        1. Provide comprehensive, personalized advice based on their complete profile
        2. Address their specific question with detailed explanations
        3. Include relevant calculations and projections
        4. Consider their age, income, family situation, and goals
        5. Provide actionable recommendations they can implement
        6. Explain complex concepts in simple, understandable terms
        7. Include relevant Australian superannuation rules and benefits
        8. Address any concerns or questions they might have
        9. Provide encouragement while being realistic
        10. Suggest next steps and follow-up actions
        11. Use specific numbers and examples from their profile

        Provide a comprehensive, helpful response that directly addresses their question with specific, actionable advice.
        """
        
        return await self.query_gemini(prompt, self.response_lengths["general"])
    
    async def route_query(self, user_id: str, message: str) -> str:
        """Main routing function for chat queries"""
        try:
            intent = self.parse_query_intent(message)
            
            if intent == "risk":
                response = await self.handle_risk_query(user_id, message)
            elif intent == "contribution":
                response = await self.handle_contribution_query(user_id, message)
            elif intent == "projection":
                response = await self.handle_projection_query(user_id, message)
            elif intent == "peer":
                response = await self.handle_peer_query(user_id, message)
            else:
                response = await self.handle_general_query(user_id, message)
            
            # Check if the response indicates API failure
            if "unable to connect" in response.lower() or "trouble connecting" in response.lower():
                # Provide a fallback response based on the intent
                return await self.get_fallback_response(user_id, message, intent)
            
            return response
                
        except Exception as e:
            return f"I'm sorry, I encountered an error processing your request: {str(e)}"
    
    async def get_fallback_response(self, user_id: str, message: str, intent: str) -> str:
        """Provide fallback responses when AI service is unavailable"""
        try:
            context = await self.get_user_context(user_id)
            if "error" in context:
                return f"I couldn't retrieve your information: {context['error']}"
            
            user_profile = context["profile"]
            summary = context["summary"]
            
            if intent == "risk":
                risk_level = user_profile.get('Risk_Tolerance', 'Unknown')
                return f"Based on your profile, you have a {risk_level} risk tolerance. This means you prefer {'conservative' if risk_level == 'Low' else 'moderate' if risk_level == 'Medium' else 'aggressive'} investment strategies. Your current investment type is {user_profile.get('Investment_Type', 'Unknown')}."
            
            elif intent == "contribution":
                current_contrib = user_profile.get('Contribution_Amount', 0)
                annual_income = user_profile.get('Annual_Income', 0)
                contribution_rate = (current_contrib * 12) / annual_income * 100 if annual_income > 0 else 0
                return f"You're currently contributing ${current_contrib:,.0f} per month (${current_contrib * 12:,.0f} annually), which is {contribution_rate:.1f}% of your annual income. The recommended contribution rate is typically 9.5% of your salary for superannuation."
            
            elif intent == "projection":
                current_savings = user_profile.get('Current_Savings', 0)
                projected_amount = user_profile.get('Projected_Pension_Amount', 0)
                return f"With your current savings of ${current_savings:,.0f}, you're projected to have approximately ${projected_amount:,.0f} at retirement. This projection assumes continued contributions and market growth."
            
            elif intent == "peer":
                age = user_profile.get('Age', 0)
                income = user_profile.get('Annual_Income', 0)
                return f"You're {age} years old with an annual income of ${income:,.0f}. People in your age group typically have similar financial goals and risk profiles. Your investment strategy should align with your age and income level."
            
            else:
                return f"Hello! I can see you're asking about your superannuation. You currently have ${summary.get('current_savings', 0):,.0f} in savings with a {summary.get('risk_tolerance', 'Unknown')} risk profile. While I'm experiencing some technical difficulties with my AI service, I can still provide basic information about your account. What specific aspect would you like to know more about?"
                
        except Exception as e:
            return f"I'm having trouble accessing your information right now. Please try again later or contact support if the issue persists."
