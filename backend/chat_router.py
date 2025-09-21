import os
import json
import requests
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import google.generativeai as genai
import logging
from inference import SuperannuationInference
from integrated_ml_pipeline import IntegratedMLPipeline


# Try to load .env file, but don't fail if it doesn't exist
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Shared prompt instructions for consistent formatting
SHARED_PROMPT_INSTRUCTIONS = """
You are a senior superannuation advisor with 20+ years of experience in financial planning.

IMPORTANT SCOPE RESTRICTIONS:
- ONLY answer questions related to superannuation, retirement planning, financial planning, investments, pensions, and personal finance
- If asked about topics outside of finance/superannuation (like sports, entertainment, general knowledge, etc.), politely decline and redirect to financial topics
- Say: "I'm specialized in superannuation and financial planning. I can help you with questions about your retirement savings, investment strategies, contribution advice, or any other financial planning matters. What would you like to know about your superannuation?"

Always answer in a way that is:
- **Clear, concise, and professional** but still conversational and approachable.
- **Structured** into sections with headings and bullet points.
- **Action-oriented**, giving the user concrete next steps.
- **Personalized**, using the numbers, predictions, and confidence levels provided.

Formatting rules:
1. Always begin with a short Summary (2–3 sentences) that captures the key insight for the user.
2. Break the answer into clear sections. Each section heading should be on its own line, followed by a blank line, then the content:
   
   Strengths
   
   - First strength point here
   - Second strength point here
   
   Areas for Improvement
   
   - First area to improve
   - Second area to improve
   
   Recommendations
   
   1. First recommendation
   2. Second recommendation
   
3. Use plain text headings WITHOUT any asterisks or special characters
4. For bullet points, use "- " at the start of each line
5. For numbered lists, use "1. ", "2. ", etc. at the start of each line
6. Always include proper line breaks between sections
7. If referencing ML predictions:
   - Explicitly state: "Our model predicts X with Y% confidence."
   - If prediction contradicts user profile, explain both perspectives
8. Always end with:
   
   Encouragement Statement
   
   A positive, forward-looking message about their retirement journey.

Tone guidelines:
- Encourage, don't lecture
- Use simple language without financial jargon
- Be specific with numbers and percentages
"""

# Separate instruction for greetings
GREETING_PROMPT = """
You are a friendly superannuation advisor. The user has just greeted you. 
Respond with a warm, brief greeting (1-2 sentences maximum) and ask how you can help with their retirement planning today.
Do NOT provide any analysis or detailed information unless specifically asked.
Keep it short, friendly, and conversational.
"""

class SuperannuationChatRouter:
    def __init__(self, api_base_url: str = "http://localhost:8000", gemini_api_key: str = None):
        self.api_base_url = api_base_url
        self.inference_engine = SuperannuationInference()
        self.advanced_ml_pipeline = IntegratedMLPipeline()
        
        # Configure Gemini API
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY") or "AIzaSyDBVO_5h4L6j2-M1q-1PecgC42sFQYWv0w"
        
        if self.gemini_api_key and self.gemini_api_key != "your_gemini_api_key_here":
            try:
                genai.configure(api_key=self.gemini_api_key)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("✅ Gemini API configured successfully")
            except Exception as e:
                logger.error(f"❌ Error configuring Gemini API: {e}")
                self.model = None
        else:
            logger.warning("⚠️ GEMINI_API_KEY not set. LLM features will not work.")
            self.model = None
        
        # Response length configuration (in tokens)
        self.base_response_lengths = {
            "risk": 900,
            "contribution": 1000,
            "projection": 1000,
            "peer": 800,
            "general": 900,
            "greeting": 150  # Short for greetings
        }
        
        # Greeting keywords for detection
        self.greeting_keywords = [
            'hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 
            'good evening', 'howdy', 'hola', 'bonjour', 'g\'day', 'sup', 'yo',
            'hi there', 'hello there', 'hey there'
        ]
    
    def is_simple_greeting(self, message: str) -> bool:
        """Check if the message is just a simple greeting without a financial question"""
        message_lower = message.lower().strip()
        
        # Remove punctuation for better matching
        message_clean = ''.join(c for c in message_lower if c.isalnum() or c.isspace())
        
        # Check if message is very short (typical of greetings)
        if len(message_clean.split()) <= 3:
            # Check if it matches common greetings
            for greeting in self.greeting_keywords:
                if greeting in message_clean or message_clean in greeting:
                    return True
        
        # Also check for exact matches
        if message_clean in self.greeting_keywords:
            return True
            
        return False
    
    def safe_get(self, data: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get a value from a dictionary, handling nested keys and missing data"""
        try:
            if isinstance(key, str) and '.' in key:
                # Handle nested keys like 'risk_prediction.current_risk'
                keys = key.split('.')
                value = data
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return default
                return value
            else:
                return data.get(key, default)
        except (KeyError, TypeError, AttributeError):
            return default
    
    def validate_ml_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean ML model outputs"""
        validated = {}
        
        # Validate risk prediction
        risk_pred = context.get("risk_prediction", {})
        validated["risk_prediction"] = {
            "current_risk": self.safe_get(risk_pred, "current_risk", "Unknown"),
            "predicted_risk": self.safe_get(risk_pred, "predicted_risk", "Unknown"),
            "confidence": self.safe_get(risk_pred, "confidence", 0.0)
        }
        
        # Validate segment data
        segment = context.get("segment", {})
        peer_stats = self.safe_get(segment, "peer_stats", {})
        validated["segment"] = {
            "peer_stats": {
                "total_peers": self.safe_get(peer_stats, "total_peers", 0),
                "avg_age": self.safe_get(peer_stats, "avg_age", 0.0),
                "avg_income": self.safe_get(peer_stats, "avg_income", 0.0),
                "avg_savings": self.safe_get(peer_stats, "avg_savings", 0.0),
                "avg_contribution": self.safe_get(peer_stats, "avg_contribution", 0.0),
                "common_investment_types": self.safe_get(peer_stats, "common_investment_types", "Unknown"),
                "risk_distribution": self.safe_get(peer_stats, "risk_distribution", "Unknown"),
                "contribution_percentile": self.safe_get(peer_stats, "contribution_percentile", 0.0)
            }
        }
        
        # Validate projection data
        projection = context.get("projection", {})
        validated["projection"] = {
            "adjusted_projection": self.safe_get(projection, "adjusted_projection", 0.0),
            "monthly_income_at_retirement": self.safe_get(projection, "monthly_income_at_retirement", 0.0),
            "years_to_retirement": self.safe_get(projection, "years_to_retirement", 0),
            "improvement": self.safe_get(projection, "improvement", 0.0)
        }
        
        # Keep other context data
        validated["profile"] = context.get("profile", {})
        validated["summary"] = context.get("summary", {})
        
        return validated
    
    async def query_gemini(self, prompt: str, max_tokens: int = 500) -> str:
        """Query Gemini LLM via Google Generative AI API with improved truncation handling"""
        if not self.model:
            return "I'm sorry, but I need to be configured with a Gemini API key to provide AI responses."
        
        try:
            # Configure generation parameters for response length control
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.5,
                top_p=0.8,
                top_k=40
            )
            
            # Generate content using Gemini with length control
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config
            )
            
            if response and response.text:
                # Clean and post-process the response
                cleaned_response = self.clean_response(response.text.strip())
                formatted_response = self.prettify_response(cleaned_response)
                
                # Check if response appears complete, if not, try with more tokens
                if not self.is_response_complete(formatted_response) and max_tokens < 3000:
                    logger.info(f"Response appears truncated, retrying with more tokens: {max_tokens} -> {max_tokens * 2}")
                    # Retry with 100% more tokens (doubled)
                    retry_config = genai.types.GenerationConfig(
                        max_output_tokens=int(max_tokens * 2),
                        temperature=0.5,
                        top_p=0.8,
                        top_k=40
                    )
                    
                    retry_response = self.model.generate_content(prompt, generation_config=retry_config)
                    if retry_response and retry_response.text:
                        cleaned_retry = self.clean_response(retry_response.text.strip())
                        formatted_retry = self.prettify_response(cleaned_retry)
                        if self.is_response_complete(formatted_retry):
                            return formatted_retry
                
                return formatted_response
            else:
                return "I couldn't generate a response. Please try again."
                
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"I'm currently unable to connect to the AI service. Error: {str(e)}"
    
    def clean_response(self, response: str) -> str:
        """Clean up response text by removing regex artifacts and fixing truncation issues"""
        import re
        
        # Remove common regex artifacts and formatting issues
        cleaned = response
        
        # Remove any leftover regex patterns or escaped characters (more comprehensive)
        cleaned = re.sub(r'\\[a-zA-Z]', '', cleaned)
        cleaned = re.sub(r'\\[0-9]', '', cleaned)
        cleaned = re.sub(r'\\[^a-zA-Z0-9]', '', cleaned)
        cleaned = re.sub(r'\{[^}]*\}', '', cleaned)
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        cleaned = re.sub(r'&[a-zA-Z]+;', '', cleaned)
        
        # Remove specific artifacts that appear in responses
        cleaned = re.sub(r'USER\'S SPECIFIC QUESTION:.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'FOCUS ON.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'INSTRUCTIONS:.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'COMPREHENSIVE USER PROFILE:.*', '', cleaned, flags=re.IGNORECASE)
        
        # Remove any remaining asterisks from text (convert **text** to text)
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        
        # Remove any remaining prompt template remnants
        cleaned = re.sub(r'^.*?(?=Summary|Hello|Based on)', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Fix common truncation issues
        if cleaned and not cleaned.endswith(('.', '!', '?', ':', ';', '...')):
            last_sentence = cleaned.split('.')[-1].strip()
            if len(last_sentence) > 10 and not any(punct in last_sentence for punct in ['!', '?', ':', ';']):
                cleaned += "..."
        
        # Remove excessive whitespace but preserve paragraph breaks
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def prettify_response(self, response: str) -> str:
        """Post-process response for better formatting and readability with truncation handling"""
        import re
        
        # First, clean up any remaining artifacts
        response = re.sub(r'\\[a-zA-Z0-9]', '', response)
        response = re.sub(r'\\[^a-zA-Z0-9]', '', response)
        response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
        
        lines = response.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Handle section headings
            if line.startswith('**') and line.endswith('**'):
                heading = line[2:-2].strip()
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                formatted_lines.append(heading)
                formatted_lines.append('')
            
            # Ensure numbered points start on new lines with proper spacing
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                formatted_lines.append(line)
            
            # Add blank line before bullet points and ensure proper formatting
            elif line.startswith(('-', '•', '*')):
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                if line.startswith('*') and not line.startswith('**'):
                    line = '- ' + line[1:].strip()
                formatted_lines.append(line)
            
            else:
                formatted_lines.append(line)
        
        result = '\n'.join(formatted_lines)
        result = re.sub(r'\n\n\n+', '\n\n', result)
        
        return result.strip()
    
    def is_response_complete(self, response: str) -> bool:
        """Check if the response appears to be complete or truncated"""
        if not response:
            return False
        
        if response.endswith(('.', '!', '?', ':', ';')):
            return True
        
        completion_patterns = [
            "encouragement statement",
            "next steps",
            "recommendations",
            "summary",
            "conclusion",
            "feel free to ask",
            "let me know if",
            "happy to help"
        ]
        
        response_lower = response.lower()
        if any(pattern in response_lower for pattern in completion_patterns):
            return True
        
        last_sentence = response.split('.')[-1].strip()
        if len(last_sentence) > 20 and not any(punct in last_sentence for punct in ['!', '?', ':', ';']):
            return False
        
        return True
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context from ML models with validation"""
        try:
            # Get user profile - includes ALL fields from the database
            user_profile = self.inference_engine.get_user_profile(user_id)
            
            # Get basic ML predictions
            risk_pred = self.inference_engine.predict_risk_tolerance(user_id)
            segment = self.inference_engine.get_user_segment(user_id)
            projection = self.inference_engine.predict_pension_projection(user_id)
            summary = self.inference_engine.get_summary_stats(user_id)
            
            # Get advanced ML predictions from IntegratedMLPipeline
            advanced_analysis = {}
            try:
                advanced_analysis = self.advanced_ml_pipeline.get_comprehensive_user_analysis(user_id)
            except Exception as e:
                logger.warning(f"Could not get advanced ML analysis: {e}")
            
            raw_context = {
                "profile": user_profile,
                "risk_prediction": risk_pred,
                "segment": segment,
                "projection": projection,
                "summary": summary,
                "advanced_ml": advanced_analysis  # Add all advanced ML outputs
            }
            
            # Validate and clean the ML data
            validated_context = self.validate_ml_data(raw_context)
            validated_context["advanced_ml"] = advanced_analysis  # Keep advanced ML data
            
            # Log confidence levels for monitoring
            confidence = self.safe_get(risk_pred, "confidence", 0.0)
            if confidence < 0.5:
                logger.warning(f"Low confidence prediction for user {user_id}: {confidence:.2f}")
            
            return validated_context
            
        except Exception as e:
            logger.error(f"Error getting user context for {user_id}: {e}")
            return {"error": str(e)}
    
    def parse_query_intent(self, message: str) -> str:
        """Simplified intent parsing - only used for logging"""
        message_lower = message.lower().strip()
        
        if self.is_simple_greeting(message):
            return "greeting"
        elif any(word in message_lower for word in ["risk", "risky", "safe", "conservative"]):
            return "risk"  
        elif any(word in message_lower for word in ["contribute", "increase", "super", "save"]):
            return "contribution"
        elif any(word in message_lower for word in ["retire", "retirement", "pension"]):
            return "projection"
        else:
            return "general"
    
    async def handle_greeting(self, user_id: str, message: str) -> str:
        """Handle simple greetings with a brief, friendly response"""
        greeting_prompt = f"""
        {GREETING_PROMPT}
        
        User's greeting: "{message}"
        
        Respond naturally and briefly. Keep it under 2 sentences.
        """
        
        # Use very short token limit for greetings
        return await self.query_gemini(greeting_prompt, max_tokens=150)
    
    async def route_query(self, user_id: str, message: str) -> str:
        """Unified routing function that handles ALL queries through appropriate handlers"""
        try:
            # Check if it's just a simple greeting
            if self.is_simple_greeting(message):
                return await self.handle_greeting(user_id, message)
            
            # For all other queries, get full context and use unified handler
            context = await self.get_user_context(user_id)
            if "error" in context:
                return f"I couldn't retrieve your information: {context['error']}"
            
            return await self.handle_unified_query(user_id, message, context)
                    
        except Exception as e:
            logger.error(f"Error routing query for user {user_id}: {e}")
            return f"I'm sorry, I encountered an error processing your request: {str(e)}"

    async def handle_unified_query(self, user_id: str, message: str, context: Dict[str, Any]) -> str:
        """Handle all queries with a single intelligent prompt that includes ALL ML data"""
        user_profile = context["profile"]
        summary = context["summary"]
        projection = context["projection"]
        risk_data = context["risk_prediction"]
        segment = context["segment"]
        advanced_ml = context.get("advanced_ml", {})
        
        # Extract ALL fields from user profile for comprehensive context
        age = self.safe_get(user_profile, 'Age', 0)
        monthly_contrib = self.safe_get(user_profile, 'Contribution_Amount', 0)
        annual_income = self.safe_get(user_profile, 'Annual_Income', 0)
        current_savings = self.safe_get(summary, 'current_savings', 0)
        projected_pension = self.safe_get(projection, 'adjusted_projection', 0)
        years_to_retirement = self.safe_get(projection, 'years_to_retirement', 0)
        contribution_rate = (monthly_contrib * 12) / annual_income * 100 if annual_income > 0 else 0
        
        # Peer comparison data
        peer_stats = self.safe_get(segment, "peer_stats", {})
        avg_savings = self.safe_get(peer_stats, 'avg_savings', 0)
        avg_contrib = self.safe_get(peer_stats, 'avg_contribution', 0)
        
        # Advanced ML metrics - Extract all available outputs
        financial_health = self.safe_get(advanced_ml, 'financial_health', {})
        churn_risk = self.safe_get(advanced_ml, 'churn_risk', {})
        anomaly_detection = self.safe_get(advanced_ml, 'anomaly_detection', {})
        fund_recommendations = self.safe_get(advanced_ml, 'fund_recommendations', {})
        monte_carlo = self.safe_get(advanced_ml, 'monte_carlo_simulation', {})
        peer_matching = self.safe_get(advanced_ml, 'peer_matching', {})
        portfolio_optimization = self.safe_get(advanced_ml, 'portfolio_optimization', {})
        
        unified_prompt = f"""
        {SHARED_PROMPT_INSTRUCTIONS}
        
        You are an expert superannuation advisor with access to comprehensive ML analysis. Provide personalized advice using ALL available data.

        CRITICAL SCOPE RESTRICTION:
        - ONLY answer questions about superannuation, retirement planning, investments, financial planning, pensions, and personal finance
        - For ANY non-financial questions (sports, weather, general knowledge, entertainment, technology unrelated to finance, etc.), respond EXACTLY with: "I'm specialized in superannuation and financial planning. I can help you with questions about your retirement savings, investment strategies, contribution advice, or any other financial planning matters. What would you like to know about your superannuation?"
        - Do NOT attempt to answer or engage with non-financial topics in any way
        
        IMPORTANT: When relevant to the user's question, include formatted tables using this exact format:
        
        | Header 1 | Header 2 | Header 3 |
        |----------|----------|----------|
        | Data 1   | Data 2   | Data 3   |
        | Data 4   | Data 5   | Data 6   |
        
        COMPREHENSIVE USER PROFILE:
        - Name: {self.safe_get(user_profile, 'Name', user_id)}
        - Age: {age}
        - Gender: {self.safe_get(user_profile, 'Gender', 'Not specified')}
        - Country: {self.safe_get(user_profile, 'Country', 'Not specified')}
        - Annual Income: ${annual_income:,.0f}
        - Current Savings: ${current_savings:,.0f}
        - Monthly Contribution: ${monthly_contrib:,.0f}
        - Employer Contribution: ${self.safe_get(user_profile, 'Employer_Contribution', 0):,.0f}
        - Years Contributed: {self.safe_get(user_profile, 'Years_Contributed', 0)}
        - Risk Tolerance: {self.safe_get(user_profile, 'Risk_Tolerance', 'Unknown')}
        - Investment Type: {self.safe_get(user_profile, 'Investment_Type', 'Unknown')}
        - Fund Name: {self.safe_get(user_profile, 'Fund_Name', 'Unknown')}
        - Insurance Coverage: {self.safe_get(user_profile, 'Insurance_Coverage', 'Not specified')}
        - Marital Status: {self.safe_get(user_profile, 'Marital_Status', 'Unknown')}
        - Number of Dependents: {self.safe_get(user_profile, 'Number_of_Dependents', 0)}
        - Education Level: {self.safe_get(user_profile, 'Education_Level', 'Unknown')}
        - Health Status: {self.safe_get(user_profile, 'Health_Status', 'Unknown')}
        - Employment Status: {self.safe_get(user_profile, 'Employment_Status', 'Unknown')}
        - Home Ownership: {self.safe_get(user_profile, 'Home_Ownership_Status', 'Unknown')}
        - Investment Experience: {self.safe_get(user_profile, 'Investment_Experience_Level', 'Unknown')}
        - Financial Goals: {self.safe_get(user_profile, 'Financial_Goals', 'Not specified')}
        - Pension Type: {self.safe_get(user_profile, 'Pension_Type', 'Unknown')}
        - Withdrawal Strategy: {self.safe_get(user_profile, 'Withdrawal_Strategy', 'Unknown')}
        - Debt Level: {self.safe_get(user_profile, 'Debt_Level', 'Unknown')}
        - Contribution Frequency: {self.safe_get(user_profile, 'Contribution_Frequency', 'Unknown')}
        - Portfolio Diversity Score: {self.safe_get(user_profile, 'Portfolio_Diversity_Score', 0):.2f}
        - Savings Rate: {self.safe_get(user_profile, 'Savings_Rate', 0):.2%}
        - Annual Return Rate: {self.safe_get(user_profile, 'Annual_Return_Rate', 0):.1f}%
        - Volatility: {self.safe_get(user_profile, 'Volatility', 0):.1f}%
        - Fees Percentage: {self.safe_get(user_profile, 'Fees_Percentage', 0):.2f}%
        - Life Expectancy Estimate: {self.safe_get(user_profile, 'Life_Expectancy_Estimate', 85)} years
        - Retirement Age Goal: {self.safe_get(user_profile, 'Retirement_Age_Goal', 65)} years
        - Years to Retirement: {years_to_retirement}
        - Projected Pension: ${projected_pension:,.0f}
        - Monthly Retirement Income: ${self.safe_get(projection, 'monthly_income_at_retirement', 0):,.0f}
        
        RISK ANALYSIS:
        - Current Risk Level: {self.safe_get(risk_data, 'current_risk', 'Unknown')}
        - Model Predicted Risk: {self.safe_get(risk_data, 'predicted_risk', 'Unknown')}
        - Prediction Confidence: {self.safe_get(risk_data, 'confidence', 0):.1%}
        
        FINANCIAL HEALTH ANALYSIS (Advanced ML):
        - Financial Health Score: {self.safe_get(financial_health, 'financial_health_score', 0):.0f}/100
        - Health Status: {self.safe_get(financial_health, 'status', 'Unknown')}
        - Key Recommendations: {', '.join(self.safe_get(financial_health, 'recommendations', [])[:3])}
        
        CHURN RISK ANALYSIS:
        - Churn Probability: {self.safe_get(churn_risk, 'churn_probability', 0):.1%}
        - Risk Level: {self.safe_get(churn_risk, 'risk_level', 'Unknown')}
        - Retention Recommendations: {', '.join(self.safe_get(churn_risk, 'recommendations', [])[:2])}
        
        ANOMALY DETECTION:
        - Anomaly Score: {self.safe_get(anomaly_detection, 'anomaly_percentage', 0):.1f}%
        - Status: {'Anomaly Detected' if self.safe_get(anomaly_detection, 'is_anomaly', False) else 'Normal Activity'}
        - Action Required: {self.safe_get(anomaly_detection, 'recommendations', ['Continue monitoring'])[0]}
        
        FUND RECOMMENDATIONS:
        - Current Fund: {self.safe_get(fund_recommendations, 'current_fund', 'Unknown')}
        - Top 3 Recommended Funds: {', '.join(self.safe_get(fund_recommendations, 'recommendations', [])[:3])}
        - Recommendation Basis: {self.safe_get(fund_recommendations, 'reasoning', 'Based on similar user profiles')}
        
        MONTE CARLO SIMULATION RESULTS:
        - Simulations Run: {self.safe_get(monte_carlo, 'simulations', 0)}
        - Mean Projected Balance: ${self.safe_get(monte_carlo, 'mean_result', 0):,.0f}
        - Median Projected Balance: ${self.safe_get(monte_carlo, 'median_result', 0):,.0f}
        - 10th Percentile: ${self.safe_get(monte_carlo, 'percentile_10', 0):,.0f}
        - 90th Percentile: ${self.safe_get(monte_carlo, 'percentile_90', 0):,.0f}
        - Probability of Meeting Target: {self.safe_get(monte_carlo, 'probability_above_target', 0):.1%}
        
        PEER MATCHING ANALYSIS:
        - Similar Peers Found: {self.safe_get(peer_matching, 'total_peers_found', 0)}
        - Peer Average Age: {self.safe_get(peer_matching.get('peer_stats', {}), 'avg_age', 0):.0f}
        - Peer Average Income: ${self.safe_get(peer_matching.get('peer_stats', {}), 'avg_income', 0):,.0f}
        - Peer Average Savings: ${self.safe_get(peer_matching.get('peer_stats', {}), 'avg_savings', 0):,.0f}
        - Peer Average Contribution: ${self.safe_get(peer_matching.get('peer_stats', {}), 'avg_contribution', 0):,.0f}
        - Your Contribution Percentile: {self.safe_get(peer_stats, 'contribution_percentile', 0):.1f}%
        
        PORTFOLIO OPTIMIZATION:
        - Expected Return: {self.safe_get(portfolio_optimization, 'expected_return', 0):.1f}%
        - Expected Volatility: {self.safe_get(portfolio_optimization, 'expected_volatility', 0):.1f}%
        - Risk Tolerance Used: {self.safe_get(portfolio_optimization, 'risk_tolerance', 'Unknown')}
        - Optimized Allocation: {self.safe_get(portfolio_optimization, 'optimized_allocation', [])}
        - Portfolio Recommendations: {', '.join(self.safe_get(portfolio_optimization, 'recommendations', [])[:3])}
        
        SUGGESTED TABLES FOR DIFFERENT QUESTION TYPES:
        
        For RISK questions, include:
        | Metric | Current | Recommended | ML Confidence |
        |--------|---------|-------------|---------------|
        | Risk Level | {self.safe_get(risk_data, 'current_risk', 'Unknown')} | {self.safe_get(risk_data, 'predicted_risk', 'Unknown')} | {self.safe_get(risk_data, 'confidence', 0):.1%} |
        | Age-Based Risk | - | {"High" if age < 35 else "Medium" if age < 50 else "Low"} | - |
        | Years to Retire | {years_to_retirement} | - | - |
        | Portfolio Volatility | {self.safe_get(user_profile, 'Volatility', 0):.1f}% | - | - |
        
        For CONTRIBUTION questions, include:
        | Contribution Analysis | Current | Recommended | Peer Average |
        |----------------------|---------|-------------|--------------|
        | Monthly Amount | ${monthly_contrib:,.0f} | ${annual_income * 0.12 / 12:,.0f} | ${avg_contrib / 12:,.0f} |
        | Annual Amount | ${monthly_contrib * 12:,.0f} | ${annual_income * 0.12:,.0f} | ${avg_contrib:,.0f} |
        | % of Income | {contribution_rate:.1f}% | 12.0% | {avg_contrib/self.safe_get(peer_stats, 'avg_income', 1)*100:.1f}% |
        | Employer Match | ${self.safe_get(user_profile, 'Employer_Contribution', 0):,.0f} | - | - |
        
        For PEER COMPARISON questions, include:
        | Metric | You | Peer Average | Your Percentile |
        |--------|-----|--------------|-----------------|
        | Age | {age} | {self.safe_get(peer_stats, 'avg_age', 0):.0f} | - |
        | Income | ${annual_income:,.0f} | ${self.safe_get(peer_stats, 'avg_income', 0):,.0f} | - |
        | Savings | ${current_savings:,.0f} | ${avg_savings:,.0f} | - |
        | Contributions | ${monthly_contrib * 12:,.0f} | ${avg_contrib:,.0f} | {self.safe_get(peer_stats, 'contribution_percentile', 0):.0f}% |
        | Financial Health | {self.safe_get(financial_health, 'financial_health_score', 0):.0f}/100 | - | Top {100-self.safe_get(financial_health, 'peer_percentile', 50):.0f}% |
        
        For FINANCIAL HEALTH questions, include:
        | Health Metric | Score/Value | Status | ML Insight |
        |---------------|-------------|--------|------------|
        | Overall Health | {self.safe_get(financial_health, 'financial_health_score', 0):.0f}/100 | {self.safe_get(financial_health, 'status', 'Unknown')} | Top {100-self.safe_get(financial_health, 'peer_percentile', 50):.0f}% of peers |
        | Churn Risk | {self.safe_get(churn_risk, 'churn_probability', 0):.1%} | {self.safe_get(churn_risk, 'risk_level', 'Unknown')} | {self.safe_get(churn_risk, 'recommendations', ['Monitor engagement'])[0]} |
        | Anomaly Detection | {self.safe_get(anomaly_detection, 'anomaly_percentage', 0):.1f}% | {'Alert' if self.safe_get(anomaly_detection, 'is_anomaly', False) else 'Normal'} | {self.safe_get(anomaly_detection, 'recommendations', ['Continue monitoring'])[0]} |
        | Savings Rate | {self.safe_get(user_profile, 'Savings_Rate', 0):.1%} | {"Good" if self.safe_get(user_profile, 'Savings_Rate', 0) > 0.1 else "Needs Improvement"} | - |
        
        For FUND RECOMMENDATION questions, include:
        | Fund Analysis | Current | ML Recommended | Expected Return | Risk Level |
        |---------------|---------|----------------|-----------------|------------|
        | Your Fund | {self.safe_get(fund_recommendations, 'current_fund', 'Unknown')} | - | {self.safe_get(user_profile, 'Annual_Return_Rate', 0):.1f}% | {self.safe_get(user_profile, 'Risk_Tolerance', 'Unknown')} |
        | Top Pick | - | {self.safe_get(fund_recommendations, 'recommendations', ['None'])[0] if self.safe_get(fund_recommendations, 'recommendations', []) else 'None'} | - | Based on profile |
        | Alternative 1 | - | {self.safe_get(fund_recommendations, 'recommendations', ['None'])[1] if len(self.safe_get(fund_recommendations, 'recommendations', [])) > 1 else 'None'} | - | Diversification |
        | Similar Users | - | {self.safe_get(fund_recommendations, 'similar_users_count', 0)} users | - | - |
        
        For MONTE CARLO/PROJECTION questions, include:
        | Retirement Scenario | Balance at {self.safe_get(user_profile, 'Retirement_Age_Goal', 65)} | Monthly Income | Probability |
        |--------------------|------------------------|----------------|-------------|
        | Conservative (P10) | ${self.safe_get(monte_carlo, 'percentile_10', 0):,.0f} | ${self.safe_get(monte_carlo, 'percentile_10', 0)/240:,.0f} | 90% chance above |
        | Expected (P50) | ${self.safe_get(monte_carlo, 'median_result', 0):,.0f} | ${self.safe_get(monte_carlo, 'median_result', 0)/240:,.0f} | 50% chance above |
        | Optimistic (P90) | ${self.safe_get(monte_carlo, 'percentile_90', 0):,.0f} | ${self.safe_get(monte_carlo, 'percentile_90', 0)/240:,.0f} | 10% chance above |
        | Target Achievement | ${projected_pension:,.0f} | ${self.safe_get(projection, 'monthly_income_at_retirement', 0):,.0f} | {self.safe_get(monte_carlo, 'probability_above_target', 0):.0f}% |
        
        For PORTFOLIO OPTIMIZATION questions, include:
        | Portfolio Metrics | Current | Optimized | Improvement |
        |------------------|---------|-----------|-------------|
        | Expected Return | {self.safe_get(user_profile, 'Annual_Return_Rate', 0):.1f}% | {self.safe_get(portfolio_optimization, 'expected_return', 0):.1f}% | {self.safe_get(portfolio_optimization, 'expected_return', 0) - self.safe_get(user_profile, 'Annual_Return_Rate', 0):.1f}% |
        | Volatility | {self.safe_get(user_profile, 'Volatility', 0):.1f}% | {self.safe_get(portfolio_optimization, 'expected_volatility', 0):.1f}% | {self.safe_get(portfolio_optimization, 'expected_volatility', 0) - self.safe_get(user_profile, 'Volatility', 0):.1f}% |
        | Sharpe Ratio | - | {self.safe_get(portfolio_optimization.get('portfolio_metrics', {}), 'sharpe_ratio', 0):.2f} | - |
        | Risk Level | {self.safe_get(user_profile, 'Risk_Tolerance', 'Unknown')} | {self.safe_get(portfolio_optimization, 'risk_tolerance', 'Unknown')} | Match |
        
        USER'S QUESTION: "{message}"
        
        INSTRUCTIONS:
        1. If this is a simple greeting, respond briefly and warmly (1-2 sentences max)
        2. For financial questions, use ALL the ML model outputs above to provide comprehensive, personalized advice
        3. Include relevant tables from the suggestions above when they add value
        4. Reference specific ML predictions with confidence levels when available
        5. Use the exact table format shown with proper markdown syntax
        6. Be specific with numbers, percentages, and recommendations from the ML models
        """
        
        # Adjust response length based on query type
        intent = self.parse_query_intent(message)
        base_length = self.base_response_lengths.get(intent, 900)
        
        # If it's a very short message but not a greeting, still provide reasonable detail
        if len(message.split()) < 5 and intent != "greeting":
            base_length = 600
        
        return await self.query_gemini(unified_prompt, base_length)