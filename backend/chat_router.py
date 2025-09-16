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
        
        # Response length configuration (in tokens) - Significantly increased to prevent truncation
        self.base_response_lengths = {
            "risk": 900,       # Doubled for risk questions
            "contribution": 1000, # Doubled for contribution analysis
            "projection": 1000,  # Doubled for retirement planning
            "peer": 800,       # Doubled for peer comparisons
            "general": 900,    # Doubled default length
            "greeting": 100     # Short for greetings
        }
        
        # Prompt length control (concise vs detailed)
        self.concise_mode = True  # Set to False for detailed responses
    
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
    
    def adjust_response_length(self, message: str, base_length: int, intent: str = "general") -> int:
        """Dynamically adjust response length based on query keywords and intent"""
        message_lower = message.lower()
        
        # For greetings, always use minimal length
        if intent == "greeting":
            return 100  # Short for greetings
        
        # For casual queries (short messages without question words), reduce length
        casual_indicators = len(message.strip()) < 20 and not any(word in message_lower for word in ["what", "how", "why", "when", "where", "can", "should", "would", "could"])
        if casual_indicators:
            return int(base_length * 0.6)  # Reduce by 40% for casual queries (less aggressive)
        
        # Keywords that suggest detailed responses
        detail_keywords = ["detailed", "step by step", "explain", "comprehensive", "thorough", "in depth", "analysis", "complete"]
        
        if any(keyword in message_lower for keyword in detail_keywords):
            return int(base_length * 1.5)  # Increase by 50% for detailed requests
        
        # Keywords that suggest concise responses
        concise_keywords = ["brief", "quick", "summary", "short", "simple"]
        
        if any(keyword in message_lower for keyword in concise_keywords):
            return int(base_length * 0.7)  # Decrease by 30% for concise requests
        
        return base_length
    
    async def query_gemini(self, prompt: str, max_tokens: int = 500) -> str:
        """Query Gemini LLM via Google Generative AI API with improved truncation handling"""
        if not self.model:
            return "I'm sorry, but I need to be configured with a Gemini API key to provide AI responses."
        
        try:
            # Configure generation parameters for response length control
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,  # Control response length
                temperature=0.5,               # Creativity level
                top_p=0.8,                     # Nucleus sampling
                top_k=40                       # Top-k sampling
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
        cleaned = re.sub(r'\\[a-zA-Z]', '', cleaned)  # Remove escaped characters
        cleaned = re.sub(r'\\[0-9]', '', cleaned)     # Remove escaped numbers
        cleaned = re.sub(r'\\[^a-zA-Z0-9]', '', cleaned)  # Remove other escaped chars
        cleaned = re.sub(r'\{[^}]*\}', '', cleaned)    # Remove any remaining {} patterns
        cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)  # Remove any remaining [] patterns
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)   # Remove parentheses content
        cleaned = re.sub(r'<[^>]*>', '', cleaned)      # Remove HTML-like tags
        cleaned = re.sub(r'&[a-zA-Z]+;', '', cleaned) # Remove HTML entities
        
        # Remove specific artifacts that appear in responses
        cleaned = re.sub(r'USER\'S SPECIFIC QUESTION:.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'FOCUS ON.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'INSTRUCTIONS:.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'COMPREHENSIVE USER PROFILE:.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'RISK ANALYSIS:.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'CONTRIBUTION ANALYSIS:.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'RETIREMENT PROJECTION ANALYSIS:.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'PEER GROUP ANALYSIS:.*', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'FINANCIAL HEALTH OVERVIEW:.*', '', cleaned, flags=re.IGNORECASE)
        
        # Remove any remaining asterisks from text (convert **text** to text)
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        
        # Remove any remaining prompt template remnants
        cleaned = re.sub(r'^.*?(?=Summary|Hello|Based on)', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
        
        # Fix common truncation issues
        # If response ends abruptly without proper punctuation, try to complete it
        if cleaned and not cleaned.endswith(('.', '!', '?', ':', ';', '...')):
            # Check if it looks like it was cut off mid-sentence
            last_sentence = cleaned.split('.')[-1].strip()
            if len(last_sentence) > 10 and not any(punct in last_sentence for punct in ['!', '?', ':', ';']):
                # Add ellipsis to indicate continuation
                cleaned += "..."
        
        # Remove excessive whitespace but preserve paragraph breaks
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Multiple spaces/tabs to single space
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)  # Multiple newlines to double newline
        
        return cleaned.strip()
    
    def create_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Create a formatted table for display in the chat"""
        if not headers or not rows:
            return ""
        
        # Create header row
        header_row = "| " + " | ".join(headers) + " |"
        
        # Create separator row
        separator = "|" + "|".join(["-" * (len(h) + 2) for h in headers]) + "|"
        
        # Create data rows
        data_rows = []
        for row in rows:
            if len(row) == len(headers):
                data_row = "| " + " | ".join([str(cell) for cell in row]) + " |"
                data_rows.append(data_row)
        
        return "\n".join([header_row, separator] + data_rows)
    
    def prettify_response(self, response: str) -> str:
        """Post-process response for better formatting and readability with truncation handling"""
        import re
        
        # First, clean up any remaining artifacts (comprehensive cleaning)
        response = re.sub(r'\\[a-zA-Z0-9]', '', response)  # Remove escaped characters and numbers
        response = re.sub(r'\\[^a-zA-Z0-9]', '', response)  # Remove other escaped chars
        response = re.sub(r'\{[^}]*\}', '', response)    # Remove {} patterns
        response = re.sub(r'\[[^\]]*\]', '', response)    # Remove [] patterns
        response = re.sub(r'\([^)]*\)', '', response)     # Remove parentheses content
        response = re.sub(r'<[^>]*>', '', response)       # Remove HTML-like tags
        response = re.sub(r'&[a-zA-Z]+;', '', response)   # Remove HTML entities
        
        # Remove asterisks from text (convert **text** to text)
        response = re.sub(r'\*\*([^*]+)\*\*', r'\1', response)
        
        lines = response.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            
            # Handle section headings (remove asterisks and ensure proper spacing)
            if line.startswith('**') and line.endswith('**'):
                # Remove asterisks to create plain text heading
                heading = line[2:-2].strip()
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                formatted_lines.append(heading)
                formatted_lines.append('')  # Add blank line after heading
            
            # Ensure numbered points start on new lines with proper spacing
            elif line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                formatted_lines.append(line)
            
            # Add blank line before bullet points and ensure proper formatting
            elif line.startswith(('-', '•', '*')):
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                # Ensure bullet points are properly formatted
                if line.startswith('*') and not line.startswith('**'):
                    line = '- ' + line[1:].strip()  # Convert * to - for consistency
                formatted_lines.append(line)
            
            else:
                formatted_lines.append(line)
        
        # Join lines and normalize spacing
        result = '\n'.join(formatted_lines)
        
        # Collapse excessive blank lines (more than 2 consecutive)
        result = re.sub(r'\n\n\n+', '\n\n', result)
        
        # Final cleanup - remove any remaining artifacts (comprehensive)
        result = re.sub(r'USER\'S SPECIFIC QUESTION:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'FOCUS ON.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'INSTRUCTIONS:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'COMPREHENSIVE USER PROFILE:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'RISK ANALYSIS:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'CONTRIBUTION ANALYSIS:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'RETIREMENT PROJECTION ANALYSIS:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'PEER GROUP ANALYSIS:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'FINANCIAL HEALTH OVERVIEW:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'SIMULATION RESULTS.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'PERFORMANCE METRICS:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'COMPARATIVE INSIGHTS:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'INFLATION CONSIDERATIONS:.*', '', result, flags=re.IGNORECASE)
        result = re.sub(r'RETIREMENT READINESS INDICATORS:.*', '', result, flags=re.IGNORECASE)
        
        # Handle truncation - if response seems cut off, add indication
        if result and not result.endswith(('.', '!', '?', ':', ';', '...')):
            # Check if the last sentence is incomplete
            last_sentence = result.split('.')[-1].strip()
            if len(last_sentence) > 15 and not any(punct in last_sentence for punct in ['!', '?', ':', ';']):
                result += "..."
        
        # Final formatting pass to ensure proper spacing
        result = self.finalize_formatting(result)
        
        return result.strip()
    
    def finalize_formatting(self, response: str) -> str:
        """Final formatting pass to ensure proper spacing and structure"""
        import re
        
        lines = response.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Handle section headings (remove asterisks and ensure proper spacing)
            if line.startswith('**') and line.endswith('**'):
                # Remove asterisks to create plain text heading
                heading = line[2:-2].strip()
                if formatted_lines and formatted_lines[-1].strip():
                    formatted_lines.append('')
                formatted_lines.append(heading)
                # Add blank line after
                formatted_lines.append('')
            
            # Ensure bullet points are properly formatted
            elif line.startswith(('*', '-', '•')):
                # Convert * to - for consistency (unless it's **bold**)
                if line.startswith('*') and not line.startswith('**'):
                    line = '- ' + line[1:].strip()
                elif line.startswith('•'):
                    line = '- ' + line[1:].strip()
                elif line.startswith('-') and not line.startswith('- '):
                    line = '- ' + line[1:].strip()
                
                formatted_lines.append(line)
            
            # Ensure numbered points are properly formatted
            elif re.match(r'^\d+\.', line):
                if not re.match(r'^\d+\. ', line):
                    line = re.sub(r'^(\d+)\.', r'\1. ', line)
                formatted_lines.append(line)
            
            else:
                formatted_lines.append(line)
        
        # Join and clean up spacing
        result = '\n'.join(formatted_lines)
        
        # Ensure proper spacing around sections
        result = re.sub(r'\n\n\n+', '\n\n', result)  # Max 2 consecutive newlines
        
        return result
    
    def is_response_complete(self, response: str) -> bool:
        """Check if the response appears to be complete or truncated"""
        if not response:
            return False
        
        # Check for proper ending punctuation
        if response.endswith(('.', '!', '?', ':', ';')):
            return True
        
        # Check for common completion patterns
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
        
        # Check if response seems to end mid-sentence
        last_sentence = response.split('.')[-1].strip()
        if len(last_sentence) > 20 and not any(punct in last_sentence for punct in ['!', '?', ':', ';']):
            return False
        
        return True
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context from ML models with validation"""
        try:
            # Get user profile
            user_profile = self.inference_engine.get_user_profile(user_id)
            
            # Get ML predictions
            risk_pred = self.inference_engine.predict_risk_tolerance(user_id)
            segment = self.inference_engine.get_user_segment(user_id)
            projection = self.inference_engine.predict_pension_projection(user_id)
            summary = self.inference_engine.get_summary_stats(user_id)
            
            raw_context = {
                "profile": user_profile,
                "risk_prediction": risk_pred,
                "segment": segment,
                "projection": projection,
                "summary": summary
            }
            
            # Validate and clean the ML data
            validated_context = self.validate_ml_data(raw_context)
            
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
       
    async def route_query(self, user_id: str, message: str) -> str:
        """Unified routing function that handles ALL queries through a single intelligent prompt"""
        try:
            context = await self.get_user_context(user_id)
            if "error" in context:
                return f"I couldn't retrieve your information: {context['error']}"
            
            return await self.handle_unified_query(user_id, message, context)
                    
        except Exception as e:
            logger.error(f"Error routing query for user {user_id}: {e}")
            return f"I'm sorry, I encountered an error processing your request: {str(e)}"

    async def handle_unified_query(self, user_id: str, message: str, context: Dict[str, Any]) -> str:
        """Handle all queries with a single intelligent prompt"""
        user_profile = context["profile"]
        summary = context["summary"]
        projection = context["projection"]
        risk_data = context["risk_prediction"]
        segment = context["segment"]
        
        # Get advanced ML analysis
        try:
            advanced_analysis = self.advanced_ml_pipeline.get_comprehensive_user_analysis(user_id)
        except Exception as e:
            logger.warning(f"Could not get advanced ML analysis: {e}")
            advanced_analysis = {}
        
        # Calculate key metrics for tables
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
        
        # Advanced ML metrics
        financial_health = self.safe_get(advanced_analysis, 'financial_health', {})
        churn_risk = self.safe_get(advanced_analysis, 'churn_risk', {})
        anomaly_detection = self.safe_get(advanced_analysis, 'anomaly_detection', {})
        fund_recommendations = self.safe_get(advanced_analysis, 'fund_recommendations', {})
        monte_carlo = self.safe_get(advanced_analysis, 'monte_carlo_simulation', {})
        peer_matching = self.safe_get(advanced_analysis, 'peer_matching', {})
        portfolio_optimization = self.safe_get(advanced_analysis, 'portfolio_optimization', {})
        
        unified_prompt = f"""
        {SHARED_PROMPT_INSTRUCTIONS}
        
        You are an expert superannuation advisor. The user will ask you questions ranging from simple greetings to complex financial advice. Handle each appropriately.
        
        For greetings (hi, hello, etc.): Respond warmly and briefly, then ask how you can help.
        For financial questions: Provide comprehensive, personalized advice using their data.
        
        IMPORTANT: When relevant to the user's question, include formatted tables using this exact format:
        
        | Header 1 | Header 2 | Header 3 |
        |----------|----------|----------|
        | Data 1   | Data 2   | Data 3   |
        | Data 4   | Data 5   | Data 6   |
        
        USER PROFILE:
        - Name: {self.safe_get(user_profile, 'Name', user_id)}
        - Age: {age}
        - Annual Income: ${annual_income:,.0f}
        - Current Savings: ${current_savings:,.0f}
        - Monthly Contribution: ${monthly_contrib:,.0f}
        - Risk Tolerance: {self.safe_get(user_profile, 'Risk_Tolerance', 'Unknown')}
        - Investment Type: {self.safe_get(user_profile, 'Investment_Type', 'Unknown')}
        - Insurance Coverage: {self.safe_get(user_profile, 'Insurance_Coverage', 'Not specified')}
        - Marital Status: {self.safe_get(user_profile, 'Marital_Status', 'Unknown')}
        - Education Level: {self.safe_get(user_profile, 'Education_Level', 'Unknown')}
        - Employment Status: {self.safe_get(user_profile, 'Employment_Status', 'Unknown')}
        - Projected Pension: ${projected_pension:,.0f}
        - Years to Retirement: {years_to_retirement}
        - Monthly Retirement Income: ${self.safe_get(projection, 'monthly_income_at_retirement', 0):,.0f}
        - Model Risk Prediction: {self.safe_get(risk_data, 'predicted_risk', 'Unknown')} (confidence: {self.safe_get(risk_data, 'confidence', 0):.1%})
        
        ADVANCED ML INSIGHTS:
        - Financial Health Score: {self.safe_get(financial_health, 'financial_health_score', 0):.0f}/100 (Above {self.safe_get(financial_health, 'peer_percentile', 0):.0f}% of peers)
        - Churn Risk: {self.safe_get(churn_risk, 'churn_probability', 0):.1%} ({self.safe_get(churn_risk, 'risk_level', 'Unknown')} risk)
        - Anomaly Score: {self.safe_get(anomaly_detection, 'anomaly_percentage', 0):.1f}% ({'Anomaly Detected' if self.safe_get(anomaly_detection, 'is_anomaly', False) else 'Normal Activity'})
        - Current Fund: {self.safe_get(fund_recommendations, 'current_fund', 'Unknown')}
        - Recommended Funds: {', '.join(self.safe_get(fund_recommendations, 'recommendations', [])[:3])}
        - Monte Carlo Simulations: {self.safe_get(monte_carlo, 'simulations', 0)} scenarios
        - Probability of Meeting Target: {self.safe_get(monte_carlo, 'probability_above_target', 0):.1%}
        - Similar Peers Found: {self.safe_get(peer_matching, 'total_peers_found', 0)}
        - Portfolio Sharpe Ratio: {self.safe_get(portfolio_optimization, 'portfolio_metrics', {}).get('sharpe_ratio', 0):.2f}
        
        PEER COMPARISON DATA:
        - Peer Group Size: {self.safe_get(peer_stats, 'total_peers', 0)}
        - Average Peer Savings: ${avg_savings:,.0f}
        - Average Peer Income: ${self.safe_get(peer_stats, 'avg_income', 0):,.0f}
        - Average Peer Contribution: ${avg_contrib:,.0f}
        - Contribution Percentile: {self.safe_get(peer_stats, 'contribution_percentile', 0):.1f}%
        
        SUGGESTED TABLES FOR DIFFERENT QUESTION TYPES:
        
        For RISK questions, include:
        | Metric | Current | Recommended |
        |--------|---------|-------------|
        | Risk Level | {self.safe_get(risk_data, 'current_risk', 'Unknown')} | {self.safe_get(risk_data, 'predicted_risk', 'Unknown')} |
        | Age-Based Risk | - | {"High" if age < 35 else "Medium" if age < 50 else "Low"} |
        | Model Confidence | - | {self.safe_get(risk_data, 'confidence', 0):.1%} |
        | Years to Retire | - | {years_to_retirement} |
        
        For CONTRIBUTION questions, include:
        | Contribution Type | Amount | % of Income |
        |-------------------|--------|-------------|
        | Current Monthly | ${monthly_contrib:,.0f} | {contribution_rate:.1f}% |
        | Current Annual | ${monthly_contrib * 12:,.0f} | {contribution_rate:.1f}% |
        | Recommended Min | ${annual_income * 0.095:,.0f} | 9.5% |
        | Optimal Target | ${annual_income * 0.12:,.0f} | 12.0% |
        
        For PEER COMPARISON questions, include:
        | Metric | You | Peer Average | Performance |
        |--------|-----|--------------|-------------|
        | Savings | ${current_savings:,.0f} | ${avg_savings:,.0f} | {current_savings/avg_savings*100 if avg_savings > 0 else 0:.0f}% |
        | Contributions | ${monthly_contrib * 12:,.0f} | ${avg_contrib:,.0f} | {(monthly_contrib * 12)/avg_contrib*100 if avg_contrib > 0 else 0:.0f}% |
        
        For FINANCIAL HEALTH questions, include:
        | Health Metric | Score | Status | Peer Percentile |
        |---------------|-------|--------|-----------------|
        | Financial Health | {self.safe_get(financial_health, 'financial_health_score', 0):.0f}/100 | {self.safe_get(financial_health, 'status', 'Unknown')} | {self.safe_get(financial_health, 'peer_percentile', 0):.0f}% |
        | Churn Risk | {self.safe_get(churn_risk, 'churn_probability', 0):.1%} | {self.safe_get(churn_risk, 'risk_level', 'Unknown')} | - |
        | Anomaly Score | {self.safe_get(anomaly_detection, 'anomaly_percentage', 0):.1f}% | {'Anomaly' if self.safe_get(anomaly_detection, 'is_anomaly', False) else 'Normal'} | - |
        
        For FUND RECOMMENDATION questions, include:
        | Fund Type | Current | Recommended | Reason |
        |-----------|---------|-------------|--------|
        | Current Fund | {self.safe_get(fund_recommendations, 'current_fund', 'Unknown')} | - | Your current choice |
        | Top Recommendation | - | {self.safe_get(fund_recommendations, 'recommendations', ['None'])[0] if self.safe_get(fund_recommendations, 'recommendations', []) else 'None'} | {self.safe_get(fund_recommendations, 'reasoning', 'Based on similar users')} |
        | Alternative 1 | - | {self.safe_get(fund_recommendations, 'recommendations', ['None'])[1] if len(self.safe_get(fund_recommendations, 'recommendations', [])) > 1 else 'None'} | Diversification |
        | Alternative 2 | - | {self.safe_get(fund_recommendations, 'recommendations', ['None'])[2] if len(self.safe_get(fund_recommendations, 'recommendations', [])) > 2 else 'None'} | Risk optimization |
        
        For MONTE CARLO questions, include:
        | Scenario | Retirement Balance | Probability |
        |----------|-------------------|-------------|
        | Conservative (10th percentile) | ${self.safe_get(monte_carlo, 'percentiles', {}).get('p10', 0):,.0f} | 10% |
        | Moderate (50th percentile) | ${self.safe_get(monte_carlo, 'percentiles', {}).get('p50', 0):,.0f} | 50% |
        | Optimistic (90th percentile) | ${self.safe_get(monte_carlo, 'percentiles', {}).get('p90', 0):,.0f} | 90% |
        | Target Achievement | - | {self.safe_get(monte_carlo, 'probability_above_target', 0):.1%} |
        
        USER'S QUESTION: "{message}"
        
        Based on the question, provide relevant advice and include appropriate tables when they add value. Always use the exact table format shown above with proper markdown syntax.
        """
        
        # Simple length adjustment
        base_length = 600 if len(message.split()) < 5 else 1200
        
        return await self.query_gemini(unified_prompt, base_length)