import os
import json
import requests
from typing import Dict, Any, List, Optional
import asyncio
import aiohttp
import google.generativeai as genai
import logging
from inference import SuperannuationInference


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
2. Break the answer into clear sections with proper spacing:
   - Strengths (what the user is doing well)
   - Areas for Improvement (what could be optimized, explained with reasoning)
   - Recommendations / Next Steps (practical, numbered actions they can take)
3. Use plain text headings for sections (NO asterisks or markdown) and ALWAYS insert a blank line before AND after each heading.
4. For bullet points under sections, use "- " format and ensure each bullet point is on its own line.
5. For numbered lists, use "1. ", "2. ", etc. format with proper spacing.
6. If referencing ML predictions:
   - Explicitly state: "Our model predicts X with Y% confidence."
   - If prediction contradicts user profile, explain both perspectives and reconcile them.
7. Always end with an Encouragement Statement (e.g., "You're on the right track, and small changes now can make a big difference in your retirement outlook.").

Tone guidelines:
- Encourage, don't lecture.
- Soften negative findings with positive framing ("You're slightly below average, but you have great potential to catch up").
- Use simple language to explain financial concepts without jargon.
"""

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
                logger.info("✅ Gemini API configured successfully")
            except Exception as e:
                logger.error(f"❌ Error configuring Gemini API: {e}")
                self.model = None
        else:
            logger.warning("⚠️ GEMINI_API_KEY not set. LLM features will not work.")
            self.model = None
        
        # Response length configuration (in tokens) - Significantly increased to prevent truncation
        self.base_response_lengths = {
            "risk": 1200,       # Doubled for risk questions
            "contribution": 1500, # Doubled for contribution analysis
            "projection": 2000,  # Doubled for retirement planning
            "peer": 1400,       # Doubled for peer comparisons
            "general": 1600,    # Doubled default length
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
                # Clean and post-process the response
                cleaned_response = self.clean_response(response.text.strip())
                formatted_response = self.prettify_response(cleaned_response)
                
                # Check if response appears complete, if not, try with more tokens
                if not self.is_response_complete(formatted_response) and max_tokens < 3000:
                    logger.info(f"Response appears truncated, retrying with more tokens: {max_tokens} -> {max_tokens * 2}")
                    # Retry with 100% more tokens (doubled)
                    retry_config = genai.types.GenerationConfig(
                        max_output_tokens=int(max_tokens * 2),
                        temperature=0.7,
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
        """Parse user query to determine intent"""
        message_lower = message.lower().strip()
        
        # Simple greetings and casual responses
        greeting_patterns = [
            "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
            "how are you", "how's it going", "what's up", "sup", "yo"
        ]
        
        if any(greeting in message_lower for greeting in greeting_patterns):
            return "greeting"
        
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
        
        # General advice queries (map to general)
        if any(word in message_lower for word in ["advice", "recommend", "should", "what", "how"]):
            return "general"
        
        return "general"
    
    async def handle_greeting_query(self, user_id: str, message: str) -> str:
        """Handle simple greetings with concise, friendly responses"""
        try:
            context = await self.get_user_context(user_id)
            if "error" in context:
                return "Hello! I'm here to help with your superannuation questions."
            
            user_profile = context["profile"]
            name = self.safe_get(user_profile, 'Name', user_id)
            
            # Simple, friendly greeting with minimal context
            return f"Hello {name}! 👋 I'm your superannuation advisor. How can I help you today?"
            
        except Exception as e:
            logger.error(f"Error handling greeting for user {user_id}: {e}")
            return "Hello! I'm here to help with your superannuation questions."
    
    async def handle_risk_query(self, user_id: str, message: str) -> str:
        """Handle risk-related queries with improved structure and ML confidence handling"""
        context = await self.get_user_context(user_id)
        
        if "error" in context:
            return f"I couldn't retrieve your information: {context['error']}"
        
        risk_data = context["risk_prediction"]
        user_profile = context["profile"]
        summary = context["summary"]
        projection = context["projection"]
        
        # Calculate risk appropriateness based on age
        age = self.safe_get(user_profile, 'Age', 0)
        years_to_retirement = self.safe_get(projection, 'years_to_retirement', max(0, 65 - age))
        
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
        
        # Check for low confidence predictions
        confidence = self.safe_get(risk_data, 'confidence', 0.0)
        confidence_note = ""
        if confidence < 0.5:
            confidence_note = f" Note: Our model has lower confidence ({confidence:.1%}) in this prediction, so consider this as guidance rather than definitive advice."
        
        # Determine if there's a contradiction between current and predicted risk
        current_risk = self.safe_get(risk_data, 'current_risk', 'Unknown')
        predicted_risk = self.safe_get(risk_data, 'predicted_risk', 'Unknown')
        contradiction_note = ""
        if current_risk != predicted_risk and current_risk != 'Unknown' and predicted_risk != 'Unknown':
            contradiction_note = f" There's a difference between your current risk profile ({current_risk}) and what our model predicts ({predicted_risk}). This could indicate evolving preferences or changing circumstances."
        
        prompt = f"""
        {SHARED_PROMPT_INSTRUCTIONS}

        COMPREHENSIVE USER PROFILE:
        - Name: {self.safe_get(user_profile, 'Name', user_id)}
        - Age: {age} years old
        - Years to Retirement: {years_to_retirement} years
        - Current Risk Tolerance: {current_risk}
        - Predicted Risk Tolerance: {predicted_risk}
        - Model Confidence: {confidence:.1%}{confidence_note}
        - Investment Experience: {self.safe_get(user_profile, 'Investment_Experience_Level', 'Unknown')}
        - Current Investment Type: {self.safe_get(user_profile, 'Investment_Type', 'Unknown')}
        - Annual Income: ${self.safe_get(user_profile, 'Annual_Income', 0):,.0f}
        - Current Savings: ${self.safe_get(user_profile, 'Current_Savings', 0):,.0f}
        - Monthly Contribution: ${self.safe_get(user_profile, 'Contribution_Amount', 0):,.0f}
        - Marital Status: {self.safe_get(user_profile, 'Marital_Status', 'Unknown')}
        - Dependents: {self.safe_get(user_profile, 'Number_of_Dependents', 0)}
        - Home Ownership: {self.safe_get(user_profile, 'Home_Ownership_Status', 'Unknown')}

        RISK ANALYSIS:
        - Age-Appropriate Risk Level: {age_appropriate_risk}
        - Risk Assessment: {risk_advice}
        - Current vs Predicted Risk Match: {'Yes' if current_risk == predicted_risk else 'No'}{contradiction_note}

        USER'S SPECIFIC QUESTION: "{message}"

        Focus on risk management, portfolio allocation, and how their risk tolerance aligns with their retirement timeline and goals.
        """
        
        # Adjust response length based on query
        base_length = self.base_response_lengths["risk"]
        adjusted_length = self.adjust_response_length(message, base_length, "risk")
        
        return await self.query_gemini(prompt, adjusted_length)
    
    async def handle_contribution_query(self, user_id: str, message: str) -> str:
        """Handle contribution/saving queries with improved structure and real-world context"""
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
            try:
                sim_projection = self.inference_engine.predict_pension_projection(user_id, extra_amount)
                improvement = self.safe_get(sim_projection, 'improvement', 0.0)
            except Exception as e:
                logger.warning(f"Simulation failed for user {user_id}: {e}")
                improvement = 0
        else:
            improvement = 0
        
        # Calculate contribution analysis
        monthly_contrib = self.safe_get(user_profile, 'Contribution_Amount', 0)
        current_annual_contrib = monthly_contrib * 12
        annual_income = self.safe_get(user_profile, 'Annual_Income', 0)
        contribution_rate = (current_annual_contrib / annual_income * 100) if annual_income > 0 else 0
        
        # Determine if contributions are adequate with softer language
        if contribution_rate >= 12:
            contrib_assessment = "Excellent - You're contributing above the recommended 9.5%"
        elif contribution_rate >= 9.5:
            contrib_assessment = "Good - You're meeting the minimum requirement"
        else:
            contrib_assessment = "Below recommended - Consider increasing contributions"
        
        # Calculate lifestyle coverage
        monthly_income_at_retirement = self.safe_get(projection, 'monthly_income_at_retirement', 0)
        current_monthly_income = annual_income / 12 if annual_income > 0 else 0
        lifestyle_coverage = (monthly_income_at_retirement / current_monthly_income * 100) if current_monthly_income > 0 else 0
        
        prompt = f"""
        {SHARED_PROMPT_INSTRUCTIONS}

        COMPREHENSIVE USER PROFILE:
        - Name: {self.safe_get(user_profile, 'Name', user_id)}
        - Age: {self.safe_get(user_profile, 'Age', 0)} years old
        - Annual Income: ${annual_income:,.0f}
        - Current Monthly Contribution: ${monthly_contrib:,.0f}
        - Current Annual Contribution: ${current_annual_contrib:,.0f}
        - Employer Contribution: ${self.safe_get(user_profile, 'Employer_Contribution', 0):,.0f}
        - Total Annual Contribution: ${self.safe_get(user_profile, 'Total_Annual_Contribution', 0):,.0f}
        - Current Savings: ${self.safe_get(user_profile, 'Current_Savings', 0):,.0f}
        - Years to Retirement: {self.safe_get(projection, 'years_to_retirement', 0)}
        - Risk Tolerance: {self.safe_get(user_profile, 'Risk_Tolerance', 'Unknown')}

        CONTRIBUTION ANALYSIS:
        - Current Contribution Rate: {contribution_rate:.1f}% of income
        - Assessment: {contrib_assessment}
        - Recommended Rate: 9.5% minimum, 12-15% optimal
        - Current Projected Pension: ${self.safe_get(projection, 'adjusted_projection', 0):,.0f}
        - Monthly Income at Retirement: ${monthly_income_at_retirement:,.0f}
        - Lifestyle Coverage: {lifestyle_coverage:.1f}% of current income

        SIMULATION RESULTS (if applicable):
        - Proposed Additional Monthly: ${extra_amount:,.0f}
        - Additional Annual Contribution: ${extra_amount * 12:,.0f}
        - Pension Improvement: ${improvement:,.0f}
        - New Projected Pension: ${self.safe_get(projection, 'adjusted_projection', 0) + improvement:,.0f}

        USER'S SPECIFIC QUESTION: "{message}"

        Focus on contribution strategies, tax benefits, employer matching, and how changes will impact their retirement lifestyle. Tie numbers back to real-world goals and affordability.
        """
        
        # Adjust response length based on query
        base_length = self.base_response_lengths["contribution"]
        adjusted_length = self.adjust_response_length(message, base_length, "contribution")
        
        return await self.query_gemini(prompt, adjusted_length)
    
    async def handle_projection_query(self, user_id: str, message: str) -> str:
        """Handle retirement projection queries with inflation and lifestyle considerations"""
        context = await self.get_user_context(user_id)
        
        if "error" in context:
            return f"I couldn't retrieve your information: {context['error']}"
        
        projection = context["projection"]
        summary = context["summary"]
        user_profile = context["profile"]
        
        # Calculate lifestyle analysis
        monthly_income = self.safe_get(projection, 'monthly_income_at_retirement', 0)
        annual_income = self.safe_get(user_profile, 'Annual_Income', 0)
        income_replacement_ratio = (monthly_income * 12) / annual_income if annual_income > 0 else 0
        
        # Determine lifestyle adequacy with softer language
        if income_replacement_ratio >= 0.7:
            lifestyle_assessment = "Excellent - You'll maintain a comfortable lifestyle"
        elif income_replacement_ratio >= 0.5:
            lifestyle_assessment = "Good - You'll have a decent retirement income"
        else:
            lifestyle_assessment = "Below recommended - Consider increasing savings"
        
        # Calculate years of coverage
        projected_pension = self.safe_get(projection, 'adjusted_projection', 0)
        years_of_coverage = projected_pension / (monthly_income * 12) if monthly_income > 0 else 0
        
        # Calculate inflation-adjusted projections (assuming 2.5% annual inflation)
        years_to_retirement = self.safe_get(projection, 'years_to_retirement', 0)
        inflation_factor = (1.025 ** years_to_retirement) if years_to_retirement > 0 else 1
        inflation_adjusted_monthly = monthly_income / inflation_factor
        
        prompt = f"""
        {SHARED_PROMPT_INSTRUCTIONS}

        COMPREHENSIVE USER PROFILE:
        - Name: {self.safe_get(user_profile, 'Name', user_id)}
        - Age: {self.safe_get(user_profile, 'Age', 0)} years old
        - Retirement Age Goal: {self.safe_get(user_profile, 'Retirement_Age_Goal', 65)} years
        - Years to Retirement: {years_to_retirement} years
        - Current Annual Income: ${annual_income:,.0f}
        - Current Savings: ${self.safe_get(summary, 'current_savings', 0):,.0f}
        - Monthly Contribution: ${self.safe_get(user_profile, 'Contribution_Amount', 0):,.0f}
        - Risk Tolerance: {self.safe_get(user_profile, 'Risk_Tolerance', 'Unknown')}
        - Investment Type: {self.safe_get(user_profile, 'Investment_Type', 'Unknown')}

        RETIREMENT PROJECTION ANALYSIS:
        - Projected Pension Amount: ${projected_pension:,.0f}
        - Monthly Income at Retirement: ${monthly_income:,.0f}
        - Annual Retirement Income: ${monthly_income * 12:,.0f}
        - Income Replacement Ratio: {income_replacement_ratio:.1%}
        - Lifestyle Assessment: {lifestyle_assessment}
        - Years of Coverage: {years_of_coverage:.1f} years
        - Progress to Goal: {self.safe_get(summary, 'percent_to_goal', 0):.1f}%

        INFLATION CONSIDERATIONS:
        - Inflation-Adjusted Monthly Income: ${inflation_adjusted_monthly:,.0f}
        - Inflation Factor (2.5% annually): {inflation_factor:.2f}x
        - Note: This shows purchasing power in today's dollars

        RETIREMENT READINESS INDICATORS:
        - ASFA Comfortable Standard: $62,000/year for couples, $44,000/year for singles
        - ASFA Modest Standard: $43,000/year for couples, $31,000/year for singles
        - Recommended Replacement Ratio: 70-80% of pre-retirement income

        USER'S SPECIFIC QUESTION: "{message}"

        Focus on retirement readiness, lifestyle adequacy, inflation impact, early retirement scenarios, and strategies to improve their retirement outlook.
        """
        
        # Adjust response length based on query
        base_length = self.base_response_lengths["projection"]
        adjusted_length = self.adjust_response_length(message, base_length, "projection")
        
        return await self.query_gemini(prompt, adjusted_length)
    
    async def handle_peer_query(self, user_id: str, message: str) -> str:
        """Handle peer comparison queries with constructive tone and actionable insights"""
        context = await self.get_user_context(user_id)
        
        if "error" in context:
            return f"I couldn't retrieve your information: {context['error']}"
        
        segment = context["segment"]
        peer_stats = self.safe_get(segment, "peer_stats", {})
        user_profile = context["profile"]
        summary = context["summary"]
        
        # Calculate performance metrics
        user_savings = self.safe_get(user_profile, 'Current_Savings', 0)
        user_income = self.safe_get(user_profile, 'Annual_Income', 0)
        user_contrib = self.safe_get(user_profile, 'Contribution_Amount', 0) * 12
        
        avg_savings = self.safe_get(peer_stats, 'avg_savings', 0)
        avg_income = self.safe_get(peer_stats, 'avg_income', 0)
        avg_contrib = self.safe_get(peer_stats, 'avg_contribution', 0)
        
        # Calculate relative performance
        savings_vs_peers = (user_savings / avg_savings * 100) if avg_savings > 0 else 0
        contrib_vs_peers = (user_contrib / avg_contrib * 100) if avg_contrib > 0 else 0
        
        # Determine performance level with constructive language
        if savings_vs_peers >= 120:
            performance_level = "Above Average"
            performance_note = "You're performing well above your peer group - keep up the excellent work!"
        elif savings_vs_peers >= 80:
            performance_level = "Average"
            performance_note = "You're right on track with your peer group - steady progress is key."
        else:
            performance_level = "Below Average"
            performance_note = "You're currently behind peers, but your high income gives you strong catch-up potential."
        
        # Calculate catch-up potential
        income_advantage = user_income - avg_income
        catch_up_potential = ""
        if income_advantage > 0:
            catch_up_potential = f"Your income is ${income_advantage:,.0f} above average, giving you excellent potential to catch up quickly."
        elif income_advantage < 0:
            catch_up_potential = f"While your income is ${abs(income_advantage):,.0f} below average, strategic contributions can still help you reach your goals."
        
        prompt = f"""
        {SHARED_PROMPT_INSTRUCTIONS}

        COMPREHENSIVE USER PROFILE:
        - Name: {self.safe_get(user_profile, 'Name', user_id)}
        - Age: {self.safe_get(user_profile, 'Age', 0)} years old
        - Annual Income: ${user_income:,.0f}
        - Current Savings: ${user_savings:,.0f}
        - Annual Contribution: ${user_contrib:,.0f}
        - Investment Type: {self.safe_get(user_profile, 'Investment_Type', 'Unknown')}
        - Risk Tolerance: {self.safe_get(user_profile, 'Risk_Tolerance', 'Unknown')}
        - Marital Status: {self.safe_get(user_profile, 'Marital_Status', 'Unknown')}
        - Dependents: {self.safe_get(user_profile, 'Number_of_Dependents', 0)}

        PEER GROUP ANALYSIS:
        - Total Peers in Group: {self.safe_get(peer_stats, 'total_peers', 0)}
        - Average Age: {self.safe_get(peer_stats, 'avg_age', 0):.1f} years
        - Average Income: ${avg_income:,.0f}
        - Average Savings: ${avg_savings:,.0f}
        - Average Annual Contribution: ${avg_contrib:,.0f}
        - Common Investment Types: {self.safe_get(peer_stats, 'common_investment_types', 'Unknown')}
        - Risk Distribution: {self.safe_get(peer_stats, 'risk_distribution', 'Unknown')}

        PERFORMANCE METRICS:
        - Your Savings vs Peers: {savings_vs_peers:.1f}% of peer average
        - Your Contributions vs Peers: {contrib_vs_peers:.1f}% of peer average
        - Overall Performance Level: {performance_level}
        - Performance Note: {performance_note}
        - Contribution Percentile: {self.safe_get(peer_stats, 'contribution_percentile', 0):.1f}%
        - Catch-up Potential: {catch_up_potential}

        COMPARATIVE INSIGHTS:
        - Income Comparison: {'Above' if user_income > avg_income else 'Below'} peer average by ${abs(user_income - avg_income):,.0f}
        - Savings Comparison: {'Above' if user_savings > avg_savings else 'Below'} peer average by ${abs(user_savings - avg_savings):,.0f}
        - Contribution Comparison: {'Above' if user_contrib > avg_contrib else 'Below'} peer average by ${abs(user_contrib - avg_contrib):,.0f}

        USER'S SPECIFIC QUESTION: "{message}"

        Focus on constructive peer insights, actionable strategies based on what's working for similar people, and specific steps to improve their position.
        """
        
        # Adjust response length based on query
        base_length = self.base_response_lengths["peer"]
        adjusted_length = self.adjust_response_length(message, base_length, "peer")
        
        return await self.query_gemini(prompt, adjusted_length)
    
    async def handle_general_query(self, user_id: str, message: str) -> str:
        """Handle general queries with comprehensive personalized advice"""
        context = await self.get_user_context(user_id)
        
        if "error" in context:
            return f"I couldn't retrieve your information: {context['error']}"
        
        summary = context["summary"]
        user_profile = context["profile"]
        projection = context["projection"]
        
        # Calculate key insights
        age = self.safe_get(user_profile, 'Age', 0)
        years_to_retirement = self.safe_get(projection, 'years_to_retirement', max(0, 65 - age))
        current_savings = self.safe_get(summary, 'current_savings', 0)
        projected_pension = self.safe_get(projection, 'adjusted_projection', 0)
        monthly_contrib = self.safe_get(user_profile, 'Contribution_Amount', 0)
        
        # Determine overall financial health with softer language
        if projected_pension >= 500000 and years_to_retirement > 10:
            overall_assessment = "Excellent - You're well on track for a comfortable retirement"
        elif projected_pension >= 300000:
            overall_assessment = "Good - You're making solid progress toward retirement"
        else:
            overall_assessment = "Needs attention - Consider increasing contributions or adjusting strategy"
        
        # Calculate key ratios for context
        annual_income = self.safe_get(user_profile, 'Annual_Income', 0)
        contribution_rate = (monthly_contrib * 12) / annual_income * 100 if annual_income > 0 else 0
        monthly_retirement_income = self.safe_get(projection, 'monthly_income_at_retirement', 0)
        income_replacement_ratio = (monthly_retirement_income * 12) / annual_income if annual_income > 0 else 0
        
        prompt = f"""
        {SHARED_PROMPT_INSTRUCTIONS}

        COMPREHENSIVE USER PROFILE:
        - Name: {self.safe_get(user_profile, 'Name', user_id)}
        - Age: {age} years old
        - Years to Retirement: {years_to_retirement} years
        - Annual Income: ${annual_income:,.0f}
        - Current Savings: ${current_savings:,.0f}
        - Monthly Contribution: ${monthly_contrib:,.0f}
        - Projected Pension: ${projected_pension:,.0f}
        - Risk Tolerance: {self.safe_get(user_profile, 'Risk_Tolerance', 'Unknown')}
        - Investment Type: {self.safe_get(user_profile, 'Investment_Type', 'Unknown')}
        - Marital Status: {self.safe_get(user_profile, 'Marital_Status', 'Unknown')}
        - Dependents: {self.safe_get(user_profile, 'Number_of_Dependents', 0)}
        - Home Ownership: {self.safe_get(user_profile, 'Home_Ownership_Status', 'Unknown')}

        FINANCIAL HEALTH OVERVIEW:
        - Overall Assessment: {overall_assessment}
        - Progress to Goal: {self.safe_get(summary, 'percent_to_goal', 0):.1f}%
        - Monthly Income at Retirement: ${monthly_retirement_income:,.0f}
        - Investment Experience: {self.safe_get(user_profile, 'Investment_Experience_Level', 'Unknown')}
        - Contribution Rate: {contribution_rate:.1f}% of income
        - Income Replacement Ratio: {income_replacement_ratio:.1%}

        USER'S SPECIFIC QUESTION: "{message}"

        Provide comprehensive, personalized advice that addresses their specific question while considering their complete financial profile, age, family situation, and retirement goals.
        """
        
        # Adjust response length based on query
        base_length = self.base_response_lengths["general"]
        adjusted_length = self.adjust_response_length(message, base_length, "general")
        
        return await self.query_gemini(prompt, adjusted_length)
    
    async def route_query(self, user_id: str, message: str) -> str:
        """Main routing function for chat queries with improved error handling"""
        try:
            intent = self.parse_query_intent(message)
            
            if intent == "greeting":
                response = await self.handle_greeting_query(user_id, message)
            elif intent == "risk":
                response = await self.handle_risk_query(user_id, message)
            elif intent == "contribution":
                response = await self.handle_contribution_query(user_id, message)
            elif intent == "projection":
                response = await self.handle_projection_query(user_id, message)
            elif intent == "peer":
                response = await self.handle_peer_query(user_id, message)
            else:
                response = await self.handle_general_query(user_id, message)
            
            # Check if the response indicates API failure (skip for greetings)
            if intent != "greeting" and ("unable to connect" in response.lower() or "trouble connecting" in response.lower()):
                logger.warning(f"Gemini API failure for user {user_id}, using fallback")
                # Provide a fallback response based on the intent
                return await self.get_fallback_response(user_id, message, intent)
            
            return response
                
        except Exception as e:
            logger.error(f"Error routing query for user {user_id}: {e}")
            return f"I'm sorry, I encountered an error processing your request: {str(e)}"
    
    async def get_fallback_response(self, user_id: str, message: str, intent: str) -> str:
        """Provide enhanced fallback responses with ML stats when AI service is unavailable"""
        try:
            context = await self.get_user_context(user_id)
            if "error" in context:
                return f"I couldn't retrieve your information: {context['error']}"
            
            user_profile = context["profile"]
            summary = context["summary"]
            projection = context["projection"]
            
            # Extract basic stats using safe_get
            age = self.safe_get(user_profile, 'Age', 0)
            current_savings = self.safe_get(summary, 'current_savings', 0)
            monthly_contrib = self.safe_get(user_profile, 'Contribution_Amount', 0)
            annual_income = self.safe_get(user_profile, 'Annual_Income', 0)
            projected_pension = self.safe_get(projection, 'adjusted_projection', 0)
            
            if intent == "greeting":
                name = self.safe_get(user_profile, 'Name', user_id)
                return f"Hello {name}! 👋 I'm your superannuation advisor. How can I help you today?"
            
            elif intent == "risk":
                risk_level = self.safe_get(user_profile, 'Risk_Tolerance', 'Unknown')
                return f"**Risk Profile Summary**\n\nBased on your profile, you have a {risk_level} risk tolerance. This means you prefer {'conservative' if risk_level == 'Low' else 'moderate' if risk_level == 'Medium' else 'aggressive'} investment strategies. Your current investment type is {self.safe_get(user_profile, 'Investment_Type', 'Unknown')}.\n\n**Encouragement Statement**\nYour risk profile aligns well with your age and retirement timeline. While I'm experiencing technical difficulties with my AI service, I can still provide basic information about your account."
            
            elif intent == "contribution":
                contribution_rate = (monthly_contrib * 12) / annual_income * 100 if annual_income > 0 else 0
                return f"**Contribution Analysis**\n\nYou're currently contributing ${monthly_contrib:,.0f} per month (${monthly_contrib * 12:,.0f} annually), which is {contribution_rate:.1f}% of your annual income. The recommended contribution rate is typically 9.5% of your salary for superannuation.\n\n**Encouragement Statement**\nEvery contribution counts toward your retirement goals. Consider increasing your contributions gradually to improve your retirement outlook."
            
            elif intent == "projection":
                years_to_retirement = self.safe_get(projection, 'years_to_retirement', max(0, 65 - age))
                return f"**Retirement Projection**\n\nWith your current savings of ${current_savings:,.0f}, you're projected to have approximately ${projected_pension:,.0f} at retirement in {years_to_retirement} years. This projection assumes continued contributions and market growth.\n\n**Encouragement Statement**\nYou're making progress toward your retirement goals. Small increases in contributions can significantly improve your retirement outlook."
            
            elif intent == "peer":
                return f"**Peer Comparison**\n\nYou're {age} years old with an annual income of ${annual_income:,.0f}. People in your age group typically have similar financial goals and risk profiles. Your investment strategy should align with your age and income level.\n\n**Encouragement Statement**\nFocus on your own financial journey and goals rather than comparing to others. Every step forward is progress."
            
            else:
                return f"**Financial Overview**\n\nHello! I can see you're asking about your superannuation. You currently have ${current_savings:,.0f} in savings with a {self.safe_get(summary, 'risk_tolerance', 'Unknown')} risk profile. Your projected pension is ${projected_pension:,.0f}.\n\n**Encouragement Statement**\nWhile I'm experiencing some technical difficulties with my AI service, I can still provide basic information about your account. What specific aspect would you like to know more about?"
                
        except Exception as e:
            logger.error(f"Fallback response error for user {user_id}: {e}")
            return f"I'm having trouble accessing your information right now. Please try again later or contact support if the issue persists."
