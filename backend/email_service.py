import json
import logging
import os
import smtplib
from datetime import datetime
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from jinja2 import Template

# Load environment variables from project root (optional)
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(env_path):
    try:
        load_dotenv(env_path, encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        # Skip loading if file is corrupted
        print(
            f"Warning: Could not load .env file at {env_path} - using system environment variables"
        )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self):
        # Load environment variables
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.admin_email = os.getenv("ADMIN_EMAIL", "")

        # News API configuration
        self.news_api_key = os.getenv("NEWS_API_KEY", "your_news_api_key_here")

        # Gemini API configuration
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
            self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    def get_financial_news(self, use_gemini: bool = False) -> Dict[str, List[Dict]]:
        """Fetch financial news from various sources"""
        if use_gemini and self.gemini_api_key:
            logger.info("Using Gemini for news content generation")
            return self._generate_news_with_gemini()
        elif self.news_api_key and self.news_api_key != "your_news_api_key_here":
            logger.info("Using NewsAPI for real-time news")
            return self._fetch_newsapi_news()
        else:
            logger.warning("No API keys configured, using mock data")
            return self._get_mock_news()

    def _fetch_newsapi_news(self) -> Dict[str, List[Dict]]:
        """Fetch real-time news from NewsAPI"""
        news_categories = {
            "stock_market": [],
            "real_estate": [],
            "policy_changes": [],
            "general_news": [],
        }

        try:
            # Stock Market News - Economic turmoil and market affecting events
            stock_response = requests.get(
                f"https://newsapi.org/v2/everything?q=(stock market OR economy OR market turmoil OR economic indicators OR recession OR inflation)&apiKey={self.news_api_key}&language=en&sortBy=publishedAt&pageSize=5"
            )
            if stock_response.status_code == 200:
                stock_data = stock_response.json()
                news_categories["stock_market"] = stock_data.get("articles", [])[:3]

            # Real Estate News - Bonds and property market changes
            real_estate_response = requests.get(
                f"https://newsapi.org/v2/everything?q=(real estate OR bonds OR property market OR mortgage rates OR bond yields)&apiKey={self.news_api_key}&language=en&sortBy=publishedAt&pageSize=5"
            )
            if real_estate_response.status_code == 200:
                real_estate_data = real_estate_response.json()
                news_categories["real_estate"] = real_estate_data.get("articles", [])[
                    :3
                ]

            # Policy Changes News - Tax rates, term life, policy changes
            policy_response = requests.get(
                f"https://newsapi.org/v2/everything?q=(tax rates OR term life OR policy changes OR GST OR superannuation OR financial regulation)&apiKey={self.news_api_key}&language=en&sortBy=publishedAt&pageSize=5"
            )
            if policy_response.status_code == 200:
                policy_data = policy_response.json()
                news_categories["policy_changes"] = policy_data.get("articles", [])[:3]

            # General Financial News - GST and policy changes
            general_response = requests.get(
                f"https://newsapi.org/v2/everything?q=(financial news OR economic updates OR GST OR government policy)&apiKey={self.news_api_key}&language=en&sortBy=publishedAt&pageSize=5"
            )
            if general_response.status_code == 200:
                general_data = general_response.json()
                news_categories["general_news"] = general_data.get("articles", [])[:3]

        except Exception as e:
            logger.error(f"Error fetching NewsAPI data: {e}")
            return self._get_mock_news()

        return news_categories

    def _generate_news_with_gemini(self) -> Dict[str, List[Dict]]:
        """Generate financial news content using Gemini AI"""
        try:
            prompt = """
            Generate current financial news content for a superannuation and investment newsletter. 
            Create 3 articles for each category with realistic, current-sounding titles and descriptions.
            
            Categories needed:
            1. Stock Market & Economic Updates (economic turmoil, market-affecting events)
            2. Real Estate & Bonds (bonds and property market changes)
            3. Policy Changes & Tax Updates (tax rates, term life, policy changes)
            4. General Financial News (GST, policy changes)
            
            IMPORTANT: Return ONLY valid JSON in this exact format (no additional text):
            {
                "stock_market": [
                    {"title": "Article Title", "description": "Article description", "url": "#", "publishedAt": "2024-01-01T00:00:00Z"}
                ],
                "real_estate": [
                    {"title": "Article Title", "description": "Article description", "url": "#", "publishedAt": "2024-01-01T00:00:00Z"}
                ],
                "policy_changes": [
                    {"title": "Article Title", "description": "Article description", "url": "#", "publishedAt": "2024-01-01T00:00:00Z"}
                ],
                "general_news": [
                    {"title": "Article Title", "description": "Article description", "url": "#", "publishedAt": "2024-01-01T00:00:00Z"}
                ]
            }
            
            Make the content relevant to Australian financial markets and superannuation.
            """

            response = self.gemini_model.generate_content(prompt)

            # Clean the response text to extract JSON
            response_text = response.text.strip()

            # Try to find JSON in the response
            if response_text.startswith("```json"):
                response_text = (
                    response_text.replace("```json", "").replace("```", "").strip()
                )
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()

            # Find JSON object boundaries
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx != -1 and end_idx > start_idx:
                json_text = response_text[start_idx:end_idx]
                news_data = json.loads(json_text)
            else:
                raise ValueError("No valid JSON found in response")

            # Add current timestamp to articles
            current_time = datetime.now().isoformat()
            for category in news_data.values():
                for article in category:
                    article["publishedAt"] = current_time

            return news_data

        except Exception as e:
            logger.error(f"Error generating news with Gemini: {e}")
            logger.error(
                f"Response text: {response.text if 'response' in locals() else 'No response'}"
            )
            return self._get_mock_news()

    def _get_mock_news(self) -> Dict[str, List[Dict]]:
        """Fallback mock news data"""
        return {
            "stock_market": [
                {
                    "title": "Market Analysis: Economic Indicators Show Mixed Signals",
                    "description": "Recent economic data suggests cautious optimism in global markets.",
                    "url": "#",
                    "publishedAt": datetime.now().isoformat(),
                }
            ],
            "real_estate": [
                {
                    "title": "Property Market Update: Bond Yields Impact Real Estate",
                    "description": "Rising bond yields are affecting property investment strategies.",
                    "url": "#",
                    "publishedAt": datetime.now().isoformat(),
                }
            ],
            "policy_changes": [
                {
                    "title": "Tax Policy Updates: New Rates Effective Next Quarter",
                    "description": "Government announces updated tax brackets and rates.",
                    "url": "#",
                    "publishedAt": datetime.now().isoformat(),
                }
            ],
            "general_news": [
                {
                    "title": "Financial Services Update: Regulatory Changes",
                    "description": "New regulations impact financial planning and superannuation.",
                    "url": "#",
                    "publishedAt": datetime.now().isoformat(),
                }
            ],
        }

    def create_email_template(
        self, news_data: Dict[str, List[Dict]], user_data: Dict = None
    ) -> str:
        """Create professional HTML email template"""

        template_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>MUFG Financial Update</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }
                .container {
                    background: white;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }
                .header {
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    color: white;
                    padding: 30px;
                    text-align: center;
                }
                .header h1 {
                    margin: 0;
                    font-size: 28px;
                    font-weight: 300;
                }
                .header p {
                    margin: 10px 0 0 0;
                    opacity: 0.9;
                    font-size: 16px;
                }
                .content {
                    padding: 30px;
                }
                .section {
                    margin-bottom: 30px;
                }
                .section h2 {
                    color: #1e3c72;
                    font-size: 22px;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #e9ecef;
                }
                .news-item {
                    background: #f8f9fa;
                    border-left: 4px solid #2a5298;
                    padding: 15px;
                    margin-bottom: 15px;
                    border-radius: 0 5px 5px 0;
                }
                .news-item h3 {
                    margin: 0 0 8px 0;
                    color: #1e3c72;
                    font-size: 16px;
                }
                .news-item p {
                    margin: 0 0 8px 0;
                    color: #666;
                    font-size: 14px;
                }
                .news-item a {
                    color: #2a5298;
                    text-decoration: none;
                    font-size: 14px;
                    font-weight: 500;
                }
                .news-item a:hover {
                    text-decoration: underline;
                }
                .highlight-box {
                    background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
                    border: 1px solid #bbdefb;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 20px 0;
                }
                .highlight-box h3 {
                    color: #1e3c72;
                    margin-top: 0;
                }
                .footer {
                    background: #f8f9fa;
                    padding: 20px 30px;
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                }
                .footer a {
                    color: #2a5298;
                    text-decoration: none;
                }
                .cta-button {
                    display: inline-block;
                    background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
                    color: white;
                    padding: 12px 24px;
                    text-decoration: none;
                    border-radius: 5px;
                    font-weight: 500;
                    margin: 20px 0;
                }
                .cta-button:hover {
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                }
                .date-stamp {
                    color: #666;
                    font-size: 12px;
                    text-align: right;
                    margin-bottom: 20px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏦 MUFG Financial Update</h1>
                    <p>Your comprehensive financial market and policy update</p>
                </div>
                
                <div class="content">
                    <div class="date-stamp">
                        {{ current_date }}
                    </div>
                    
                    <div class="highlight-box">
                        <h3>📊 Market Overview</h3>
                        <p>Stay informed with the latest developments in financial markets, real estate, and policy changes that may impact your superannuation and investment strategies.</p>
                    </div>
                    
                    <div class="section">
                        <h2>📈 Stock Market & Economic Updates</h2>
                        {% for article in stock_market_news %}
                        <div class="news-item">
                            <h3>{{ article.title }}</h3>
                            <p>{{ article.description }}</p>
                            <a href="{{ article.url }}" target="_blank">Read More →</a>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="section">
                        <h2>🏠 Real Estate & Bonds</h2>
                        {% for article in real_estate_news %}
                        <div class="news-item">
                            <h3>{{ article.title }}</h3>
                            <p>{{ article.description }}</p>
                            <a href="{{ article.url }}" target="_blank">Read More →</a>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="section">
                        <h2>📋 Policy Changes & Tax Updates</h2>
                        {% for article in policy_news %}
                        <div class="news-item">
                            <h3>{{ article.title }}</h3>
                            <p>{{ article.description }}</p>
                            <a href="{{ article.url }}" target="_blank">Read More →</a>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div class="section">
                        <h2>📰 General Financial News</h2>
                        {% for article in general_news %}
                        <div class="news-item">
                            <h3>{{ article.title }}</h3>
                            <p>{{ article.description }}</p>
                            <a href="{{ article.url }}" target="_blank">Read More →</a>
                        </div>
                        {% endfor %}
                    </div>
                    
                    <div style="text-align: center; margin: 30px 0;">
                            <a href="http://localhost:8080/" class="cta-button" style="color: white;">View Your Dashboard</a>
                    </div>
                </div>
                
                <div class="footer">
                    <p>This is an automated update from MUFG Financial Services Unofficial.</p>
                    <p>For personalized advice, please contact your financial advisor or visit our <a href="http://localhost:8080/">online portal</a>.</p>
                    <p>© 2025 MUFG Financial Services. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """

        template = Template(template_html)
        return template.render(
            current_date=datetime.now().strftime("%B %d, %Y at %I:%M %p"),
            stock_market_news=news_data.get("stock_market", []),
            real_estate_news=news_data.get("real_estate", []),
            policy_news=news_data.get("policy_changes", []),
            general_news=news_data.get("general_news", []),
        )

    def send_email(
        self, to_email: str, subject: str, html_content: str, is_admin: bool = False
    ) -> bool:
        """Send email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart("alternative")
            msg["From"] = self.smtp_user
            msg["To"] = to_email
            msg["Subject"] = subject

            # Add HTML content
            html_part = MIMEText(html_content, "html")
            msg.attach(html_part)

            # Connect to SMTP server
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.smtp_user, self.smtp_password)

            # Send email
            text = msg.as_string()
            server.sendmail(self.smtp_user, to_email, text)
            server.quit()

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Error sending email to {to_email}: {e}")
            return False

    def send_financial_update(
        self, user_email: str = None, is_admin: bool = False, use_gemini: bool = False
    ) -> bool:
        """Send comprehensive financial update email"""
        try:
            # Get news data
            news_data = self.get_financial_news(use_gemini=use_gemini)

            # Create email content
            html_content = self.create_email_template(news_data)

            # Determine recipient
            recipient = user_email if user_email else self.admin_email
            subject = "MUFG Financial Update - Market & Policy Insights"

            # Send email
            return self.send_email(recipient, subject, html_content, is_admin)

        except Exception as e:
            logger.error(f"Error sending financial update: {e}")
            return False

    def send_user_specific_update(self, user_email: str, user_data: Dict) -> bool:
        """Send personalized financial update to specific user"""
        try:
            # Get news data
            news_data = self.get_financial_news()

            # Create personalized email content
            html_content = self.create_email_template(news_data, user_data)

            subject = f"Personalized Financial Update - {user_data.get('name', 'Valued Client')}"

            # Send email
            return self.send_email(user_email, subject, html_content)

        except Exception as e:
            logger.error(f"Error sending user-specific update: {e}")
            return False


# Global email service instance
email_service = EmailService()
