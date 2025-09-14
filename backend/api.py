import asyncio
import os
from typing import Any, Dict, Optional

import uvicorn
from azure_speech import STTRequest, TTSRequest, VoiceListResponse, azure_speech_service
from chat_router import SuperannuationChatRouter
from dotenv import load_dotenv
from email_service import email_service
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from inference import SuperannuationInference
from pydantic import BaseModel
from scheduler import email_scheduler

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

# Initialize FastAPI app
app = FastAPI(
    title="Superannuation AI Advisor API",
    description="ML-powered superannuation advisor with risk prediction, user segmentation, and investment recommendations",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://localhost:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference engine and chat router
inference_engine = None
chat_router = None


@app.on_event("startup")
async def startup_event():
    global inference_engine, chat_router
    try:
        inference_engine = SuperannuationInference()
        chat_router = SuperannuationChatRouter()

        # Initialize and start email scheduler
        await email_scheduler.initialize()
        email_scheduler.start_scheduler()

        print(
            "API startup complete - ML models, chat router, and email scheduler loaded!"
        )
    except Exception as e:
        print(f"Error during startup: {e}")
        raise


# Pydantic models for request/response
class SimulationRequest(BaseModel):
    user_id: str
    extra_monthly: float = 0


class ChatRequest(BaseModel):
    user_id: str
    message: str


class UserSignupRequest(BaseModel):
    name: str
    age: int
    gender: str
    country: str
    employment_status: str
    annual_income: float
    current_savings: float
    retirement_age_goal: int
    risk_tolerance: str
    contribution_amount: float
    contribution_frequency: str
    employer_contribution: float
    years_contributed: int
    investment_type: str
    fund_name: str
    marital_status: str
    number_of_dependents: int
    education_level: str
    health_status: str
    home_ownership_status: str
    investment_experience_level: str
    financial_goals: str
    insurance_coverage: str
    pension_type: str
    withdrawal_strategy: str


class EmailRequest(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    send_to_all: bool = False
    use_gemini: bool = False


# API Endpoints
@app.get("/")
async def root():
    return {"message": "Superannuation AI Advisor API", "status": "running"}


@app.get("/user/{user_id}")
async def get_user_profile(user_id: str):
    """Get raw user profile from CSV"""
    try:
        user_profile = inference_engine.get_user_profile(user_id)
        return {"success": True, "data": user_profile}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving user: {str(e)}")


@app.get("/predict/{user_id}")
async def predict_pension_and_risk(user_id: str):
    """Returns pension projection + risk category"""
    try:
        # Get pension projection
        projection = inference_engine.predict_pension_projection(user_id)

        # Get risk prediction
        risk_pred = inference_engine.predict_risk_tolerance(user_id)

        return {
            "success": True,
            "data": {"pension_projection": projection, "risk_prediction": risk_pred},
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error making predictions: {str(e)}"
        )


@app.get("/summary/{user_id}")
async def get_summary_stats(user_id: str):
    """Returns summary card values"""
    try:
        summary = inference_engine.get_summary_stats(user_id)
        return {"success": True, "data": summary}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting summary: {str(e)}")


@app.get("/peer_stats/{user_id}")
async def get_peer_statistics(user_id: str):
    """Returns peer comparison data"""
    try:
        segment = inference_engine.get_user_segment(user_id)
        return {"success": True, "data": segment}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting peer stats: {str(e)}"
        )


@app.post("/simulate")
async def simulate_scenario(request: SimulationRequest):
    """Run new projection with adjusted contributions"""
    try:
        print(
            f"Simulating scenario for user {request.user_id} with extra monthly: {request.extra_monthly}"
        )
        projection = inference_engine.predict_pension_projection(
            request.user_id, request.extra_monthly
        )
        print(f"Projection successful: {projection}")
        return {"success": True, "data": projection}
    except ValueError as e:
        print(f"ValueError in simulate: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Exception in simulate: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error running simulation: {str(e)}"
        )


@app.get("/risk/{user_id}")
async def get_risk_prediction(user_id: str):
    """Get risk tolerance prediction"""
    try:
        risk_pred = inference_engine.predict_risk_tolerance(user_id)
        return {"success": True, "data": risk_pred}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting risk: {str(e)}")


@app.get("/segment/{user_id}")
async def get_user_segment(user_id: str):
    """Get user segmentation"""
    try:
        segment = inference_engine.get_user_segment(user_id)
        return {"success": True, "data": segment}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting segment: {str(e)}")


@app.get("/recommendations/{user_id}")
async def get_investment_recommendations(user_id: str):
    """Get investment recommendations"""
    try:
        projection = inference_engine.predict_pension_projection(user_id)
        risk_pred = inference_engine.predict_risk_tolerance(user_id)
        segment = inference_engine.get_user_segment(user_id)

        return {
            "success": True,
            "data": {"projection": projection, "risk": risk_pred, "segment": segment},
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting recommendations: {str(e)}"
        )


@app.get("/projection/{user_id}")
async def get_pension_projection(user_id: str):
    """Get pension projection"""
    try:
        projection = inference_engine.predict_pension_projection(user_id)
        return {"success": True, "data": projection}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting projection: {str(e)}"
        )


@app.post("/chat")
async def chat_with_ai(request: ChatRequest):
    """Chat with AI advisor"""
    try:
        response = await chat_router.route_query(request.user_id, request.message)
        return {"success": True, "data": response}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.post("/signup")
async def signup_user(request: UserSignupRequest):
    """Register a new user"""
    try:
        if not inference_engine:
            raise HTTPException(
                status_code=500, detail="Inference engine not initialized"
            )

        # Generate a new User_ID
        import pandas as pd

        existing_ids = inference_engine.df["User_ID"].tolist()
        user_numbers = [int(uid[1:]) for uid in existing_ids if uid.startswith("U")]
        next_number = max(user_numbers) + 1 if user_numbers else 1000
        new_user_id = f"U{next_number}"

        # Create new user data
        new_user_data = {
            "User_ID": new_user_id,
            "Name": request.name,
            "Age": request.age,
            "Gender": request.gender,
            "Country": request.country,
            "Employment_Status": request.employment_status,
            "Annual_Income": request.annual_income,
            "Current_Savings": request.current_savings,
            "Retirement_Age_Goal": request.retirement_age_goal,
            "Risk_Tolerance": request.risk_tolerance,
            "Contribution_Amount": request.contribution_amount,
            "Contribution_Frequency": request.contribution_frequency,
            "Employer_Contribution": request.employer_contribution,
            "Total_Annual_Contribution": request.contribution_amount
            + request.employer_contribution,
            "Years_Contributed": request.years_contributed,
            "Investment_Type": request.investment_type,
            "Fund_Name": request.fund_name,
            "Marital_Status": request.marital_status,
            "Number_of_Dependents": request.number_of_dependents,
            "Education_Level": request.education_level,
            "Health_Status": request.health_status,
            "Home_Ownership_Status": request.home_ownership_status,
            "Investment_Experience_Level": request.investment_experience_level,
            "Financial_Goals": request.financial_goals,
            "Insurance_Coverage": request.insurance_coverage,
            "Pension_Type": request.pension_type,
            "Withdrawal_Strategy": request.withdrawal_strategy,
            # Calculate derived fields
            "Annual_Return_Rate": 7.0,  # Default assumption
            "Volatility": 15.0,  # Default assumption
            "Fees_Percentage": 1.0,  # Default assumption
            "Projected_Pension_Amount": request.current_savings
            * (1.07 ** (request.retirement_age_goal - request.age)),
            "Expected_Annual_Payout": 0,
            "Inflation_Adjusted_Payout": 0,
            "Years_of_Payout": 0,
            "Survivor_Benefits": "Standard",
            "Tax_Benefits_Eligibility": True,
            "Government_Pension_Eligibility": request.annual_income < 50000,
            "Private_Pension_Eligibility": True,
            "Portfolio_Diversity_Score": 0.5,
            "Savings_Rate": (
                request.contribution_amount / request.annual_income
                if request.annual_income > 0
                else 0
            ),
            "Debt_Level": "Low",  # Default assumption
        }

        # Add to dataframe
        new_row = pd.DataFrame([new_user_data])
        inference_engine.df = pd.concat(
            [inference_engine.df, new_row], ignore_index=True
        )

        # Save updated data
        inference_engine.df.to_csv(inference_engine.csv_path, index=False)

        return {
            "user_id": new_user_id,
            "message": f"User {request.name} registered successfully",
            "user_data": new_user_data,
        }
    except Exception as e:
        print(f"Signup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users")
async def get_all_users():
    """Get all registered users"""
    try:
        if not inference_engine:
            raise HTTPException(
                status_code=500, detail="Inference engine not initialized"
            )

        # Use only columns that exist in the CSV
        available_columns = [
            "User_ID",
            "Age",
            "Risk_Tolerance",
            "Annual_Income",
            "Current_Savings",
        ]

        # Check which columns actually exist
        existing_columns = [
            col for col in available_columns if col in inference_engine.df.columns
        ]

        users = inference_engine.df[existing_columns].to_dict("records")

        # Add Name field using User_ID as fallback
        for user in users:
            user["Name"] = user["User_ID"]  # Use User_ID as name for now

        return {"users": users}
    except Exception as e:
        print(f"Get users error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/send-email")
async def send_email_update(request: EmailRequest):
    """Send financial update email manually"""
    try:
        if request.send_to_all:
            # Send to admin with general update
            success = email_service.send_financial_update(
                is_admin=True, use_gemini=request.use_gemini
            )
            return {"success": success, "message": "Email sent to admin"}

        elif request.user_id and request.email:
            # Send personalized email to specific user
            user_profile = inference_engine.get_user_profile(request.user_id)
            user_data = {
                "name": user_profile.get("Name", "Valued Client"),
                "user_id": request.user_id,
                "age": user_profile.get("Age", 0),
                "risk_tolerance": user_profile.get("Risk_Tolerance", "Unknown"),
                "annual_income": user_profile.get("Annual_Income", 0),
                "current_savings": user_profile.get("Current_Savings", 0),
            }
            success = email_service.send_user_specific_update(request.email, user_data)
            return {"success": success, "message": f"Email sent to {request.email}"}

        else:
            # Send general update to admin
            success = email_service.send_financial_update(
                is_admin=True, use_gemini=request.use_gemini
            )
            return {"success": success, "message": "Email sent to admin"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")


@app.get("/trigger-email")
async def trigger_email_now():
    """Trigger email sending immediately (for testing)"""
    try:
        success = email_scheduler.trigger_manual_email()
        return {"success": success, "message": "Email triggered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering email: {str(e)}")


@app.get("/trigger-email/{news_source}")
async def trigger_email_with_source(news_source: str):
    """Trigger email with specific news source (gemini or newsapi)"""
    try:
        use_gemini = news_source.lower() == "gemini"
        success = email_service.send_financial_update(
            is_admin=True, use_gemini=use_gemini
        )
        source_name = "Gemini AI" if use_gemini else "NewsAPI"
        return {
            "success": success,
            "message": f"Email triggered successfully using {source_name}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error triggering email: {str(e)}")


@app.post("/text-to-speech")
async def text_to_speech(request: TTSRequest):
    """Convert text to speech using Azure Speech Services"""
    try:
        result = azure_speech_service.create_audio_response(
            request.text, request.voice_name
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in text-to-speech: {str(e)}"
        )


@app.post("/speech-to-text")
async def speech_to_text(request: STTRequest):
    """Convert speech to text using Azure Speech Services"""
    try:
        # Convert hex string back to bytes
        audio_bytes = bytes.fromhex(request.audio_data)
        result = azure_speech_service.process_audio_input(audio_bytes, request.language)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error in speech-to-text: {str(e)}"
        )


@app.get("/voices")
async def get_available_voices():
    """Get list of available voices for TTS"""
    try:
        voices = azure_speech_service.get_available_voices()
        return VoiceListResponse(voices=voices, enabled=azure_speech_service.enabled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting voices: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": inference_engine is not None,
        "chat_router_loaded": chat_router is not None,
        "data_loaded": inference_engine.df is not None if inference_engine else False,
        "email_scheduler_running": (
            email_scheduler.scheduler.running if email_scheduler.scheduler else False
        ),
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
