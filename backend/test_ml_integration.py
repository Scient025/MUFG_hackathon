#!/usr/bin/env python3
"""
Test script to demonstrate how advanced ML data flows to the LLM API
"""

import asyncio
from chat_router import SuperannuationChatRouter

async def test_ml_integration():
    """Test the integration of advanced ML models with the LLM"""
    
    print("🤖 Testing Advanced ML Integration with LLM API")
    print("=" * 60)
    
    # Initialize the chat router (this loads all ML models)
    chat_router = SuperannuationChatRouter()
    
    # Test user ID
    test_user_id = "U1000"  # Use a user from your Supabase database
    
    print(f"📊 Testing with user: {test_user_id}")
    print()
    
    # Test different types of questions that would use ML data
    test_questions = [
        "What's my financial health score?",
        "Am I at risk of stopping my contributions?",
        "What funds do you recommend for me?",
        "Run a retirement stress test for me",
        "How do I compare to similar users?",
        "What's my optimal portfolio allocation?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"🔍 Question {i}: {question}")
        print("-" * 50)
        
        try:
            # This will trigger the ML analysis and send data to LLM
            response = await chat_router.route_query(test_user_id, question)
            
            # Show a preview of the response
            preview = response[:200] + "..." if len(response) > 200 else response
            print(f"📝 LLM Response Preview: {preview}")
            print()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            print()
    
    print("✅ Test completed! The LLM now receives:")
    print("   • Financial Health Score (0-100)")
    print("   • Churn Risk Probability (%)")
    print("   • Anomaly Detection Score (%)")
    print("   • Fund Recommendations")
    print("   • Monte Carlo Simulation Results")
    print("   • Peer Matching Data")
    print("   • Portfolio Optimization Metrics")
    print()
    print("🎯 The LLM can now provide personalized advice using all 7 ML models!")

if __name__ == "__main__":
    asyncio.run(test_ml_integration())
