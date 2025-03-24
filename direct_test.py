#!/usr/bin/env python3
"""
Direct test to compare database values with NCUA chatbot's interpretation.
This is a focused test to debug the Navy Federal asset value issue.
"""

import asyncio
import json
import logging
from typing import Dict, Any
import openai
from dotenv import load_dotenv

from ncua_chatbot import NCUAChatbot
from direct_query import query_total_assets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_direct_interpretation():
    """Test the direct interpretation of asset data"""
    print("\n=== Testing Direct Interpretation of Asset Data ===")
    
    # Initialize the chatbot
    chatbot = NCUAChatbot()
    
    # Navy Federal's cu_number
    cu_number = "5536"
    
    # Get the asset data directly from the database
    assets = query_total_assets(cu_number)
    if not assets:
        print("Failed to get asset data, cannot proceed with test")
        return
    
    # Create a minimal financial data structure with just the asset info
    mock_financial_data = {
        "credit_union_info": {
            "cu_number": cu_number,
            "cu_name": "NAVY FEDERAL CREDIT UNION"
        },
        "results": {
            "fs220_2024_12": [
                {
                    "id": 641,
                    "cu_number": int(cu_number),
                    "cycle_date": "12/31/2024 0:00:00",
                    "acct_010": assets
                }
            ]
        }
    }
    
    # Extract metrics from this simplified dataset
    print("\n1. Testing metric extraction:")
    extracted_data, source_citations = chatbot.extract_key_financial_metrics(mock_financial_data, "What is Navy Federal Credit Union's total assets?")
    print("Extracted data that will be sent to OpenAI:")
    print(extracted_data)
    
    if "Total Assets" not in extracted_data:
        print("\nERROR: Total Assets not found in extracted data!")
        print("This explains why OpenAI isn't getting the correct data.")
    
    # Test OpenAI interpretation directly
    print("\n2. Testing direct OpenAI interpretation:")
    # Set up minimal client with the OpenAI API key
    openai.api_key = chatbot.openai_api_key
    
    # Create the system message
    system_message = f"""
You are a financial assistant providing information about Navy Federal Credit Union.

IMPORTANT INSTRUCTIONS:
1. Be extremely concise - answer in 1-2 sentences maximum
2. Do NOT explain which tables or fields you used to find the information
3. Do NOT explain your reasoning or process
4. Simply state the answer directly and nothing more
5. Include dollar signs and commas for financial amounts
6. Focus ONLY on answering the specific question asked
7. Use ONLY the financial metrics provided in the user message
8. DO NOT include any source citations in your answer
9. DO NOT use your own knowledge about Navy Federal - use ONLY the provided data

Example format:
Question: "What is the asset size of Navy Federal?"
Good answer: "Navy Federal Credit Union has total assets of $180,813,031,049."
Bad answer: "According to my knowledge, Navy Federal Credit Union has assets of $175,893,456,789."
"""

    # Create the user message with extracted data
    user_message = f"""
Question: What is Navy Federal Credit Union's total assets?

Here are the relevant financial metrics for Navy Federal Credit Union:

{extracted_data}
"""
    
    # Call OpenAI directly
    response = openai.chat.completions.create(
        model=chatbot.chat_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0,
        max_tokens=100
    )
    
    answer = response.choices[0].message.content
    print(f"OpenAI response: {answer}")
    
    # Now test using the chatbot's interpret_results method
    print("\n3. Testing chatbot's interpret_results method:")
    chatbot_answer = chatbot.interpret_results(
        "What is Navy Federal Credit Union's total assets?",
        {"cu_name": "NAVY FEDERAL CREDIT UNION", "cu_number": cu_number},
        mock_financial_data
    )
    print(f"Chatbot interpretation: {chatbot_answer}")
    
    # Compare results
    print("\n4. Comparison:")
    print(f"Actual database value: ${assets:,.2f}")
    print(f"Direct OpenAI response: {answer}")
    print(f"Chatbot interpretation: {chatbot_answer}")

if __name__ == "__main__":
    asyncio.run(test_direct_interpretation()) 