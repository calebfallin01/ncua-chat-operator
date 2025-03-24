#!/usr/bin/env python3
"""
Test script for the dynamic SQL query integration with NCUA chatbot.

This script tests:
1. Account description mapping functionality
2. Dynamic SQL query functionality
3. Integration with NCUA chatbot process flow
"""

import os
import logging
import json
import asyncio
from ncua_chatbot import NCUAChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_chatbot_query(chatbot, question, cu_name=None):
    """
    Test the chatbot's query process flow with a specific question.
    
    Args:
        chatbot: Initialized NCUAChatbot instance
        question: Question to ask
        cu_name: Optional credit union name to prepend to question
        
    Returns:
        The chatbot's answer
    """
    print(f"\n--- Testing Question ---")
    if cu_name:
        full_question = f"What is {cu_name}'s {question}?"
    else:
        full_question = question
        
    print(f"Question: {full_question}")
    
    # Process the query using the chatbot's query handler
    answer = await chatbot.process_query(full_question)
    
    print(f"Answer: {answer}")
    return answer

async def test_credit_union_extraction(chatbot, cu_name):
    """Test the credit union extraction functionality."""
    print(f"\n--- Testing Credit Union Extraction ---")
    print(f"Credit Union: {cu_name}")
    
    # Extract credit union info
    cu_info = chatbot.extract_credit_union_from_query(f"Tell me about {cu_name}")
    
    if cu_info:
        print(f"Found Credit Union: {cu_info.get('cu_name', 'Unknown')}")
        print(f"CU Number: {cu_info.get('cu_number', 'Unknown')}")
        return cu_info
    else:
        print("Failed to extract credit union info")
        return None

async def test_account_description_mapping(chatbot, financial_terms):
    """Test the account description mapping functionality."""
    print(f"\n--- Testing Account Description Mapping ---")
    print(f"Financial Terms: {', '.join(financial_terms)}")
    
    # Get account mappings
    account_mappings = chatbot.enhanced_query_account_descriptions(financial_terms)
    
    print(f"Found {len(account_mappings)} account mappings")
    
    # Print a sample of the mappings
    if account_mappings:
        print("\nSample Account Mappings:")
        for i, (acct_code, mapping) in enumerate(account_mappings.items()):
            print(f"  {acct_code}: {mapping.get('name', 'Unknown')} ({mapping.get('tablename', 'Unknown')})")
            if i >= 4:  # Show at most 5 mappings
                print("  ...")
                break
    
    return account_mappings

async def run_tests():
    """Run all the tests."""
    print("=== Starting NCUA Chatbot Dynamic Query Integration Tests ===")
    
    # Initialize the chatbot
    chatbot = NCUAChatbot()
    
    # Test with a well-known credit union
    navy_federal = "Navy Federal Credit Union"
    penfed = "PenFed"
    
    # Test credit union extraction
    navy_cu_info = await test_credit_union_extraction(chatbot, navy_federal)
    
    # Test account description mapping
    financial_terms = ["assets", "loans", "net income", "members"]
    account_mappings = await test_account_description_mapping(chatbot, financial_terms)
    
    # Test basic questions
    await test_chatbot_query(chatbot, "total assets", navy_federal)
    await test_chatbot_query(chatbot, "net income", navy_federal)
    
    # Test another credit union to check context switching
    await test_chatbot_query(chatbot, "total assets", penfed)
    
    # Test follow-up question using context
    await test_chatbot_query(chatbot, "How many members do they have?")
    
    # Save detailed test results
    with open("dynamic_integration_results.json", "w") as f:
        results = {
            "navy_federal_info": navy_cu_info,
            "account_mappings": account_mappings,
        }
        json.dump(results, f, indent=2)
    
    print("\nTests completed and results saved to dynamic_integration_results.json")

if __name__ == "__main__":
    asyncio.run(run_tests()) 