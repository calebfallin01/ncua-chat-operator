#!/usr/bin/env python3
"""
Test script to specifically test the asset retrieval and interpretation for Navy Federal.
This will help identify why we're getting the wrong asset value.
"""

import asyncio
import json
import logging
from ncua_chatbot import NCUAChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

async def test_navy_federal_assets():
    """Test Navy Federal's asset retrieval and interpretation"""
    print("\n=== Testing Navy Federal Asset Retrieval ===")
    
    # Initialize the chatbot
    chatbot = NCUAChatbot()
    
    # Navy Federal's cu_number
    cu_number = "5536"
    
    # First, test direct financial data retrieval
    print("\n1. Testing direct query_financial_data:")
    financial_data = await chatbot.query_financial_data_async(cu_number)
    
    if financial_data:
        print(f"Retrieved data from {len(financial_data.get('results', {}))} tables")
        
        # Check for asset data
        for table_name, records in financial_data.get('results', {}).items():
            if not records:
                continue
                
            record = records[0]
            if 'acct_010' in record:
                print(f"Found Total Assets in {table_name}: ${record['acct_010']:,}")
    else:
        print("Failed to retrieve any financial data")
    
    # Now test the full query and interpretation pipeline
    print("\n2. Testing full process_query pipeline:")
    test_query = "What is Navy Federal Credit Union's total assets?"
    answer = await chatbot.process_query(test_query)
    print(f"Query: {test_query}")
    print(f"Answer: {answer}")
    
    # Examine what data is being sent to OpenAI
    print("\n3. Testing data extraction for OpenAI:")
    # Create test data with just Navy Federal info
    cu_info = {
        "cu_name": "NAVY FEDERAL CREDIT UNION",
        "cu_number": cu_number
    }
    
    # Extract metrics that would be sent to OpenAI
    if financial_data:
        extracted_data, _ = chatbot.extract_key_financial_metrics(financial_data, test_query)
        print("Data that would be sent to OpenAI:")
        print(extracted_data)

async def main():
    """Run the test"""
    await test_navy_federal_assets()

if __name__ == "__main__":
    asyncio.run(main()) 