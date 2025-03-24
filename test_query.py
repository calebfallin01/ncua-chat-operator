#!/usr/bin/env python3
"""
Test script to debug the issue with shares queries
This directly tests the key problematic functions with Navy Federal and shares data
"""

import sys
import json
import logging
import subprocess
from typing import Dict, Any, Optional, List
from ncua_chatbot import NCUAChatbot

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_query")

def test_shares_query():
    """Test the shares query directly with verbose output"""
    print("\n=== TESTING SHARES QUERY FOR NAVY FEDERAL ===\n")
    
    try:
        # Initialize the chatbot
        chatbot = NCUAChatbot()
        
        # First find Navy Federal
        cu_name = "Navy Federal Credit Union"
        print(f"Finding credit union: {cu_name}")
        cu_info = chatbot.find_credit_union(cu_name)
        
        if not cu_info:
            print("ERROR: Could not find Navy Federal Credit Union")
            return
            
        print(f"Successfully found credit union: {cu_info}")
        cu_number = cu_info.get('cu_number')
        
        # Test specific shares query
        metrics = ["total shares and deposits", "shares"]
        print(f"\n=== Testing enhanced_query_account_descriptions with metrics: {metrics} ===")
        account_mappings = chatbot.enhanced_query_account_descriptions(metrics)
        
        print(f"Account mappings result: {json.dumps(account_mappings, indent=2)}")
        print(f"Found {len(account_mappings)} account mappings")
        
        # Check the tablename_to_table mapping
        print("\n=== Checking tablename_to_table mapping ===")
        print(f"tablename_to_table: {json.dumps(chatbot.tablename_to_table, indent=2)}")
        
        # Now test the enhanced targeted query
        print(f"\n=== Testing enhanced_targeted_financial_query for cu_number {cu_number} ===")
        result = chatbot.enhanced_targeted_financial_query(cu_number, metrics)
        
        if result:
            print(f"Query successful! Results for {len(result.get('results', {}))} tables")
            for table, records in result.get('results', {}).items():
                print(f"  - Table: {table}, Records: {len(records)}")
        else:
            print("ERROR: Query returned no results")
            
        # Try direct lookup of specific account code
        print("\n=== Testing direct account code lookup ===")
        # Look for ACCT_018 (Total Shares and Deposits)
        cmd = ["python", "interactive_query.py", 
               "--table", "fs220_2024_12", 
               "--cu-number", cu_number,
               "--columns", "cu_number,acct_018",
               "--output-json"]
               
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print(f"Return code: {result.returncode}")
        print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
            
    except Exception as e:
        import traceback
        print(f"Error in test: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_shares_query() 