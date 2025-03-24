#!/usr/bin/env python3
"""
Test script to verify that our fix for the shares query works
"""

import sys
import logging
from ncua_chatbot import NCUAChatbot

# Set up simple console logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_shares_query():
    print("\n=== TESTING SHARES QUERY FOR NAVY FEDERAL CREDIT UNION ===\n")
    
    # Initialize the chatbot
    chatbot = NCUAChatbot()
    
    # Test finding Navy Federal Credit Union
    print("Looking up Navy Federal Credit Union...")
    cu_info = chatbot.find_credit_union("Navy Federal Credit Union")
    
    if not cu_info:
        print("ERROR: Could not find Navy Federal Credit Union")
        return
        
    print(f"Found credit union: {cu_info.get('cu_name')} with number {cu_info.get('cu_number')}")
    
    # Test extracting shares-related terms from a query
    test_query = "how much does navy federal have in shares and deposits?"
    print(f"\nExtracting financial concepts from query: '{test_query}'")
    terms = chatbot.enhanced_extract_query_terms(test_query)
    print(f"Extracted terms: {terms}")
    
    # Test account mapping with shares terms
    print(f"\nTesting account mapping for terms: {terms}")
    account_mappings = chatbot.enhanced_query_account_descriptions(terms)
    print(f"Found {len(account_mappings)} account mappings:")
    for code, details in account_mappings.items():
        print(f"  - {code}: {details}")
    
    # Test the full query
    print(f"\nTesting full enhanced targeted query...")
    cu_number = cu_info.get('cu_number')
    results = chatbot.enhanced_targeted_financial_query(cu_number, terms)
    
    if results:
        print(f"Query successful! Found results from {len(results.get('results', {}))} tables")
        
        # Print actual share values if found
        share_found = False
        for table, records in results.get('results', {}).items():
            print(f"Data from table: {table} ({len(records)} records)")
            for record in records[:1]:  # Just the first record for brevity
                for field, value in record.items():
                    if field.lower() == 'acct_018' or field == 'acct_013':
                        share_found = True
                        print(f"SHARES VALUE FOUND: {field} = {value}")
        
        if not share_found:
            print("No specific shares values found in the results")
    else:
        print("ERROR: Query returned no results")

if __name__ == "__main__":
    test_shares_query() 