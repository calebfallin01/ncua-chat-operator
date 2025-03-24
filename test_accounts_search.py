#!/usr/bin/env python3
"""
Test script for accounts_vector_search.py

This script demonstrates how to use the accounts_vector_search.py module
to search for financial account codes in the vector database.
"""

import json
import subprocess
import os
import sys

# Sample queries to test with
TEST_QUERIES = [
    # Basic financial metrics
    "total assets",
    "net income",
    "total loans",
    "shares and deposits",
    "members",
    
    # More specific complex financial metrics
    "return on assets",
    "capital adequacy ratio",
    "loan-to-share ratio",
    "delinquent loans as a percentage of total loans",
    "investment in credit union service organizations",
    "net interest margin",
    "total equity ratio",
    "allowance for credit losses",
    "non-interest income",
    "mortgage loan originations",
    
    # Descriptive language-based queries
    "dollar amount of member business loans for agriculture purposes",
    "sum of all deposits with maturity less than 1 year",
    "total number of delinquent loans",
    "report the monthly amount of income from investments",
    "average balance of share draft accounts"
]

# Sample queries with explicit categories to test the category refinement
CATEGORY_TEST_QUERIES = [
    # Format: (query, expected_category)
    ("total assets", "Other Assets"),
    ("net income", "Income"),
    ("total loans", "Loans"),
    ("delinquent loans", "Delinquency"),
    ("mortgage originations", "Specialized Lending"),
    ("cash on hand", "Cash and Cash Equivalents"),
    ("net worth ratio", "Net Worth"),
    ("tier 1 capital", "Equity"),
    ("investment in CUSOs", "CUSO"),
    ("charge-offs", "Charge Offs and Recoveries")
]

# Test text with embedded queries
TEST_TEXT = "What is the {total assets} of Navy Federal? How much {net income} did they report? What's their {loan-to-share ratio}? How many {members} do they have? What {investments in CUSOs} do they report?"

# Test text with more descriptive embedded queries
DESCRIPTIVE_TEST_TEXT = "What is the {dollar amount of member business loans for agriculture purposes}? What is the {sum of all deposits with maturity less than 1 year}? What is the {total number of delinquent loans}?"

def run_test(test_type, command, description):
    """Run a test and print the results"""
    print(f"\n{'=' * 80}")
    print(f"TEST: {test_type}")
    print(f"DESCRIPTION: {description}")
    print(f"COMMAND: {' '.join(command)}")
    print(f"{'-' * 80}")
    
    try:
        # Run the command and capture output
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if successful
        if result.returncode == 0:
            # Parse the JSON output
            try:
                output = json.loads(result.stdout)
                print(f"RESULT: Success ✓")
                print(f"OUTPUT:\n{json.dumps(output, indent=2)}")
            except json.JSONDecodeError:
                print(f"RESULT: Output parsing failed ✗")
                print(f"OUTPUT:\n{result.stdout}")
        else:
            print(f"RESULT: Command failed ✗")
            print(f"ERROR:\n{result.stderr}")
    
    except Exception as e:
        print(f"RESULT: Exception occurred ✗")
        print(f"ERROR: {str(e)}")

def main():
    """Run tests for the accounts vector search script"""
    print("ACCOUNTS VECTOR SEARCH TEST SCRIPT")
    print("This script tests the functionality of accounts_vector_search.py")
    
    # Make sure the main script exists
    if not os.path.exists("accounts_vector_search.py"):
        print("ERROR: accounts_vector_search.py not found in the current directory")
        return
    
    # Test 1: Direct query mode for basic metrics
    for query in TEST_QUERIES[:5]:  # Test basic metrics
        command = ["python", "accounts_vector_search.py", "--query", query]
        run_test("Direct Basic Query", command, f"Search for '{query}' directly")
    
    # Test 2: Direct query mode for complex metrics
    for query in TEST_QUERIES[5:10]:  # Test complex metrics
        command = ["python", "accounts_vector_search.py", "--query", query]
        run_test("Direct Complex Query", command, f"Search for '{query}' directly")
        
    # Test 3: Direct query mode for descriptive queries
    for query in TEST_QUERIES[15:]:  # Test descriptive queries
        command = ["python", "accounts_vector_search.py", "--query", query]
        run_test("Direct Descriptive Query", command, f"Search for '{query}' directly")
    
    # Test 4: Category-based search
    for query, expected_category in CATEGORY_TEST_QUERIES:
        command = ["python", "accounts_vector_search.py", "--query", query]
        run_test(f"Category Search: {expected_category}", command, f"Search for '{query}' in category '{expected_category}'")
    
    # Test 5: Input text with embedded queries
    command = ["python", "accounts_vector_search.py", "--input", TEST_TEXT]
    run_test("Embedded Queries", command, "Process text with multiple embedded queries")
    
    # Test 6: Input text with descriptive embedded queries
    command = ["python", "accounts_vector_search.py", "--input", DESCRIPTIVE_TEST_TEXT]
    run_test("Descriptive Embedded Queries", command, "Process text with descriptive embedded queries")
    
    # Test 7: Compact output mode
    command = ["python", "accounts_vector_search.py", "--query", "total assets", "--compact"]
    run_test("Compact Output", command, "Get simplified output without scores")
    
    # Test 8: Test enriched query through the chatbot with descriptive queries
    try:
        print("\nTesting query enrichment through chatbot...")
        from ncua_chatbot import NCUAChatbot
        
        chatbot = NCUAChatbot()
        
        # Test complex financial metric queries
        for query in TEST_QUERIES[6:9]:  # Test a few complex queries
            print(f"\nEnriching query: '{query}'")
            enriched_query = chatbot.generate_account_search_query(query)
            print(f"\nEnriched to: {enriched_query}")
            
            # Test the enriched query
            command = ["python", "accounts_vector_search.py", "--input", f"{{{enriched_query}}}"]
            run_test("Enriched Query", command, f"Search with enriched query for '{query}'")
            
        # Test descriptive language queries
        for query in TEST_QUERIES[15:18]:  # Test a few descriptive queries
            print(f"\nEnriching descriptive query: '{query}'")
            enriched_query = chatbot.generate_account_search_query(query)
            print(f"\nEnriched to: {enriched_query}")
            
            # Test the enriched query
            command = ["python", "accounts_vector_search.py", "--input", f"{{{enriched_query}}}"]
            run_test("Enriched Descriptive Query", command, f"Search with enriched query for '{query}'")
            
    except Exception as e:
        print(f"ERROR testing query enrichment: {str(e)}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main() 