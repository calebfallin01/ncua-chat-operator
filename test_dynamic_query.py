#!/usr/bin/env python
"""
Test script for the dynamic SQL query functionality.

This script tests:
1. Basic SQL query construction
2. SQL query execution
3. Query by account codes
"""

import json
import logging
from dynamic_query import DynamicSQLQuerier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_query(querier, cu_number):
    """Test a basic query for a credit union's total assets."""
    print("\n--- Testing Basic Query ---")
    
    # Build a query for total assets (acct_010)
    sql, params = querier.build_sql_query(
        "fs220_2024_12",
        cu_number,
        ["acct_010"]  # Total assets account code
    )
    
    print(f"SQL Query: {sql}")
    print(f"Parameters: {params}")
    
    # Execute the query
    results = querier.execute_sql_query(sql, params)
    
    # Print the results
    if results:
        print(f"Found {len(results)} records")
        print(f"Total Assets: ${results[0].get('acct_010', 0):,.2f}")
    else:
        print("No results found")
        
    return results

def test_multiple_accounts(querier, cu_number):
    """Test querying multiple accounts at once."""
    print("\n--- Testing Multiple Accounts Query ---")
    
    # Query for assets, loans, and shares
    sql, params = querier.build_sql_query(
        "fs220_2024_12",
        cu_number,
        ["acct_010", "acct_018", "acct_025"]  # Assets, Shares, Loans
    )
    
    print(f"SQL Query: {sql}")
    
    # Execute the query
    results = querier.execute_sql_query(sql, params)
    
    # Print the results
    if results:
        print(f"Found {len(results)} records")
        
        # Format the results
        record = results[0]
        print(f"Total Assets (acct_010): ${record.get('acct_010', 0):,.2f}")
        print(f"Total Shares (acct_018): ${record.get('acct_018', 0):,.2f}")
        print(f"Total Loans (acct_025): ${record.get('acct_025', 0):,.2f}")
    else:
        print("No results found")
        
    return results

def test_account_codes_query(querier, cu_number):
    """Test the query_by_account_codes function."""
    print("\n--- Testing Query by Account Codes ---")
    
    # Create sample account mappings
    account_mappings = {
        "acct_010": {"tablename": "FS220", "name": "Total Assets"},
        "acct_018": {"tablename": "FS220", "name": "Total Shares/Deposits"},
        "acct_025": {"tablename": "FS220", "name": "Total Loans"},
        "acct_661a": {"tablename": "FS220A", "name": "Net Income"}
    }
    
    # Run the query
    all_results = querier.query_by_account_codes(cu_number, account_mappings)
    
    # Print the results
    print(f"Queried {len(all_results.get('results', {}))} tables")
    
    # Check if we have results
    results = all_results.get("results", {})
    for table_name, table_results in results.items():
        print(f"\nTable: {table_name}")
        print(f"Records: {len(table_results)}")
        
        if table_results:
            # Print the first record's fields
            print("Fields in first record:")
            for field, value in table_results[0].items():
                if field.startswith("acct_") and value:
                    print(f"  {field}: ${value:,.2f}")
    
    # Print credit union info
    cu_info = all_results.get("credit_union_info", {})
    if cu_info:
        print(f"\nCredit Union: {cu_info.get('cu_name', 'Unknown')}")
        print(f"CU Number: {cu_info.get('cu_number', 'Unknown')}")
    
    return all_results

def run_tests():
    """Run all the tests."""
    # Initialize the querier
    querier = DynamicSQLQuerier()
    
    # Test with a well-known credit union
    print("\n=== Testing with Navy Federal Credit Union ===")
    navy_federal_cu_number = "5536"  # Navy Federal Credit Union
    
    # Run the tests
    basic_results = test_basic_query(querier, navy_federal_cu_number)
    multiple_results = test_multiple_accounts(querier, navy_federal_cu_number)
    account_codes_results = test_account_codes_query(querier, navy_federal_cu_number)
    
    # Save detailed results to a file
    with open("test_dynamic_query_results.json", "w") as f:
        json.dump({
            "basic_results": basic_results,
            "multiple_results": multiple_results,
            "account_codes_results": account_codes_results
        }, f, indent=2)
    
    print("\nTest results saved to test_dynamic_query_results.json")

if __name__ == "__main__":
    run_tests() 