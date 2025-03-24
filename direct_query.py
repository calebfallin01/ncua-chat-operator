#!/usr/bin/env python3
"""
Direct query tool to get specific credit union data without the complexity of the chatbot.
"""

import os
import json
import sys
from supabase import create_client
from dotenv import load_dotenv

def query_total_assets(cu_number):
    """Directly query a credit union's total assets"""
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_API_KEY environment variables must be set")
        sys.exit(1)
    
    # Initialize Supabase client
    print(f"Querying assets for Credit Union #{cu_number}")
    client = create_client(supabase_url, supabase_key)
    
    # Query for assets (acct_010)
    result = client.table("fs220_2024_12").select("acct_010").eq("cu_number", cu_number).execute()
    
    if result and hasattr(result, 'data') and result.data:
        assets = result.data[0].get('acct_010')
        print(f"Total Assets: ${assets:,.2f}")
        return assets
    else:
        print("No asset data found")
        return None

if __name__ == "__main__":
    # Get credit union number from command line
    if len(sys.argv) > 1:
        query_total_assets(sys.argv[1])
    else:
        print("Usage: python direct_query.py <cu_number>")
        print("Example: python direct_query.py 5536")
        sys.exit(1) 