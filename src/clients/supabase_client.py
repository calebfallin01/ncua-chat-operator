"""Supabase client wrapper for structured database operations."""

import logging
import json
import sys
import os
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add the parent directory to the path so we can import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.settings import settings

# Configure logger
logger = logging.getLogger(__name__)

class SupabaseClient:
    """Client for interacting with Supabase structured database."""
    
    def __init__(self):
        """Initialize the Supabase client."""
        self.client = create_client(
            settings.supabase_url,
            settings.supabase_api_key
        )
        
        # Financial statement tables (December 2024 version)
        self.fs_tables = [
            "fs220_2024_12",
            "fs220a_2024_12",
            "fs220b_2024_12",
            "fs220c_2024_12",
            "fs220d_2024_12",
            "fs220e_2024_12",
            "fs220f_2024_12",
            "fs220g_2024_12"
        ]
        
        # Account description table
        self.acct_desc_table = "acctdesc_2024_12"
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def find_account_for_description(self, description: str) -> dict:
        """
        Find the account and table name based on a description in the acctdesc_2024_12 table.
        
        Args:
            description: Text description to search for (e.g., "asset size", "total assets")
            
        Returns:
            Dictionary containing the account, tablename, acctname, and acctdesc
        """
        try:
            # Search in the account description table using ILIKE for case-insensitive matching
            # First try the acctname column (more specific)
            response = self.client.table(self.acct_desc_table).select("*").ilike("acctname", f"%{description}%").execute()
            
            if not response.data or len(response.data) == 0:
                # Try the acctdesc column (more descriptive)
                response = self.client.table(self.acct_desc_table).select("*").ilike("acctdesc", f"%{description}%").execute()
            
            if response.data and len(response.data) > 0:
                # Return the first matching record
                logger.info(f"Found account matching '{description}': {response.data[0]}")
                return response.data[0]
            else:
                logger.warning(f"No account found matching description: {description}")
                return None
        except Exception as e:
            logger.error(f"Error searching for account by description: {str(e)}")
            # Return a helpful error message but don't raise
            logger.warning(f"Table '{self.acct_desc_table}' may not exist in the database")
            return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def fetch_financial_data(self, cu_number: str, account_info: dict) -> dict:
        """
        Fetch financial data for a credit union based on the account_info.
        
        Args:
            cu_number: The credit union identifier
            account_info: Dictionary containing account and tablename
            
        Returns:
            Dictionary with the financial data
        """
        try:
            if not account_info:
                logger.warning("No account info provided")
                return None
                
            # Extract account and tablename from account_info
            account = account_info.get("account")
            tablename = account_info.get("tablename").lower()
            acctname = account_info.get("acctname")
            acctdesc = account_info.get("acctdesc")
            
            # Construct the full table name (e.g., fs220_2024_12)
            full_tablename = f"{tablename.lower()}_2024_12"
            
            # Check if this is a valid table name
            if full_tablename not in self.fs_tables:
                logger.warning(f"Table '{full_tablename}' is not in the list of known financial tables")
            
            # Convert account name to lowercase to match database column naming conventions
            # This fixes the case-sensitivity issue with column names
            lowercase_account = account.lower()
            logger.info(f"Using lowercase account name: {lowercase_account} (original: {account})")
            
            # Convert cu_number to integer if it's a float or string with decimal
            # This fixes the type mismatch with the database
            try:
                # Try to convert to float first to handle string values like "216.0"
                float_cu_number = float(cu_number)
                # Then convert to integer by truncating decimal part
                int_cu_number = int(float_cu_number)
                logger.info(f"Converted cu_number from {cu_number} to integer {int_cu_number}")
                cu_number = int_cu_number
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert cu_number {cu_number} to integer: {str(e)}")
                # Keep the original value if conversion fails
            
            # Query the table for the specified account and cu_number
            response = self.client.table(full_tablename).select(lowercase_account).eq("cu_number", cu_number).execute()
            
            if response.data and len(response.data) > 0:
                # Extract the value and add context
                value = response.data[0].get(lowercase_account)
                result = {
                    "value": value,
                    "account": account,
                    "account_name": acctname,
                    "account_description": acctdesc,
                    "table": full_tablename,
                    "cu_number": cu_number
                }
                logger.info(f"Found financial data: {result}")
                return result
            else:
                logger.warning(f"No financial data found for cu_number {cu_number} and account {lowercase_account} in table {full_tablename}")
                return None
        except Exception as e:
            logger.error(f"Error fetching financial data: {str(e)}")
            # Return informative message but don't raise
            logger.warning(f"Table '{full_tablename}' may not exist in the database")
            return None
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def get_data_for_query(self, cu_number: str, query_description: str) -> dict:
        """
        Combined method that finds the right account based on description and fetches the data.
        
        Args:
            cu_number: The credit union identifier
            query_description: Description of what to search for
            
        Returns:
            Dictionary with the financial data
        """
        try:
            # Step 1: Find the account information based on the description
            account_info = await self.find_account_for_description(query_description)
            
            if not account_info:
                logger.warning(f"Could not find account info for description: {query_description}")
                # Return a mock result for development purposes
                return {
                    "value": "DATA_NOT_AVAILABLE",
                    "account": "unknown",
                    "account_name": query_description,
                    "account_description": f"Data for {query_description}",
                    "table": "unknown",
                    "cu_number": cu_number,
                    "note": "This is a mock response because the database is still being set up"
                }
                
            # Step 2: Fetch the financial data using the account information
            result = await self.fetch_financial_data(cu_number, account_info)
            
            if not result:
                # Return a mock result for development purposes
                return {
                    "value": "DATA_NOT_AVAILABLE",
                    "account": account_info.get("account", "unknown"),
                    "account_name": account_info.get("acctname", query_description),
                    "account_description": account_info.get("acctdesc", f"Data for {query_description}"),
                    "table": account_info.get("tablename", "unknown"),
                    "cu_number": cu_number,
                    "note": "This is a mock response because the database is still being set up"
                }
            
            return result
        except Exception as e:
            logger.error(f"Error getting data for query: {str(e)}")
            # Return informative mock data instead of raising
            return {
                "value": "ERROR_FETCHING_DATA",
                "query": query_description,
                "cu_number": cu_number,
                "error": str(e),
                "note": "An error occurred while retrieving this data. The database may still be in setup."
            }

# Create a singleton instance
supabase_client = SupabaseClient() 