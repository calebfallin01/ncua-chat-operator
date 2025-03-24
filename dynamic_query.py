#!/usr/bin/env python
"""
Dynamic SQL Query Construction for the NCUA Chatbot

This module replaces the generic queries in query_all_tables.py with targeted
SQL queries based on specific account codes identified from account descriptions.

Key functions:
1. build_sql_query - Constructs parameterized SQL queries based on account codes
2. execute_sql_query - Executes the SQL query and returns formatted results
3. query_by_account_codes - Main function for targeted queries by account codes
"""

import os
import sys
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple, Set
from dotenv import load_dotenv
from supabase import create_client, Client
from functools import lru_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DynamicSQLQuerier:
    """
    Build and execute dynamic SQL queries against the NCUA database
    based on specific account codes.
    """
    
    def __init__(self):
        """Initialize the connection to Supabase."""
        # Load environment variables
        load_dotenv()
        
        # Get API credentials from environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_API_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_API_KEY environment variables must be set")
        
        # Initialize Supabase client
        logger.info("Initializing Supabase client for dynamic queries...")
        self.client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")
        
        # Map from tablename field to actual table name
        self.tablename_to_table = {
            "FS220": "fs220_2024_12",
            "FS220A": "fs220a_2024_12",
            "FS220B": "fs220b_2024_12", 
            "FS220C": "fs220c_2024_12",
            "FS220D": "fs220d_2024_12",
            "FS220G": "fs220g_2024_12",
            "FS220H": "fs220h_2024_12",
            "FS220I": "fs220i_2024_12",
            "FS220J": "fs220j_2024_12",
            "FS220K": "fs220k_2024_12",
            "FS220L": "fs220l_2024_12",
            "FS220M": "fs220m_2024_12",
            "FS220N": "fs220n_2024_12",
            "FS220P": "fs220p_2024_12",
            "FS220Q": "fs220q_2024_12",
            "FS220R": "fs220r_2024_12",
            "FS220S": "fs220s_2024_12"
        }
        
        # Column name mapping for common tables
        self.table_id_columns = {
            "fs220_2024_12": "cu_number",
            "fs220a_2024_12": "cu_number",
            "fs220b_2024_12": "cu_number",
            "fs220c_2024_12": "cu_number",
            "fs220d_2024_12": "cu_number",
            "fs220g_2024_12": "cu_number",
            "fs220h_2024_12": "cu_number",
            "fs220i_2024_12": "cu_number",
            "fs220j_2024_12": "cu_number",
            "fs220k_2024_12": "cu_number",
            "fs220l_2024_12": "cu_number",
            "fs220m_2024_12": "cu_number",
            "fs220n_2024_12": "cu_number",
            "fs220p_2024_12": "cu_number",
            "fs220q_2024_12": "cu_number",
            "fs220r_2024_12": "cu_number",
            "fs220s_2024_12": "cu_number",
            "foicu_2024_12": "cu_number",
            "foicudes_2024_12": "cu_number",
            "atm_locations_2024_12": "cu_number",
            "credit_union_branch_information_2024_12": "cu_number",
            "tradenames_2024_12": "cu_number"
        }
        
        # Essential columns to always include in a query
        self.essential_columns = {
            "default": ["id", "cu_number", "cycle_date"],
            "fs220_2024_12": ["id", "cu_number", "cycle_date", "acct_010"], # Always include total assets
            "foicu_2024_12": ["id", "cu_number", "cu_name", "charter_no", "charteryr", "chartertype", "state"]
        }
        
    def normalize_cu_number(self, cu_number: Any) -> int:
        """
        Normalize the cu_number to handle different formats.
        
        Args:
            cu_number: Credit union number in various formats
            
        Returns:
            Normalized integer cu_number
        """
        try:
            # If it's already an integer, return it
            if isinstance(cu_number, int):
                return cu_number
                
            # If it's a string, try to convert to a number
            if isinstance(cu_number, str):
                # Convert to float first to handle decimal values like '227.0'
                float_val = float(cu_number)
                # Then convert to integer (this will drop any decimal part)
                return int(float_val)
                
            # If it's a float, convert to integer
            if isinstance(cu_number, float):
                return int(cu_number)
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not normalize cu_number {cu_number}: {e}")
            # Return the original value if conversion fails
            return cu_number
            
        # Return the original value if none of the above conversions worked
        return cu_number
    
    def get_actual_table_name(self, table_name: str) -> str:
        """
        Convert a table name from the account description to the actual table name.
        
        Args:
            table_name: Table name (e.g., "FS220")
            
        Returns:
            Actual table name (e.g., "fs220_2024_12")
        """
        # Check if it's already the actual table name
        if table_name.lower().endswith("_2024_12"):
            return table_name.lower()
            
        # Try to get from the mapping
        if table_name.upper() in self.tablename_to_table:
            return self.tablename_to_table[table_name.upper()]
        
        # Try with lowercase
        if table_name.lower() in [k.lower() for k, v in self.tablename_to_table.items()]:
            for k, v in self.tablename_to_table.items():
                if k.lower() == table_name.lower():
                    return v
        
        # Fallback: append the date suffix
        return f"{table_name.lower()}_2024_12"
    
    def get_id_column(self, table_name: str) -> str:
        """
        Get the ID column for a specific table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Name of the ID column
        """
        return self.table_id_columns.get(table_name.lower(), "cu_number")
    
    def build_sql_query(self, 
                       table_name: str, 
                       cu_number: Any, 
                       account_codes: List[str],
                       additional_columns: Optional[List[str]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Build a parameterized SQL query for a specific table and account codes.
        
        Args:
            table_name: Table name
            cu_number: Credit union number
            account_codes: List of account codes to include in the query
            additional_columns: Additional columns to include
            
        Returns:
            Tuple of (SQL query string, parameters)
        """
        # Normalize the cu_number
        normalized_cu_number = self.normalize_cu_number(cu_number)
        
        # Get the actual table name
        actual_table = self.get_actual_table_name(table_name)
        
        # Get the ID column for this table
        id_column = self.get_id_column(actual_table)
        
        # Ensure account_codes don't have duplicates and are properly formatted
        formatted_account_codes = set()
        for code in account_codes:
            # Ensure code has acct_ prefix if not already there
            if not code.lower().startswith("acct_") and not code.isdigit():
                formatted_code = f"acct_{code.lower()}"
            else:
                formatted_code = code.lower()
            formatted_account_codes.add(formatted_code)
        
        # Get essential columns for this table
        essential = self.essential_columns.get(actual_table, self.essential_columns["default"])
        
        # Create the full list of columns to select
        columns = list(essential)  # Start with essential columns
        
        # Add account codes
        columns.extend(formatted_account_codes)
        
        # Add any additional columns
        if additional_columns:
            columns.extend(additional_columns)
        
        # Remove duplicates while preserving order
        unique_columns = []
        seen = set()
        for col in columns:
            if col not in seen:
                unique_columns.append(col)
                seen.add(col)
        
        # Build the SQL query
        columns_str = ", ".join(unique_columns)
        sql = f"SELECT {columns_str} FROM {actual_table} WHERE {id_column} = :cu_number"
        
        # Parameters for the query
        params = {"cu_number": normalized_cu_number}
        
        logger.info(f"Built SQL query for table {actual_table}: {sql}")
        return sql, params
    
    def get_table_columns(self, table_name: str) -> List[str]:
        """
        Get a list of available columns for a specific table.
        
        Args:
            table_name: Table name to check
        
        Returns:
            List of column names that exist in the table
        """
        try:
            # Query a single row to get the column names
            result = self.client.table(table_name).select("*").limit(1).execute()
            
            if result and hasattr(result, 'data') and result.data:
                # Extract column names from the first record
                return list(result.data[0].keys())
            else:
                logger.warning(f"No data found in table {table_name} to determine columns")
                # Return common columns that are likely to exist
                return ["id", "cu_number", "cycle_date"]
        except Exception as e:
            logger.warning(f"Error getting columns for table {table_name}: {str(e)}")
            # Return common columns that are likely to exist
            return ["id", "cu_number", "cycle_date"]
    
    def execute_sql_query(self, sql: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute a SQL query against the database.
        
        Args:
            sql: SQL query
            params: Query parameters
            
        Returns:
            List of result records
        """
        try:
            start_time = time.time()
            logger.info(f"Executing SQL query: {sql} with params {params}")
            
            # Extract table name from SQL
            import re
            table_match = re.search(r'FROM\s+([^\s]+)', sql, re.IGNORECASE)
            column_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE)
            
            if table_match and column_match and "cu_number" in params:
                table_name = table_match.group(1)
                cu_number = params["cu_number"]
                
                # Get available columns for this table
                available_columns = self.get_table_columns(table_name)
                
                # Parse requested columns from SQL
                requested_columns = column_match.group(1).split(',')
                requested_columns = [col.strip() for col in requested_columns]
                
                # Filter to only use columns that exist in the table
                valid_columns = []
                for col in requested_columns:
                    if col == "*":
                        valid_columns.append("*")  # Select all columns
                    elif col in available_columns:
                        valid_columns.append(col)  # Column exists
                    else:
                        logger.warning(f"Column '{col}' not found in table {table_name}")
                
                # If we have valid columns, execute the query directly
                if valid_columns:
                    column_str = "*" if "*" in valid_columns else ",".join(valid_columns)
                    result = self.client.table(table_name).select(column_str).eq("cu_number", cu_number).execute()
                    if result and hasattr(result, 'data'):
                        duration = time.time() - start_time
                        logger.info(f"Direct query executed in {duration:.2f}s, returned {len(result.data)} records")
                        return result.data
            
            # If we get here, try the execute_sql RPC method as a backup
            try:
                # Format the SQL query with parameters directly
                formatted_sql = sql
                for key, value in params.items():
                    # Replace :param_name with actual values
                    placeholder = f":{key}"
                    if isinstance(value, str):
                        # Escape string values with single quotes
                        formatted_sql = formatted_sql.replace(placeholder, f"'{value}'")
                    else:
                        # For non-string values, just convert to string
                        formatted_sql = formatted_sql.replace(placeholder, str(value))
                
                # Call the RPC function with the single sql_query parameter
                result = self.client.rpc('execute_sql', {
                    'sql_query': formatted_sql
                }).execute()
                
                # Process the results
                if result and hasattr(result, 'data') and result.data:
                    duration = time.time() - start_time
                    logger.info(f"RPC query executed in {duration:.2f}s, returned {len(result.data)} records")
                    return result.data
                else:
                    logger.warning("RPC query returned no data or invalid response")
            except Exception as rpc_error:
                logger.error(f"Error in RPC query: {str(rpc_error)}")
            
            # Last resort: Simple select all
            try:
                if table_match and "cu_number" in params:
                    table_name = table_match.group(1)
                    cu_number = params["cu_number"]
                    result = self.client.table(table_name).select("*").eq("cu_number", cu_number).execute()
                    if result and hasattr(result, 'data'):
                        logger.info(f"Fallback query returned {len(result.data)} records")
                        return result.data
            except Exception as fallback_error:
                logger.error(f"Fallback query also failed: {str(fallback_error)}")
            
            return []
                
        except Exception as e:
            logger.error(f"Error executing SQL query: {str(e)}")
            
            # Last resort: Try a very simple select all
            try:
                if "cu_number" in params:
                    import re
                    table_match = re.search(r'FROM\s+([^\s]+)', sql, re.IGNORECASE)
                    if table_match:
                        table_name = table_match.group(1)
                        result = self.client.table(table_name).select("*").eq("cu_number", params["cu_number"]).execute()
                        if result and hasattr(result, 'data'):
                            logger.info(f"Emergency fallback query returned {len(result.data)} records")
                            return result.data
            except Exception as final_error:
                logger.error(f"All query methods failed: {str(final_error)}")
            
            return []
    
    def query_by_account_codes(self, 
                              cu_number: str, 
                              account_mappings: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """
        Perform targeted queries based on account codes.
        
        Args:
            cu_number: Credit union number
            account_mappings: Dictionary mapping account codes to table names and descriptions
            
        Returns:
            Dictionary of query results organized by table
        """
        # Organize account codes by table
        tables_to_query = {}
        
        # First, organize all the account codes by table
        for acct_code, mapping in account_mappings.items():
            table_name = mapping.get("tablename")
            if not table_name:
                continue
            
            # Get the actual table name
            actual_table = self.get_actual_table_name(table_name)
            
            # Add this account to the list for this table
            if actual_table not in tables_to_query:
                tables_to_query[actual_table] = []
            
            # Store the account code (make sure it has acct_ prefix)
            if acct_code.lower().startswith('acct_') or acct_code.isdigit():
                tables_to_query[actual_table].append(acct_code)
            else:
                tables_to_query[actual_table].append(f"acct_{acct_code}")
        
        # Add core tables that should always be queried regardless
        core_tables = ["foicu_2024_12"]
        for table in core_tables:
            if table not in tables_to_query:
                tables_to_query[table] = []
        
        logger.info(f"Querying {len(tables_to_query)} tables with targeted account codes")
        
        # Query each table and collect results
        all_results = {
            "credit_union_info": {
                "cu_number": cu_number
            },
            "results": {},
            "query_type": "dynamic_sql",
            "tables_queried": list(tables_to_query.keys()),
            "account_codes": account_mappings
        }
        
        # Query each table
        for table_name, account_codes in tables_to_query.items():
            # Build the SQL query
            sql, params = self.build_sql_query(table_name, cu_number, account_codes)
            
            # Execute the query
            results = self.execute_sql_query(sql, params)
            
            # Store the results
            if results:
                all_results["results"][table_name] = results
                logger.info(f"Retrieved {len(results)} records from {table_name}")
            else:
                # Store empty list to indicate the table was queried
                all_results["results"][table_name] = []
                logger.info(f"No records found in {table_name}")
        
        # Get credit union name and other metadata from foicu_2024_12 if available
        if "foicu_2024_12" in all_results["results"] and all_results["results"]["foicu_2024_12"]:
            foicu_data = all_results["results"]["foicu_2024_12"][0]  # Take the first record
            all_results["credit_union_info"]["cu_name"] = foicu_data.get("cu_name")
            all_results["credit_union_info"]["charter_no"] = foicu_data.get("charter_no")
            all_results["credit_union_info"]["charteryr"] = foicu_data.get("charteryr")
            all_results["credit_union_info"]["chartertype"] = foicu_data.get("chartertype")
            all_results["credit_union_info"]["state"] = foicu_data.get("state")
        
        return all_results
    
    def build_combined_query(self, 
                           cu_number: str, 
                           requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a combined query that fetches multiple tables in a single request.
        This optimizes by reducing round-trips to the database.
        
        Args:
            cu_number: Credit union number
            requests: List of query request dictionaries, each with table_name and account_codes
            
        Returns:
            Combined query results
        """
        # Implementation of combined queries - for future optimization
        pass
    
    def query_credit_union_info(self, cu_number: str) -> Dict[str, Any]:
        """
        Get basic credit union information from foicu_2024_12.
        
        Args:
            cu_number: Credit union number
            
        Returns:
            Dictionary with credit union info
        """
        # This specific query is used to get basic credit union information
        sql, params = self.build_sql_query(
            "foicu_2024_12", 
            cu_number, 
            [], 
            ["cu_name", "charter_no", "charteryr", "chartertype", "state", "cycle_date"]
        )
        
        results = self.execute_sql_query(sql, params)
        
        if results:
            # Return the first record
            credit_union_info = results[0]
            return {
                "cu_number": cu_number,
                "cu_name": credit_union_info.get("cu_name"),
                "charter_no": credit_union_info.get("charter_no"),
                "charteryr": credit_union_info.get("charteryr"),
                "chartertype": credit_union_info.get("chartertype"),
                "state": credit_union_info.get("state")
            }
        else:
            return {"cu_number": cu_number}

# Singleton instance for reuse
_instance = None

def get_querier() -> DynamicSQLQuerier:
    """
    Get or create a singleton instance of the DynamicSQLQuerier.
    
    Returns:
        DynamicSQLQuerier instance
    """
    global _instance
    if _instance is None:
        _instance = DynamicSQLQuerier()
    return _instance

if __name__ == "__main__":
    # Simple command-line interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a dynamic SQL query based on account codes')
    parser.add_argument('cu_number', help='Credit union number to query')
    parser.add_argument('--table', help='Table to query')
    parser.add_argument('--account', action='append', help='Account code to query (can specify multiple)')
    parser.add_argument('--output', help='Output file for results')
    args = parser.parse_args()
    
    querier = DynamicSQLQuerier()
    
    if args.table and args.account:
        # Run a specific query
        sql, params = querier.build_sql_query(args.table, args.cu_number, args.account)
        results = querier.execute_sql_query(sql, params)
        
        # Print or save results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            print(json.dumps(results, indent=2))
    else:
        print("Please specify both --table and at least one --account") 