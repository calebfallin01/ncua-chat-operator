#!/usr/bin/env python
"""
Script to query all tables in Supabase based on a credit union number (cu_number).
This script will connect to Supabase, retrieve a list of all tables, and then
query each table for records matching the provided cu_number.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Set
from dotenv import load_dotenv
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SupabaseAllTablesQuerier:
    """Query all tables in Supabase database based on a cu_number."""
    
    def __init__(self, verbose=False):
        """Initialize the Supabase client."""
        # Load environment variables
        load_dotenv()
        
        # Get API credentials from environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_API_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_API_KEY environment variables must be set")
        
        # Initialize Supabase client
        logger.info("Initializing Supabase client...")
        self.client = create_client(self.supabase_url, self.supabase_key)
        logger.info("Supabase client initialized successfully")
        
        # Set verbosity level
        self.verbose = verbose
        
        # List of all available tables in the database
        self.all_tables = [
            "acct_desctradenames_2024_12",
            "acctdesc_2024_12",
            "atm_locations_2024_12",
            "credit_union_branch_information_2024_12",
            "foicu_2024_12",
            "foicudes_2024_12",
            "fs220_2024_12",
            "fs220a_2024_12",
            "fs220b_2024_12",
            "fs220c_2024_12",
            "fs220d_2024_12",
            "fs220g_2024_12",
            "fs220h_2024_12",
            "fs220i_2024_12",
            "fs220j_2024_12",
            "fs220k_2024_12",
            "fs220l_2024_12",
            "fs220m_2024_12",
            "fs220n_2024_12",
            "fs220p_2024_12",
            "fs220q_2024_12",
            "fs220r_2024_12",
            "fs220s_2024_12",
            "tradenames_2024_12"
        ]
        
        # Store which ID column to use for each table
        self.table_id_columns = {}
        
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
        
        # Store account descriptions 
        self.account_descriptions = {}
        
        # Always fallback mode to query all tables
        self.fallback_mode = True

    def normalize_cu_number(self, cu_number: Any) -> int:
        """
        Normalize the cu_number to handle different formats like '227.0' or '227'.
        
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
    
    def check_table_existence(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if the table exists, False otherwise
        """
        try:
            # Simple check to see if table exists - just try to get count
            response = self.client.table(table_name).select("count").limit(1).execute()
            # If we get here, the table exists
            return True
        except Exception as e:
            # Table likely doesn't exist or we don't have permission
            return False
    
    def try_query_with_column(self, table_name: str, column_name: str, cu_number: Any) -> List[Dict[str, Any]]:
        """
        Try to query a table with a specific column for the cu_number.
        
        Args:
            table_name: Name of the table to query
            column_name: Name of the column to filter on
            cu_number: Credit union number to search for
            
        Returns:
            List of matching records, or None if the query fails
        """
        try:
            # Try to query with the exact value
            response = self.client.table(table_name).select("*").eq(column_name, cu_number).execute()
            if response and hasattr(response, 'data'):
                if response.data:
                    return response.data
                    
                # If no results, also try with string version if cu_number is an integer
                if isinstance(cu_number, int):
                    str_response = self.client.table(table_name).select("*").eq(column_name, str(cu_number)).execute()
                    if str_response and hasattr(str_response, 'data'):
                        return str_response.data
                
                # If still no results, but query succeeded, return empty list
                return []
                
        except Exception as e:
            # This column didn't work, return None to indicate failure
            return None
        
        # If we get here, the query succeeded but returned no data
        return []
    
    def query_table(self, table_name: str, cu_number: Any) -> List[Dict[str, Any]]:
        """
        Query a specific table for records matching the cu_number.
        
        Args:
            table_name: Name of the table to query
            cu_number: Credit union number to search for
            
        Returns:
            List of matching records
        """
        # Normalize the cu_number to handle decimal values from vector search
        normalized_cu_number = self.normalize_cu_number(cu_number)
        logger.info(f"Querying table {table_name} for cu_number {normalized_cu_number} (original: {cu_number})")
        
        # Use the known ID column if we have it
        if table_name in self.table_id_columns:
            id_column = self.table_id_columns[table_name]
            records = self.try_query_with_column(table_name, id_column, normalized_cu_number)
            if records is not None:  # Query succeeded
                return records
        
        # Try common column names for the cu_number
        possible_columns = [
            "cu_number", 
            "CU_NUMBER", 
            "cu_num", 
            "CU_NUM", 
            "cunumber",
            "cunum",
            "id",
            "credit_union_number",
            "credit_union_id"
        ]
        
        for column in possible_columns:
            records = self.try_query_with_column(table_name, column, normalized_cu_number)
            if records is not None:  # Query succeeded
                # Remember this column for future queries
                self.table_id_columns[table_name] = column
                if records:
                    logger.info(f"Found {len(records)} records in table {table_name} using column {column}")
                return records
                
        # If the standard approach fails, try a direct query as fallback
        logger.info(f"No records found in table {table_name} for cu_number {normalized_cu_number}")
        return []
    
    def query_table_directly(self, table_name: str, cu_number: Any) -> List[Dict[str, Any]]:
        """
        Execute a direct query against a specific table.
        This is a more robust fallback method when standard queries fail.
        
        Args:
            table_name: Name of the table to query
            cu_number: Credit union number to search for
            
        Returns:
            List of matching records
        """
        try:
            # Always normalize the cu_number to handle various formats
            normalized_cu_number = self.normalize_cu_number(cu_number)
            logger.info(f"Attempting direct query on table {table_name} for cu_number {normalized_cu_number}")
            
            # Skip if table doesn't exist
            try:
                test_query = self.client.table(table_name).select("count").limit(1).execute()
                if not test_query:
                    logger.warning(f"Table {table_name} doesn't seem to exist or accessible")
                    return []
            except Exception as e:
                logger.warning(f"Table {table_name} access test failed: {str(e)}")
                return []
            
            # Try querying with normalized cu_number directly
            try:
                response = self.client.table(table_name).select("*").eq("cu_number", normalized_cu_number).execute()
                if response and hasattr(response, 'data') and response.data:
                    logger.info(f"Direct query found {len(response.data)} records in {table_name}")
                    return response.data
            except Exception as e:
                logger.warning(f"Direct integer query on {table_name} failed: {str(e)}")
            
            # Try with string version
            try:
                str_response = self.client.table(table_name).select("*").eq("cu_number", str(normalized_cu_number)).execute()
                if str_response and hasattr(str_response, 'data') and str_response.data:
                    logger.info(f"Direct string query found {len(str_response.data)} records in {table_name}")
                    return str_response.data
            except Exception as e:
                logger.warning(f"Direct string query on {table_name} failed: {str(e)}")
            
            # Try alternate column names
            alt_column_names = ["cu_num", "CU_NUMBER", "CU_NUM", "credit_union_number", "id"]
            for alt_column in alt_column_names:
                try:
                    alt_response = self.client.table(table_name).select("*").eq(alt_column, normalized_cu_number).execute()
                    if alt_response and hasattr(alt_response, 'data') and alt_response.data:
                        logger.info(f"Found {len(alt_response.data)} records in {table_name} using column {alt_column}")
                        return alt_response.data
                except Exception:
                    # If this column doesn't work, just continue to the next one
                    continue
            
            logger.info(f"No records found with direct query in table {table_name}")
            return []
            
        except Exception as e:
            logger.error(f"Error in direct table query for {table_name}: {str(e)}")
            return []
    
    def get_account_descriptions(self) -> Dict[str, Any]:
        """
        Retrieve all account descriptions from the acctdesc_2024_12 table.
        
        Returns:
            Dictionary mapping account codes to account names
        """
        if self.account_descriptions:
            return self.account_descriptions
            
        account_descriptions = {}
        
        try:
            logger.info("Retrieving account descriptions from acctdesc_2024_12 table...")
            
            # First, get the total count of records
            count_response = self.client.table("acctdesc_2024_12").select("count", count="exact").execute()
            total_count = 0
            if count_response and hasattr(count_response, 'count'):
                total_count = count_response.count
                logger.info(f"Total account descriptions in database: {total_count}")
            
            # Use paging to retrieve all records, with a page size of 1000
            page_size = 1000
            total_retrieved = 0
            
            # Fetch all pages of account descriptions
            for offset in range(0, total_count, page_size):
                logger.info(f"Fetching account descriptions (offset: {offset}, limit: {page_size})...")
                
                # Use pagination parameters to retrieve records in batches
                # Also retrieve the tablename column if it exists
                response = self.client.table("acctdesc_2024_12").select("account,acctname,tablename").range(offset, offset + page_size - 1).execute()
                
                if response and hasattr(response, 'data') and response.data:
                    batch_size = len(response.data)
                    total_retrieved += batch_size
                    logger.info(f"Retrieved {batch_size} records (total: {total_retrieved}/{total_count})")
                    
                    # Debug the first record of the first batch
                    if offset == 0 and len(response.data) > 0 and self.verbose:
                        sample_record = response.data[0]
                        logger.info(f"Sample account record: {sample_record}")
                    
                    for item in response.data:
                        acct_code = item.get('account')
                        acct_name = item.get('acctname')
                        tablename = item.get('tablename')
                        
                        # Skip if either is None
                        if acct_code is None or acct_name is None:
                            continue
                            
                        # Always convert to string to handle numeric codes
                        acct_code_str = str(acct_code)
                        
                        # Create a more detailed entry with tablename
                        entry = {
                            'name': acct_name,
                            'tablename': tablename
                        }
                        
                        # Store the original code
                        account_descriptions[acct_code_str] = entry
                        
                        # Also store with just the numeric part, without the "Acct_" prefix
                        if acct_code_str.lower().startswith('acct_'):
                            code_num = acct_code_str.split('_')[1] if len(acct_code_str.split('_')) > 1 else None
                            if code_num:
                                account_descriptions[code_num] = entry
                                # Also without leading zeros
                                if code_num.startswith('0'):
                                    account_descriptions[code_num.lstrip('0')] = entry
                        
                        # Also store without leading zeros for easier matching
                        if acct_code_str.startswith('0'):
                            account_descriptions[acct_code_str.lstrip('0')] = entry
                else:
                    logger.warning(f"No more account descriptions found at offset {offset}")
                    break
            
            # Print summary of loaded descriptions
            logger.info(f"Loaded {len(account_descriptions)} account descriptions (from {total_retrieved} records)")
            
            # Print some stats about the types of account codes if verbose
            if self.verbose:
                types_count = {'starts_with_acct': 0, 'numeric_only': 0, 'other': 0}
                for code in account_descriptions.keys():
                    if code.lower().startswith('acct_'):
                        types_count['starts_with_acct'] += 1
                    elif code.isdigit():
                        types_count['numeric_only'] += 1
                    else:
                        types_count['other'] += 1
                
                logger.info(f"Account code types: {types_count}")
                
        except Exception as e:
            logger.warning(f"Error retrieving account descriptions: {str(e)}")
            if self.verbose:
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
        
        self.account_descriptions = account_descriptions
        return account_descriptions
    
    def query_all_tables(self, cu_number: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Query all tables for records matching the cu_number.
        
        Args:
            cu_number: Credit union number to search for
            
        Returns:
            Dictionary mapping table names to lists of matching records
        """
        # Normalize the cu_number to handle '.0' suffixes from vector search
        normalized_cu_number = self.normalize_cu_number(cu_number)
        logger.info(f"Normalized cu_number from {cu_number} to {normalized_cu_number}")
        
        # Load account descriptions
        self.get_account_descriptions()
        
        # Query each table
        results = {}
        logger.info(f"Querying {len(self.all_tables)} tables for cu_number {normalized_cu_number}")
        
        for table_name in self.all_tables:
            records = self.query_table(table_name, normalized_cu_number)
            if records:  # Only include tables with matching records
                results[table_name] = records
        
        # If no results found, try the direct query approach for all tables
        if not results:
            logger.warning(f"No results found with standard queries. Trying direct query fallback...")
            
            for table_name in self.all_tables:
                direct_records = self.query_table_directly(table_name, normalized_cu_number)
                if direct_records:
                    logger.info(f"Direct query found {len(direct_records)} records in {table_name}")
                    results[table_name] = direct_records
        
        # If still no results, try a tablename-based approach
        if not results:
            logger.warning(f"No results found with direct queries. Trying tablename-based approach...")
            
            # Get unique tablenames from account descriptions
            tablenames = set()
            for entry in self.account_descriptions.values():
                if isinstance(entry, dict) and entry.get('tablename'):
                    tablenames.add(entry['tablename'])
            
            logger.info(f"Found {len(tablenames)} unique tablenames in account descriptions")
            
            # Try direct queries based on the tablenames
            for tablename in tablenames:
                # Map the tablename (e.g., "FS220A") to the actual table (e.g., "fs220a_2024_12")
                if tablename in self.tablename_to_table:
                    table_to_query = self.tablename_to_table[tablename]
                    logger.info(f"Trying direct query on {table_to_query} based on tablename {tablename}")
                    
                    # Try a direct query
                    records = self.query_table_directly(table_to_query, normalized_cu_number)
                    if records:
                        logger.info(f"Found {len(records)} records in {table_to_query} via tablename lookup")
                        results[table_to_query] = records
        
        logger.info(f"Found matching records in {len(results)} tables")
        return results
    
    def format_results_for_display(self, cu_number: str, results: Dict[str, List[Dict[str, Any]]]) -> str:
        """
        Format the results as a readable string for display.
        
        Args:
            cu_number: The credit union number that was searched for
            results: Dictionary mapping table names to lists of matching records
            
        Returns:
            Formatted string for display
        """
        if not results:
            return f"No data found for credit union #{cu_number}"
        
        # Build a readable string output
        output = []
        output.append(f"RESULTS FOR CREDIT UNION #{cu_number}")
        output.append(f"Found data in {len(results)} tables out of {len(self.all_tables)} total tables")
        output.append("")
        
        # Summary of tables
        output.append("SUMMARY OF TABLES WITH DATA:")
        output.append("---------------------------")
        for table_name, records in results.items():
            output.append(f"{table_name}: {len(records)} records")
        output.append("")
        
        # Create lowercase versions of account descriptions keys for case-insensitive matching
        lowercase_account_map = {}
        for acct_code, entry in self.account_descriptions.items():
            if isinstance(entry, dict):
                lowercase_account_map[acct_code.lower()] = entry.get('name')
            else:
                # Handle string values for backward compatibility
                lowercase_account_map[acct_code.lower()] = entry
        
        # Detailed data for each table
        output.append("DETAILED DATA BY TABLE:")
        output.append("======================")
        
        for table_name, records in results.items():
            output.append(f"\nTABLE: {table_name}")
            output.append("=" * (len(table_name) + 7))
            
            if records:
                for record_idx, record in enumerate(records):
                    if record_idx > 0:
                        output.append("\n---\n")  # Separator between multiple records
                    
                    # Display each column and its value
                    for col in sorted(record.keys()):
                        # Skip showing cu_number as it's redundant
                        if col.lower() == "cu_number":
                            continue
                            
                        value = record.get(col)
                        
                        # Format value for display
                        if value is None:
                            formatted_value = "NULL"
                        elif isinstance(value, (dict, list)):
                            formatted_value = json.dumps(value, indent=2)
                        else:
                            formatted_value = str(value)
                        
                        # Check if column name matches an account code pattern (case-insensitive)
                        display_col = col
                        col_lower = col.lower()
                        if col_lower.startswith('acct_'):
                            # Extract the numeric part
                            code_part = col_lower.split('_')[1] if len(col_lower.split('_')) > 1 else None
                            if code_part:
                                # Try direct match
                                if code_part in lowercase_account_map:
                                    acct_name = lowercase_account_map[code_part]
                                    display_col = f"{col} ({acct_name})"
                                # Try without leading zeros
                                elif code_part.lstrip('0') in lowercase_account_map:
                                    acct_name = lowercase_account_map[code_part.lstrip('0')]
                                    display_col = f"{col} ({acct_name})"
                                # Try with Acct_ prefix
                                elif f"acct_{code_part}".lower() in lowercase_account_map:
                                    acct_name = lowercase_account_map[f"acct_{code_part}".lower()]
                                    display_col = f"{col} ({acct_name})"
                        
                        output.append(f"{display_col}: {formatted_value}")
            else:
                output.append("No data to display")
        
        return "\n".join(output)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Query all Supabase tables for a specific cu_number')
    parser.add_argument('cu_number', type=str, help='Credit union number to search for')
    parser.add_argument('--output', type=str, help='Output file for results')
    parser.add_argument('--format', choices=['json', 'readable'], default='json',
                       help='Output format: json (default) or readable text')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    try:
        # Set logging level based on verbosity
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Verbose logging enabled")
        
        # Initialize querier
        querier = SupabaseAllTablesQuerier(verbose=args.verbose)
        
        # Query all tables
        results = querier.query_all_tables(args.cu_number)
        
        # Prepare output based on format
        if args.format == 'readable':
            # Format results as readable text
            formatted_output = querier.format_results_for_display(args.cu_number, results)
            
            if args.output:
                # Write to file
                with open(args.output, 'w') as f:
                    f.write(formatted_output)
                print(f"Readable results written to {args.output}")
            else:
                # Print to console
                print(formatted_output)
        else:
            # Default JSON format
            output = {
                "cu_number": args.cu_number,
                "total_tables_with_data": len(results),
                "total_tables_checked": len(querier.all_tables),
                "results": results
            }
            
            # Output results
            if args.output:
                # Write to file
                with open(args.output, 'w') as f:
                    json.dump(output, f, indent=2)
                print(f"Results written to {args.output}")
            else:
                # Print to console (formatted JSON)
                print(json.dumps(output, indent=2))
            
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 