#!/usr/bin/env python
"""
Interactive script to query all tables in Supabase based on a credit union number (cu_number).
This version presents a user-friendly interface for entering the cu_number and viewing results.
"""

import os
import sys
import json
import logging
import asyncio
import time
import argparse
from typing import Dict, List, Any, Optional, Set
from dotenv import load_dotenv
from supabase import create_client
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.progress import Progress

# Configure logging (less verbose for interactive use)
logging.basicConfig(
    level=logging.WARNING,  # Set back to WARNING for less verbosity
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up Rich console for pretty output
console = Console()

# Add command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Query NCUA credit union data from Supabase')
    parser.add_argument('cu_number', nargs='?', help='Credit union number to search for')
    parser.add_argument('--output', help='Output file path for JSON results')
    parser.add_argument('--format', choices=['json', 'readable'], default='json', help='Output format (default: json)')
    
    # Add account mapping functionality
    parser.add_argument('--account-mapping', action='store_true', help='Query account description table for mapping')
    parser.add_argument('--search-term', action='append', help='Term to search for in account descriptions')
    parser.add_argument('--status-filter', choices=['true', 'false', 'all'], default='all', 
                        help='Filter accounts by status (active/inactive)')
    
    # Add targeted query functionality
    parser.add_argument('--table', help='Specific table to query')
    parser.add_argument('--columns', help='Comma-separated list of columns to retrieve')
    parser.add_argument('--cu-number', dest='cu_number_arg', help='Credit union number for targeted query')
    parser.add_argument('--output-json', action='store_true', help='Output as JSON to stdout')
    
    return parser.parse_args()

class SupabaseInteractiveQuerier:
    """Query all tables in Supabase database based on a cu_number with an interactive interface."""
    
    def __init__(self):
        """Initialize the Supabase client."""
        # Load environment variables
        load_dotenv()
        
        # Get API credentials from environment variables
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_API_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            console.print("[bold red]Error:[/bold red] SUPABASE_URL and SUPABASE_API_KEY environment variables must be set")
            sys.exit(1)
        
        # Initialize Supabase client
        with console.status("[bold green]Initializing connection to Supabase...[/bold green]"):
            self.client = create_client(self.supabase_url, self.supabase_key)
        
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
        
        # Will store which tables can be queried
        self.queryable_tables = set()
        
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
        
        # Always use fallback mode initially to try all tables
        self.fallback_mode = True
        
        # Test a sample table to verify if we have read access
        self.has_read_access = False
    
    def check_read_access(self):
        """
        Check if we have read access to any tables in the database.
        """
        # Try to read a sample table
        try:
            # Skip the test tables as they're most likely to exist
            sample_tables = ["fs220_2024_12", "credit_union_branch_information_2024_12"]
            
            for table_name in sample_tables:
                try:
                    # Try to read one row
                    response = self.client.table(table_name).select("*").limit(1).execute()
                    if response and hasattr(response, 'data'):
                        self.has_read_access = True
                        console.print(f"[green]Successfully verified read access to {table_name}[/green]")
                        return True
                except Exception:
                    # Try the next table
                    continue
            
            # If none of the sample tables worked
            console.print("[yellow]Warning: Could not verify read access to any sample tables[/yellow]")
            return False
            
        except Exception as e:
            console.print(f"[red]Error checking read access: {str(e)}[/red]")
            return False
    
    def check_table_existence(self, table_name):
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
    
    def discover_queryable_tables(self):
        """
        Find out which tables from our known list can be queried.
        
        Returns:
            Set of table names that can be queried
        """
        # Start with the verification
        self.check_read_access()
        
        # Always use fallback mode to try all tables
        self.queryable_tables = set(self.all_tables)
        self.fallback_mode = True
        return self.queryable_tables
    
    def normalize_cu_number(self, cu_number):
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
            console.print(f"[yellow]Warning: Could not normalize cu_number {cu_number}: {e}[/yellow]")
            # Return the original value if conversion fails
            return cu_number
            
        # Return the original value if none of the above conversions worked
        return cu_number
    
    def try_query_with_column(self, table_name, column_name, cu_number):
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
    
    def query_table_directly(self, table_name, cu_number):
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
            
            # Try multiple ways to query for the cu_number
            query_attempts = [
                # Try exact match with the normalized number
                f"SELECT * FROM {table_name} WHERE cu_number = {normalized_cu_number}",
                # Try with string version
                f"SELECT * FROM {table_name} WHERE cu_number = '{normalized_cu_number}'",
                # Try with CAST to handle type differences
                f"SELECT * FROM {table_name} WHERE CAST(cu_number AS TEXT) = '{normalized_cu_number}'"
            ]
            
            for query in query_attempts:
                try:
                    # Execute the raw SQL query
                    response = self.client.table(table_name).select("*").eq("cu_number", normalized_cu_number).execute()
                    
                    if response and hasattr(response, 'data') and response.data:
                        console.print(f"[green]Successfully queried {table_name} with direct approach[/green]")
                        return response.data
                except Exception as inner_e:
                    # This query attempt failed, try the next one
                    continue
            
            # If we get here, all query attempts failed
            return []
            
        except Exception as e:
            console.print(f"[yellow]Error in direct query for {table_name}: {str(e)}[/yellow]")
            return []
    
    def get_account_descriptions(self):
        """
        Retrieve all account descriptions from the acctdesc_2024_12 table.
        
        Returns:
            Dictionary mapping account codes to account names
        """
        account_descriptions = {}
        
        try:
            console.print("[dim]Retrieving account descriptions from database...[/dim]")
            
            # First, get the total count of records to know how many we need to fetch
            count_response = self.client.table("acctdesc_2024_12").select("count", count="exact").execute()
            total_count = 0
            if count_response and hasattr(count_response, 'count'):
                total_count = count_response.count
                console.print(f"[dim]Total account descriptions in database: {total_count}[/dim]")
            
            # Use paging to retrieve all records, with a page size of 1000
            page_size = 1000
            total_retrieved = 0
            
            # Fetch all pages of account descriptions
            for offset in range(0, total_count, page_size):
                console.print(f"[dim]Fetching account descriptions (offset: {offset}, limit: {page_size})...[/dim]")
                
                # Use pagination parameters to retrieve records in batches
                # Include status field to filter active/inactive accounts
                response = self.client.table("acctdesc_2024_12").select("account,acctname,tablename,status").range(offset, offset + page_size - 1).execute()
                
                if response and hasattr(response, 'data') and response.data:
                    batch_size = len(response.data)
                    total_retrieved += batch_size
                    console.print(f"[dim]Retrieved {batch_size} records (total: {total_retrieved}/{total_count})[/dim]")
                    
                    # Debug the first record of the first batch
                    if offset == 0 and len(response.data) > 0:
                        sample_record = response.data[0]
                        console.print(f"[dim]Sample account record: {sample_record}[/dim]")
                    
                    for item in response.data:
                        acct_code = item.get('account')
                        acct_name = item.get('acctname')
                        # Also get the tablename for more precise querying later
                        tablename = item.get('tablename')
                        # Get status to filter active/inactive accounts
                        status = item.get('status')
                        
                        # Skip if account or acctname is None
                        if acct_code is None or acct_name is None:
                            continue
                        
                        # Always convert to string to handle numeric codes
                        acct_code_str = str(acct_code)
                        
                        # Create a more detailed entry with tablename and status
                        entry = {
                            'name': acct_name,
                            'tablename': tablename,
                            'status': status
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
                    console.print(f"[yellow]No more account descriptions found at offset {offset}[/yellow]")
                    break
            
            # Print summary of loaded descriptions
            console.print(f"[green]Loaded {len(account_descriptions)} account descriptions (from {total_retrieved} records)[/green]")
            
            # Print some stats about the types of account codes and status
            types_count = {'starts_with_acct': 0, 'numeric_only': 0, 'other': 0}
            status_count = {'true': 0, 'false': 0, 'unknown': 0}
            
            for code, entry in account_descriptions.items():
                # Count code types
                if code.lower().startswith('acct_'):
                    types_count['starts_with_acct'] += 1
                elif code.isdigit():
                    types_count['numeric_only'] += 1
                else:
                    types_count['other'] += 1
                
                # Count status types
                if isinstance(entry, dict) and 'status' in entry:
                    status_val = str(entry['status']).lower()
                    if status_val == 'true':
                        status_count['true'] += 1
                    elif status_val == 'false':
                        status_count['false'] += 1
                    else:
                        status_count['unknown'] += 1
                else:
                    status_count['unknown'] += 1
            
            console.print(f"[dim]Account code types: {types_count}[/dim]")
            console.print(f"[dim]Account status counts: {status_count}[/dim]")
            
        except Exception as e:
            console.print(f"[yellow]Error retrieving account descriptions: {str(e)}[/yellow]")
            # Print the full exception traceback for debugging
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        
        return account_descriptions
    
    def query_table(self, table_name, cu_number):
        """
        Query a specific table for records matching the cu_number.
        
        Args:
            table_name: Name of the table to query
            cu_number: Credit union number to search for
            
        Returns:
            List of matching records
        """
        # Normalize the cu_number to handle '.0' suffixes from vector search
        normalized_cu_number = self.normalize_cu_number(cu_number)
        
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
                return records
        
        # If the standard approach fails, try a direct query as fallback
        console.print(f"[yellow]Standard query methods failed for {table_name}, trying direct query...[/yellow]")
        records = self.query_table_directly(table_name, normalized_cu_number)
        if records:
            return records
            
        # If none of the approaches worked, return empty list
        return []
    
    def query_all_tables(self, cu_number):
        """
        Query all tables for records matching the cu_number.
        
        Args:
            cu_number: Credit union number to search for
            
        Returns:
            Dictionary mapping table names to lists of matching records
        """
        # Normalize the cu_number to handle '.0' suffixes from vector search
        normalized_cu_number = self.normalize_cu_number(cu_number)
        console.print(f"[dim]Normalized cu_number from {cu_number} to {normalized_cu_number}[/dim]")
        
        # Always try all tables
        self.fallback_mode = True
        self.queryable_tables = set(self.all_tables)
        
        # Query each table
        results = {}
        with console.status(f"[bold green]Querying {len(self.all_tables)} tables for credit union #{normalized_cu_number}...[/bold green]"):
            for table_name in self.all_tables:
                records = self.query_table(table_name, normalized_cu_number)
                if records:  # Only include tables with matching records
                    results[table_name] = records
        
        # If no results found, try one more thing: check acctdesc for the tablename and account fields
        if not results:
            console.print("[yellow]No results found with standard queries. Trying tablename lookup approach...[/yellow]")
            account_descriptions = self.get_account_descriptions()
            
            # Get unique tablenames from account descriptions
            tablenames = set()
            for entry in account_descriptions.values():
                if isinstance(entry, dict) and entry.get('tablename'):
                    tablenames.add(entry['tablename'])
            
            console.print(f"[dim]Found {len(tablenames)} unique tablenames in account descriptions[/dim]")
            
            # Try direct queries based on the tablenames
            for tablename in tablenames:
                # Map the tablename (e.g., "FS220A") to the actual table (e.g., "fs220a_2024_12")
                if tablename in self.tablename_to_table:
                    table_to_query = self.tablename_to_table[tablename]
                    console.print(f"[dim]Trying direct query on {table_to_query} based on tablename {tablename}[/dim]")
                    
                    # Try a direct query
                    records = self.query_table_directly(table_to_query, normalized_cu_number)
                    if records:
                        console.print(f"[green]Found {len(records)} records in {table_to_query} via tablename lookup[/green]")
                        results[table_to_query] = records
        
        return results
    
    def display_results(self, cu_number, results):
        """
        Display the query results in a user-friendly format.
        
        Args:
            cu_number: The credit union number that was searched for
            results: Dictionary mapping table names to lists of matching records
        """
        if not results:
            console.print(f"[bold yellow]No data found for credit union #{cu_number}[/bold yellow]")
            return
        
        console.print(f"\n[bold green]Results for Credit Union #{cu_number}[/bold green]")
        console.print(f"Found data in [bold]{len(results)}[/bold] tables\n")
        
        # Load account descriptions for enhancing the display
        account_descriptions = self.get_account_descriptions()
        
        # Display a summary of tables with data
        summary_table = Table(title="Summary of Tables with Data")
        summary_table.add_column("Table Name", style="cyan")
        summary_table.add_column("Record Count", style="green")
        
        for table_name, records in results.items():
            summary_table.add_row(table_name, str(len(records)))
        
        console.print(summary_table)
        
        # Ask user which table to view in detail
        table_choice = Prompt.ask(
            "\nWhich table would you like to see in detail?", 
            choices=list(results.keys()) + ["all", "none"],
            default="none"
        )
        
        if table_choice == "none":
            return
        
        # Show all tables or just the selected one
        tables_to_show = list(results.keys()) if table_choice == "all" else [table_choice]
        
        # Create lowercase versions of account descriptions keys for case-insensitive matching
        lowercase_account_map = {}
        for acct_code, entry in account_descriptions.items():
            if isinstance(entry, dict):
                lowercase_account_map[acct_code.lower()] = entry.get('name')
            else:
                # Handle string values for backward compatibility
                lowercase_account_map[acct_code.lower()] = entry
        
        for table_name in tables_to_show:
            records = results[table_name]
            
            console.print(f"\n[bold cyan]Table: {table_name}[/bold cyan]")
            console.print("=" * (len(table_name) + 8))
            
            # Display records as column:value pairs
            if records:
                for record_idx, record in enumerate(records):
                    if record_idx > 0:
                        console.print("\n[dim]---[/dim]\n")  # Separator between multiple records
                    
                    # Get columns from the record
                    columns = list(record.keys())
                    
                    # Common ID column names to skip in display
                    id_columns = ["cu_number", "id", "cu_num", "credit_union_number", "credit_union_id"]
                    
                    # Display each column and its value
                    for col in sorted(columns):
                        # Skip showing ID columns as they're redundant
                        if col.lower() in [c.lower() for c in id_columns]:
                            continue
                            
                        value = record.get(col)
                        
                        # Format value for display
                        if value is None:
                            formatted_value = "[dim]NULL[/dim]"
                        elif isinstance(value, (dict, list)):
                            formatted_value = json.dumps(value, indent=2)
                        else:
                            formatted_value = str(value)
                        
                        # Create a rich text object for better formatting
                        # Check if column name matches an account code pattern (e.g., acct_995 or Acct_995) - case insensitive
                        account_code_match = None
                        # First convert column name to lowercase for case-insensitive comparison
                        col_lower = col.lower()
                        if col_lower.startswith('acct_'):
                            # Extract the numeric part
                            code_part = col_lower.split('_')[1] if len(col_lower.split('_')) > 1 else None
                            if code_part:
                                # Try direct match
                                if code_part in lowercase_account_map:
                                    account_code_match = code_part
                                # Try without leading zeros
                                elif code_part.lstrip('0') in lowercase_account_map:
                                    account_code_match = code_part.lstrip('0')
                                # Try with Acct_ prefix
                                elif f"acct_{code_part}".lower() in lowercase_account_map:
                                    account_code_match = f"acct_{code_part}".lower()
                        
                        if account_code_match:
                            # Display both the column name and its description
                            acct_name = lowercase_account_map[account_code_match]
                            label = Text(f"{col} ({acct_name}): ", style="bold yellow")
                        else:
                            label = Text(f"{col}: ", style="bold yellow")
                        
                        # Add the value with appropriate styling
                        if value is None:
                            label.append(formatted_value)
                        elif isinstance(value, (int, float)) and value != 0:
                            label.append(formatted_value, style="green")
                        elif value == 0:
                            label.append(formatted_value, style="dim")
                        else:
                            label.append(formatted_value)
                        
                        console.print(label)
            else:
                console.print(f"[yellow]No data to display for table {table_name}[/yellow]")

    def export_results(self, cu_number, results):
        """
        Export the results to a JSON file.
        
        Args:
            cu_number: The credit union number that was searched for
            results: Dictionary mapping table names to lists of matching records
        """
        if not results:
            console.print("[yellow]No data to export[/yellow]")
            return
        
        # Ask if the user wants to export the results
        export_choice = Prompt.ask(
            "Would you like to export these results to a JSON file?",
            choices=["yes", "no"],
            default="no"
        )
        
        if export_choice == "yes":
            filename = f"cu_{cu_number}_export.json"
            
            # Format data for export
            export_data = {
                "cu_number": cu_number,
                "total_tables_with_data": len(results),
                "total_tables_checked": len(self.all_tables),
                "results": results
            }
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            console.print(f"[green]Results exported to {filename}[/green]")

    def search_account_descriptions(self, search_terms: List[str], status_filter: str = 'all') -> Dict[str, Dict[str, str]]:
        """
        Search account descriptions table for matching terms.
        
        Args:
            search_terms: List of terms to search for in account descriptions
            status_filter: Filter accounts by status ('true' for active, 'false' for inactive, 'all' for both)
            
        Returns:
            Dictionary mapping account codes to information about them
        """
        account_mappings = {}
        
        try:
            # Get all account descriptions first
            account_descriptions = self.get_account_descriptions()
            
            # Normalize search terms (lowercase for case-insensitive matching)
            normalized_terms = [term.lower() for term in search_terms]
            
            # Search through the descriptions for matches
            for acct_code, entry in account_descriptions.items():
                if isinstance(entry, dict):
                    acct_name = entry.get('name', '').lower()
                    tablename = entry.get('tablename', '')
                    
                    # Check status if we're filtering by it
                    if status_filter != 'all' and 'status' in entry:
                        entry_status = str(entry['status']).lower()
                        if entry_status != status_filter:
                            continue
                    
                    # Check if any search term is in the account name
                    if any(term in acct_name for term in normalized_terms):
                        account_mappings[acct_code] = {
                            'name': entry.get('name'),
                            'tablename': tablename,
                            'status': entry.get('status', None)
                        }
                else:
                    # Handle string values (backward compatibility)
                    acct_name = str(entry).lower()
                    
                    # Can't filter by status if entry is just a string
                    if status_filter != 'all':
                        continue
                    
                    # Check if any search term is in the account name
                    if any(term in acct_name for term in normalized_terms):
                        account_mappings[acct_code] = {
                            'name': entry,
                            'tablename': 'unknown',  # Can't determine tablename for this format
                            'status': None
                        }
                        
            console.print(f"[green]Found {len(account_mappings)} account codes matching search terms[/green]")
            if status_filter != 'all':
                console.print(f"[dim]Filtered to accounts with status={status_filter}[/dim]")
            
            # Print samples of what was found
            if account_mappings:
                console.print("[dim]Sample matches:[/dim]")
                sample_count = min(3, len(account_mappings))
                for i, (code, details) in enumerate(list(account_mappings.items())[:sample_count]):
                    console.print(f"[dim]  {code}: {details}[/dim]")
                    
            return account_mappings
            
        except Exception as e:
            console.print(f"[bold red]Error searching account descriptions: {str(e)}[/bold red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return {}
    
    def targeted_table_query(self, table_name: str, cu_number: str, columns: List[str]) -> List[Dict[str, Any]]:
        """
        Query a specific table for only selected columns.
        
        Args:
            table_name: Name of the table to query
            cu_number: Credit union number
            columns: List of column names to retrieve
            
        Returns:
            List of records with only the requested columns
        """
        try:
            # Normalize the cu_number
            normalized_cu_number = self.normalize_cu_number(cu_number)
            
            console.print(f"[green]Performing targeted query on {table_name} for CU #{normalized_cu_number}[/green]")
            console.print(f"[dim]Requesting columns: {', '.join(columns)}[/dim]")
            
            # Join columns into a comma-separated string for the select query
            columns_str = ",".join(columns)
            
            # Query the table for just these columns
            response = self.client.table(table_name).select(columns_str).eq("cu_number", normalized_cu_number).execute()
            
            if response and hasattr(response, 'data'):
                records = response.data
                console.print(f"[green]Found {len(records)} records[/green]")
                return records
            
            # If the standard query fails, try a direct query as fallback
            console.print(f"[yellow]Standard query failed, trying direct approach...[/yellow]")
            
            # Build a SQL query
            query = f"SELECT {columns_str} FROM {table_name} WHERE cu_number = {normalized_cu_number}"
            try:
                # Execute the raw SQL query
                response = self.client.table(table_name).select(columns_str).eq("cu_number", normalized_cu_number).execute()
                
                if response and hasattr(response, 'data'):
                    records = response.data
                    console.print(f"[green]Found {len(records)} records with direct query[/green]")
                    return records
            except Exception as inner_e:
                console.print(f"[yellow]Direct query failed: {str(inner_e)}[/yellow]")
            
            # If both approaches fail, return empty list
            return []
            
        except Exception as e:
            console.print(f"[bold red]Error in targeted query for {table_name}: {str(e)}[/bold red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return []

def main():
    console.print(Panel.fit(
        "[bold blue]NCUA Credit Union Database Query Tool[/bold blue]\n"
        "This tool queries all tables in the database for a specific credit union number.",
        title="Welcome"
    ))
    
    # Parse command-line arguments
    args = parse_args()
    
    try:
        # Initialize querier
        querier = SupabaseInteractiveQuerier()
        
        # Handle account mapping mode
        if args.account_mapping:
            if not args.search_term:
                console.print("[yellow]No search terms provided. Please specify at least one search term with --search-term[/yellow]")
                return
                
            console.print(f"[green]Searching account descriptions for terms: {args.search_term}[/green]")
            if args.status_filter != 'all':
                console.print(f"[green]Using status filter: {args.status_filter} (only {'active' if args.status_filter == 'true' else 'inactive'} accounts)[/green]")
                
            account_mappings = querier.search_account_descriptions(args.search_term, args.status_filter)
            
            # Output as JSON if requested
            if args.output_json:
                print(json.dumps(account_mappings))
            elif args.output:
                with open(args.output, 'w') as f:
                    json.dump(account_mappings, f, indent=2)
                console.print(f"[green]Account mappings exported to {args.output}[/green]")
            else:
                # Pretty print the results
                table = Table(title=f"Account Mappings for: {', '.join(args.search_term)}")
                table.add_column("Account Code", style="cyan")
                table.add_column("Name", style="green")
                table.add_column("Table", style="magenta")
                table.add_column("Status", style="yellow")
                
                for code, details in account_mappings.items():
                    name = details.get('name', '') if isinstance(details, dict) else str(details)
                    tablename = details.get('tablename', 'unknown') if isinstance(details, dict) else 'unknown'
                    status = details.get('status', 'unknown') if isinstance(details, dict) else 'unknown'
                    
                    table.add_row(code, name, tablename, str(status))
                
                console.print(table)
            
            return
            
        # Handle targeted table query mode
        if args.table and args.cu_number_arg and args.columns:
            console.print(f"[green]Performing targeted query on {args.table}[/green]")
            
            # Split columns string into a list
            columns = [col.strip() for col in args.columns.split(',')]
            
            # Execute the targeted query
            results = querier.targeted_table_query(args.table, args.cu_number_arg, columns)
            
            # Output as JSON to stdout if requested
            if args.output_json:
                print(json.dumps(results))
            elif args.output:
                with open(args.output, 'w') as f:
                    json.dump(results, f, indent=2)
                console.print(f"[green]Results exported to {args.output}[/green]")
            else:
                # Display results in a table
                if results:
                    table = Table(title=f"Results from {args.table} for CU #{args.cu_number_arg}")
                    
                    # Add columns based on first result
                    for column in results[0].keys():
                        table.add_column(column)
                    
                    # Add rows for each result
                    for result in results:
                        table.add_row(*[str(val) for val in result.values()])
                    
                    console.print(table)
                else:
                    console.print(f"[yellow]No results found for CU #{args.cu_number_arg} in table {args.table}[/yellow]")
            
            return
        
        # Handle standard query mode
        cu_number = args.cu_number
        
        if not cu_number:
            cu_number = Prompt.ask("\nEnter a credit union number to search for")
            
        # Query all tables
        results = querier.query_all_tables(cu_number)
        
        # Handle different output formats
        if args.output:
            # Format data for export
            export_data = {
                "cu_number": cu_number,
                "total_tables_with_data": len(results),
                "total_tables_checked": len(querier.all_tables),
                "results": results
            }
            
            # Write to file
            with open(args.output, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            console.print(f"[green]Results exported to {args.output}[/green]")
        elif args.format == 'readable':
            # Display results in readable format
            querier.display_results(cu_number, results)
        else:
            # Default JSON output to stdout
            export_data = {
                "cu_number": cu_number,
                "total_tables_with_data": len(results),
                "total_tables_checked": len(querier.all_tables),
                "results": results
            }
            print(json.dumps(export_data))
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 