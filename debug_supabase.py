#!/usr/bin/env python
"""
Debug script to test Supabase connection and list available tables.
This version uses a direct PostgreSQL query to discover all tables in the database.
"""

import os
import asyncio
import json
import requests
from dotenv import load_dotenv
from supabase import create_client
from rich.console import Console
from rich.table import Table

# Console for pretty output
console = Console()

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get API credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")
    
    if not supabase_url or not supabase_key:
        console.print("[bold red]Error:[/bold red] SUPABASE_URL and SUPABASE_API_KEY environment variables must be set")
        return
    
    console.print("[bold green]Connecting to Supabase...[/bold green]")
    client = create_client(supabase_url, supabase_key)
    
    # Try to get tables using direct REST API query to PostgreSQL information_schema
    console.print("[bold cyan]Attempting to discover all tables using direct REST API...[/bold cyan]")
    
    # Remove trailing slash if it exists
    if supabase_url.endswith('/'):
        supabase_url = supabase_url[:-1]
    
    # Construct REST API URL for querying tables
    rest_api_url = f"{supabase_url}/rest/v1/"
    
    # Try to use the REST API to just list tables
    try:
        # For this approach, we'll list all available tables by making a GET request
        # to the base REST API endpoint
        headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}"
        }
        
        console.print("[bold]Method 1: Trying to list tables via REST API base endpoint...[/bold]")
        response = requests.get(rest_api_url, headers=headers)
        
        if response.status_code == 200:
            # Extract table names from the response (usually in the form of a JSON object)
            try:
                tables_data = response.json()
                # Check if we got a dict (typical Supabase REST API response)
                if isinstance(tables_data, dict):
                    all_tables = list(tables_data.keys())
                    console.print(f"[green]Success! Found {len(all_tables)} tables[/green]")
                    console.print(f"Tables: {', '.join(all_tables)}")
                else:
                    console.print("[yellow]Response was not in expected format[/yellow]")
                    console.print(f"Response: {tables_data}")
            except:
                console.print("[yellow]Could not parse response as JSON[/yellow]")
                console.print(f"Response: {response.text[:500]}")
        else:
            console.print(f"[red]Failed with status code {response.status_code}[/red]")
            console.print(f"Response: {response.text[:500]}")
    except Exception as e:
        console.print(f"[red]Error accessing REST API: {str(e)}[/red]")
    
    # Try an alternative approach using Supabase client
    try:
        console.print("\n[bold]Method 2: Trying to access a few common table names directly...[/bold]")
        
        # List of potential common table names to check
        common_tables = [
            "users", "profiles", "auth", "public", "customers", "orders", 
            "products", "data", "entries", "records", "items", "financial_data",
            "financial", "credit_unions", "cu_data", "banks", "banking",
            "institutions", "finance", "accounts", "credits", "loans"
        ]
        
        found_tables = []
        
        for table_name in common_tables:
            try:
                # Try to query the table with a limit of 1 to see if it exists
                response = await client.table(table_name).select("*").limit(1).execute()
                # If we get here, table exists
                found_tables.append(table_name)
                console.print(f"[green]✓ Table found:[/green] {table_name}")
                
                # Get a sample of columns
                if response.data:
                    sample_row = response.data[0]
                    columns = list(sample_row.keys())
                    console.print(f"  [blue]Sample columns:[/blue] {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
                else:
                    console.print(f"  [yellow]Table is empty[/yellow]")
            except Exception as e:
                # Table doesn't exist or can't be accessed
                pass
        
        if not found_tables:
            console.print("[yellow]No common tables found[/yellow]")
        else:
            console.print(f"[green]Found {len(found_tables)} tables[/green]")
    except Exception as e:
        console.print(f"[red]Error checking common tables: {str(e)}[/red]")
    
    # Try a third approach - direct SQL query
    console.print("\n[bold]Method 3: Trying SQL query via RPC if available...[/bold]")
    try:
        # Try to use RPC to run a SQL query to get all tables
        sql_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name;
        """
        
        try:
            # This might fail if the RPC function doesn't exist
            response = await client.rpc('execute_sql', {'query': sql_query}).execute()
            if response.data:
                tables = [row.get('table_name') for row in response.data]
                console.print(f"[green]Success! Found {len(tables)} tables using RPC[/green]")
                console.print(f"Tables: {', '.join(tables)}")
        except Exception as e:
            console.print(f"[yellow]execute_sql RPC failed: {str(e)}[/yellow]")
            
            # Try another common RPC function name
            try:
                response = await client.rpc('run_query', {'query': sql_query}).execute()
                if response.data:
                    tables = [row.get('table_name') for row in response.data]
                    console.print(f"[green]Success! Found {len(tables)} tables using run_query RPC[/green]")
                    console.print(f"Tables: {', '.join(tables)}")
            except Exception as e:
                console.print(f"[yellow]run_query RPC failed: {str(e)}[/yellow]")
    except Exception as e:
        console.print(f"[red]Error with RPC approach: {str(e)}[/red]")
    
    # Manual input option
    console.print("\n[bold yellow]If none of the automatic methods worked, you can manually enter table names:[/bold yellow]")
    use_manual = input("Would you like to manually enter a table name to test? (y/n): ").lower().strip() == 'y'
    
    if use_manual:
        table_name = input("Enter a table name to test: ").strip()
        
        try:
            console.print(f"[bold]Testing access to table '{table_name}'...[/bold]")
            response = await client.table(table_name).select("*").limit(5).execute()
            
            if response.data:
                console.print(f"[green]Success! Table '{table_name}' exists with {len(response.data)} records[/green]")
                
                # Print first record as a sample
                console.print("[bold]Sample record:[/bold]")
                for key, value in response.data[0].items():
                    console.print(f"  [yellow]{key}:[/yellow] {value}")
                
                # Check if table has cu_number column
                try:
                    test_response = await client.table(table_name).select("*").eq("cu_number", 99999).limit(1).execute()
                    console.print(f"[green]✓ Table has cu_number column[/green]")
                except Exception as e:
                    error_message = str(e)
                    if "column" in error_message.lower() and "does not exist" in error_message.lower():
                        console.print(f"[yellow]✗ Table does not have cu_number column[/yellow]")
                    else:
                        console.print(f"[yellow]? Could not determine if cu_number exists: {error_message}[/yellow]")
            else:
                console.print(f"[yellow]Table '{table_name}' exists but is empty[/yellow]")
        except Exception as e:
            console.print(f"[red]Could not access table '{table_name}': {str(e)}[/red]")
    
    console.print("\n[bold green]Debug complete![/bold green]")
    console.print("[yellow]Note: If no tables were found automatically, you may need to:[/yellow]")
    console.print("1. Check that your API key has the correct permissions")
    console.print("2. Verify that your Supabase URL is correct")
    console.print("3. Confirm that the database actually has tables created")

if __name__ == "__main__":
    asyncio.run(main()) 