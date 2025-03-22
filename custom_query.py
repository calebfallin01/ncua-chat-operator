#!/usr/bin/env python
"""
Simple script to query a specific table in Supabase for data related to a credit union number.
This script allows direct manual input of table names and cu_number values.
"""

import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from supabase import create_client
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Set up console for pretty output
console = Console()

async def query_table(client, table_name, cu_number):
    """
    Query a table for records matching the cu_number.
    
    Args:
        client: Supabase client
        table_name: Name of the table to query
        cu_number: Credit union number to search for
    
    Returns:
        List of matching records
    """
    try:
        console.print(f"[cyan]Querying table '{table_name}' for cu_number {cu_number}...[/cyan]")
        
        # Try to convert cu_number to integer (assuming it's stored as int)
        try:
            cu_number_int = int(float(cu_number))
            console.print(f"[dim]Converting cu_number to integer: {cu_number} → {cu_number_int}[/dim]")
            cu_number = cu_number_int
        except ValueError:
            console.print("[yellow]Warning: Could not convert cu_number to integer, using as-is[/yellow]")
        
        # Make the query
        response = await client.table(table_name).select("*").eq("cu_number", cu_number).execute()
        
        if not response.data:
            console.print(f"[yellow]No records found in '{table_name}' for cu_number {cu_number}[/yellow]")
            return []
        
        console.print(f"[green]Found {len(response.data)} records in '{table_name}'[/green]")
        return response.data
    
    except Exception as e:
        error_str = str(e)
        if "column" in error_str.lower() and "does not exist" in error_str.lower():
            console.print(f"[red]Error: Table '{table_name}' does not have a cu_number column[/red]")
        else:
            console.print(f"[red]Error querying table: {str(e)}[/red]")
        return []

def display_results(table_name, records):
    """
    Display the results in a user-friendly format.
    
    Args:
        table_name: Name of the table
        records: List of records from the table
    """
    if not records:
        return
    
    console.print(f"\n[bold green]Results from table '{table_name}':[/bold green]")
    
    for i, record in enumerate(records):
        if i > 0:
            console.print("\n[dim]---[/dim]")
        
        # Create a table for each record
        table = Table(show_header=False, expand=True)
        table.add_column("Field", style="yellow")
        table.add_column("Value")
        
        # Add all fields except cu_number (since we already know it)
        for field, value in sorted(record.items()):
            if field == "cu_number":
                continue
                
            # Format the value
            if value is None:
                formatted_value = "[dim]NULL[/dim]"
            elif isinstance(value, (dict, list)):
                formatted_value = json.dumps(value, indent=2)
            else:
                formatted_value = str(value)
            
            table.add_row(field, formatted_value)
        
        console.print(table)

async def main():
    # Show welcome message
    console.print(Panel.fit(
        "[bold blue]Credit Union Data Query Tool[/bold blue]\n"
        "This tool allows you to query a specific table for data related to a credit union number.",
        title="Welcome"
    ))
    
    # Load environment variables
    load_dotenv()
    
    # Get Supabase credentials
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")
    
    if not supabase_url or not supabase_key:
        console.print("[bold red]Error:[/bold red] SUPABASE_URL and SUPABASE_API_KEY environment variables must be set")
        return
    
    # Connect to Supabase
    try:
        console.print("[bold green]Connecting to Supabase...[/bold green]")
        client = create_client(supabase_url, supabase_key)
    except Exception as e:
        console.print(f"[bold red]Error connecting to Supabase:[/bold red] {str(e)}")
        return
    
    # Main query loop
    while True:
        # Get table name
        table_name = input("\nEnter table name to query (or 'exit' to quit): ").strip()
        
        if table_name.lower() in ['exit', 'quit', 'q']:
            break
        
        # Get cu_number
        cu_number = input("Enter credit union number to search for: ").strip()
        
        # Query the table
        records = await query_table(client, table_name, cu_number)
        
        # Display results
        display_results(table_name, records)
        
        # Ask if want to export
        if records and input("\nExport results to JSON? (y/n): ").lower().strip() == 'y':
            filename = f"{table_name}_cu{cu_number}.json"
            with open(filename, 'w') as f:
                json.dump(records, f, indent=2)
            console.print(f"[green]Results exported to {filename}[/green]")
    
    console.print("[bold green]Thank you for using the Credit Union Data Query Tool![/bold green]")

if __name__ == "__main__":
    asyncio.run(main()) 