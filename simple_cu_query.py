#!/usr/bin/env python
"""
Simple script to query all Supabase tables for a specific credit union number.
This version uses a direct, synchronous approach to minimize potential issues.
"""

import os
import sys
import json
from dotenv import load_dotenv
from supabase import create_client
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress
from rich.text import Text

# Set up Rich console for pretty output
console = Console()

def main():
    console.print(Panel.fit(
        "[bold blue]NCUA Credit Union Database Query Tool[/bold blue]\n"
        "This tool queries all tables in the database for a specific credit union number.",
        title="Welcome"
    ))
    
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment variables
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_API_KEY")
    
    if not supabase_url or not supabase_key:
        console.print("[bold red]Error:[/bold red] SUPABASE_URL and SUPABASE_API_KEY environment variables must be set")
        sys.exit(1)
    
    # Initialize Supabase client
    console.print("[bold green]Connecting to Supabase...[/bold green]")
    client = create_client(supabase_url, supabase_key)
    
    # List of tables to query
    tables = [
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
    
    # Possible column names for credit union number
    id_columns = [
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
    
    # Store which column to use for each table
    table_columns = {}
    
    # Main query loop
    while True:
        # Get cu_number from user
        cu_number_input = Prompt.ask("\nEnter a credit union number to search for (or 'exit' to quit)")
        
        if cu_number_input.lower() in ['exit', 'quit', 'q']:
            break
        
        # Convert cu_number to integer if possible
        try:
            cu_number = int(float(cu_number_input))
            console.print(f"[dim]Using credit union number: {cu_number}[/dim]")
        except (ValueError, TypeError):
            cu_number = cu_number_input
            console.print("[yellow]Warning: Could not convert to integer. Using as-is.[/yellow]")
        
        # Query all tables
        results = {}
        
        with Progress() as progress:
            task = progress.add_task(f"[cyan]Querying tables for credit union #{cu_number}...", total=len(tables))
            
            for table_name in tables:
                progress.update(task, description=f"[cyan]Querying {table_name}...")
                
                # If we already know which column to use for this table
                if table_name in table_columns:
                    column = table_columns[table_name]
                    try:
                        response = client.table(table_name).select("*").eq(column, cu_number).execute()
                        if response.data:
                            results[table_name] = response.data
                    except Exception:
                        # If the known column fails, we'll try others
                        pass
                
                # If we don't know the column or the known one failed, try all possibilities
                if table_name not in results:
                    for column in id_columns:
                        try:
                            response = client.table(table_name).select("*").eq(column, cu_number).execute()
                            if response.data:
                                # Remember this column for future queries
                                table_columns[table_name] = column
                                results[table_name] = response.data
                                break
                        except Exception:
                            # Try the next column
                            continue
                
                progress.update(task, advance=1)
        
        # Display results
        if not results:
            console.print(f"[bold yellow]No data found for credit union #{cu_number}[/bold yellow]")
            continue
        
        # Show summary
        console.print(f"\n[bold green]Results for Credit Union #{cu_number}[/bold green]")
        console.print(f"Found data in [bold]{len(results)}[/bold] tables\n")
        
        # Display table summary
        for table_name, records in results.items():
            console.print(f"[cyan]{table_name}:[/cyan] {len(records)} records")
        
        # Ask which table to view
        table_choice = Prompt.ask(
            "\nWhich table would you like to see in detail?", 
            choices=list(results.keys()) + ["all", "none"],
            default="none"
        )
        
        if table_choice == "none":
            continue
        
        # Show chosen tables
        tables_to_show = list(results.keys()) if table_choice == "all" else [table_choice]
        
        for table_name in tables_to_show:
            records = results[table_name]
            
            console.print(f"\n[bold cyan]Table: {table_name}[/bold cyan]")
            console.print("=" * (len(table_name) + 8))
            
            # Display records
            for record_idx, record in enumerate(records):
                if record_idx > 0:
                    console.print("\n[dim]---[/dim]\n")
                
                # Common ID column names to skip in display
                id_column_values = ["cu_number", "id", "cu_num", "credit_union_number", "credit_union_id"]
                
                # Display each column:value pair
                for col in sorted(record.keys()):
                    # Skip ID columns
                    if col.lower() in [c.lower() for c in id_column_values]:
                        continue
                    
                    value = record.get(col)
                    
                    # Format value
                    if value is None:
                        formatted_value = "[dim]NULL[/dim]"
                    elif isinstance(value, (dict, list)):
                        formatted_value = json.dumps(value, indent=2)
                    else:
                        formatted_value = str(value)
                    
                    # Format and display
                    label = Text(f"{col}: ", style="bold yellow")
                    
                    if value is None:
                        label.append(formatted_value)
                    elif isinstance(value, (int, float)) and value != 0:
                        label.append(formatted_value, style="green")
                    elif value == 0:
                        label.append(formatted_value, style="dim")
                    else:
                        label.append(formatted_value)
                    
                    console.print(label)
        
        # Ask about export
        export_choice = Prompt.ask(
            "\nWould you like to export these results to a JSON file?",
            choices=["yes", "no"],
            default="no"
        )
        
        if export_choice == "yes":
            filename = f"cu_{cu_number}_export.json"
            
            # Format data for export
            export_data = {
                "cu_number": cu_number,
                "total_tables_with_data": len(results),
                "results": results
            }
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            console.print(f"[green]Results exported to {filename}[/green]")
    
    console.print("[bold green]Thank you for using the Credit Union Data Query Tool![/bold green]")

if __name__ == "__main__":
    main() 