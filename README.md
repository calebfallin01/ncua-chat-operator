# NCUA Credit Union Data Query System

A command-line system that answers questions about credit unions by querying structured and vector databases.

## Features

- Natural language query processing
- Vector search of credit union data using Pinecone
- Structured data retrieval from Supabase
- Intelligent response generation using OpenAI
- Smart handling of credit union name abbreviations and modifiers
- Interactive confirmation for low-confidence matches
- Complete metadata display from vector database
- Multi-field search across primary names, tradenames, and domains
- Enhanced scoring for better matching on alternative names
- Comprehensive search combining metadata and vector similarity

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your API keys

## Usage

### Main System

Run the main script from the command line:

```
python src/main.py
```

Enter your questions about credit unions when prompted.

### Vector Database Query Tool

For simpler similarity searches, use the vector database query tool:

```
# Basic usage with comprehensive search (searches all fields at once)
python query_credit_unions.py "Navy Federal Credit Union"

# Run in interactive mode
python query_credit_unions.py --interactive

# Specify a custom minimum similarity score
python query_credit_unions.py "Navy Federal" --min-score 0.8

# Show all results above the minimum score, not just the top match
python query_credit_unions.py "Navy Federal" --all-results

# Specify how many candidates to retrieve from the vector database
python query_credit_unions.py "Navy Federal" --top-k 20

# Use only vector similarity search (turn off comprehensive search)
python query_credit_unions.py "Navy Federal" --vector-only

# Disable confirmation prompts for low-scoring matches
python query_credit_unions.py "Navy Federal" --force-min-score
```

This tool allows you to:
- Search for credit unions by name, tradename, and domain
- Find similar credit unions in the vector database
- See all metadata fields and similarity scores
- Enter multiple queries in interactive mode

#### Comprehensive Search

The tool uses a powerful comprehensive search approach that:

1. **Searches Across All Fields Simultaneously**:
   - Primary Name (`cu_name`)
   - Trade Name (`tradename`)
   - Domain Name (`domain_root`)

2. **Uses Case-Insensitive Partial Matching**:
   - Finds matches where the query is contained in any field
   - Works with abbreviations like "PenFed" for "Pentagon Federal Credit Union"
   - Handles variations in casing and formatting

3. **Adjusts Scores Based on Match Quality**:
   - Primary name matches receive highest score boost
   - Tradename matches receive high score boost
   - Domain matches receive moderate score boost
   - All matches are sorted by adjusted score

This approach ensures the best match is found regardless of which field contains the match.

#### Match Types

Each result is labeled with the type of match found:

- **(PRIMARY_NAME MATCH)**: Matched on the credit union's official name
- **(TRADENAME MATCH)**: Matched on an alternative or marketing name
- **(DOMAIN MATCH)**: Matched on the domain name identifier
- **(UNKNOWN MATCH)**: Matched through other means

#### Result Display

The tool displays:
- The credit union name with match type label
- The field where the match was found (cu_name, tradename, domain_root)
- Adjusted similarity score
- Original vector similarity score (when different)
- All available metadata fields from the vector database

#### Result Filtering

By default, the tool:
- Only shows results with a similarity score >= 0.75 (75%)
- Only displays the top matching result
- Indicates when additional matches are available
- Retrieves 20 candidates from the vector database (customize with --top-k)

#### Low-Confidence Match Handling

When the highest text similarity score is below the minimum threshold (default 0.75):
- The tool will show the best available match
- Ask you to confirm if this is what you were looking for
- Display full details if you confirm the match
- Suggest refining your search if the match isn't what you wanted

## Project Structure

- `src/` - Source code
  - `clients/` - Database and API client wrappers
  - `models/` - Data models and schemas
  - `services/` - Business logic and query processing
  - `main.py` - Entry point
- `utils/` - Utility functions for the query tool
- `config/` - Configuration files
- `tests/` - Test suite

## License

MIT 

# NCUA Credit Union Database Query Tool

This tool allows you to query all tables in your Supabase database for data related to a specific credit union number (`cu_number`).

## Features

- Query all tables in the database for a specific credit union number
- Configured with all 24 NCUA credit union database tables 
- Interactive mode with user-friendly display of results
- Command-line mode for scripting and automation
- Export results to JSON files
- Human-readable column:value format for easy data review
- Color-coded output in interactive mode

## Prerequisites

- Python 3.7+
- Supabase account with the database set up
- Environment variables for Supabase connection

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables (create a `.env` file in the project root):
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_API_KEY=your_supabase_api_key
   ```

## Usage

### Interactive Mode

For a user-friendly interface with pretty output and interactive features:

```
python interactive_query.py
```

This will:
1. Check which tables have a cu_number column to query
2. Prompt you to enter a credit union number
3. Query all tables for that credit union number
4. Display a summary of tables with matching data
5. Allow you to view specific tables in detail with column:value pairs
6. Give you the option to export results to a JSON file

The interactive mode displays data in a user-friendly format with column names and their corresponding values, making it easy to read and analyze the data.

### Command-line Mode

For scripting or automation:

```
python query_all_tables.py <cu_number> [--output <output_file>] [--format json|readable] [--verbose]
```

Arguments:
- `cu_number`: The credit union number to search for (required)
- `--output`: Optional file path for saving results
- `--format`: Output format - either `json` (default) or `readable` text
- `--verbose`: Enable verbose logging for detailed information

Examples:
```
# Output JSON to console
python query_all_tables.py 67890

# Output human-readable text to console
python query_all_tables.py 67890 --format readable

# Save JSON to file
python query_all_tables.py 67890 --output cu_67890_data.json

# Save human-readable text to file with verbose logging
python query_all_tables.py 67890 --format readable --output cu_67890_data.txt --verbose
```

## Tables Queried

The tool is configured to query the following tables:

- acct_desctradenames_2024_12
- acctdesc_2024_12
- atm_locations_2024_12
- credit_union_branch_information_2024_12
- foicu_2024_12
- foicudes_2024_12
- fs220_2024_12
- fs220a_2024_12
- fs220b_2024_12
- fs220c_2024_12
- fs220d_2024_12
- fs220g_2024_12
- fs220h_2024_12
- fs220i_2024_12
- fs220j_2024_12
- fs220k_2024_12
- fs220l_2024_12
- fs220m_2024_12
- fs220n_2024_12
- fs220p_2024_12
- fs220q_2024_12
- fs220r_2024_12
- fs220s_2024_12
- tradenames_2024_12

## How It Works

The tool first checks which of the configured tables have a `cu_number` column, and then queries only those tables for the specified credit union number. This ensures:

1. **Error Prevention**: Only tables with a cu_number column are queried
2. **Efficiency**: Skips tables that can't be queried by cu_number
3. **Complete Coverage**: All relevant tables are included in the search

## Output Format

### Interactive Mode

In interactive mode, the data is displayed as:

```
Table: table_name
=================
column1: value1
column2: value2
column3: value3
...
```

With color-coding to highlight:
- Column names (yellow)
- Numeric values (green)
- NULL values (dimmed)
- Zero values (dimmed)

### Command-line Mode (Readable Format)

In command-line mode with `--format readable`, the output follows this format:

```
RESULTS FOR CREDIT UNION #12345
Found data in 3 tables out of 15 queryable tables

SUMMARY OF TABLES WITH DATA:
---------------------------
table1: 1 records
table2: 1 records
table3: 1 records

DETAILED DATA BY TABLE:
======================

TABLE: table1
============
column1: value1
column2: value2
...

TABLE: table2
============
column1: value1
...
```

### Command-line Mode (JSON Format)

In command-line mode with the default JSON format, the output includes additional metadata:

```json
{
  "cu_number": "12345",
  "total_tables_with_data": 3,
  "total_tables_checked": 15,
  "total_tables_in_database": 24,
  "results": {
    "table1": [...],
    "table2": [...],
    "table3": [...]
  }
}
```

## Troubleshooting

Common issues:

- **Missing environment variables**: Ensure you have created a `.env` file with valid Supabase credentials
- **Connection errors**: Check that your Supabase URL and API key are correct
- **No data found**: Verify that the credit union number exists in your database
- **Type errors**: The script attempts to convert `cu_number` to an integer, as Supabase typically stores it as an integer
- **Permission issues**: Make sure your Supabase API key has permission to access the tables

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 

## NCUA Chatbot

The `ncua_chatbot.py` script provides an interactive chatbot interface for querying NCUA credit union data using natural language.

### Features
- Natural language understanding of credit union queries
- Intelligent entity extraction (credit union names, locations)
- Integration with vector search to find the right credit union
- Financial data retrieval and analysis
- Concise, human-readable answers

### Example Questions
- "What is Navy FCU's total asset size?"
- "What is the asset size of Members 1st CU in PA?"
- "How many members does Pentagon Federal Credit Union have?"
- "What was the net income for State Employees Credit Union last quarter?"

### Usage
```bash
# Install required dependencies
pip install -r requirements.txt

# Make sure environment variables are set in .env file
# OPENAI_API_KEY, SUPABASE_URL, SUPABASE_API_KEY, PINECONE_API_KEY, etc.

# Run the chatbot
python ncua_chatbot.py
``` 