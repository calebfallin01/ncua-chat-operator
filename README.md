# Accounts Vector Search

A Python script for querying the Pinecone vector database to find account information based on user queries.

## Overview

This script is designed to work with a financial chatbot system. It:

1. Takes parenthetical input from a user's query (format: `{search_term}`)
2. Searches the Pinecone vector database in the 'Accounts' namespace
3. Filters for entries with 'status' of 'true'
4. Returns the 'account', 'tablename', and 'acctname' fields for each match

## Requirements

- Python 3.7+
- OpenAI API key (set as environment variable `OPENAI_API_KEY`)
- Pinecone API key (set as environment variable `PINECONE_API_KEY`)
- Pinecone index name (set as environment variable `PINECONE_INDEX`)

## Installation

1. Clone the repository
2. Set up the required environment variables
3. Install the required dependencies:

```bash
pip install openai pinecone-client tenacity
```

## Usage

### Basic Usage

```bash
python accounts_vector_search.py --query "total assets"
```

This will search for the term "total assets" in the vector database.

### Advanced Usage

The script can be used in several ways:

1. **Direct query** - Provide a single search term:
   ```bash
   python accounts_vector_search.py --query "net income"
   ```

2. **Text with embedded queries** - Provide text with queries in curly braces:
   ```bash
   python accounts_vector_search.py --input "What is the {asset size} of Navy Federal?"
   ```

3. **Read from file** - Read input from a file:
   ```bash
   python accounts_vector_search.py --file user_questions.txt
   ```

4. **Pipe input** - Pipe text into the script:
   ```bash
   echo "What is their {net income}?" | python accounts_vector_search.py
   ```

### Output Options

- **Compact output** - Use the `--compact` flag to get simplified output without scores:
  ```bash
  python accounts_vector_search.py --query "total assets" --compact
  ```

- **Debug logging** - Use the `--debug` flag to enable detailed logging:
  ```bash
  python accounts_vector_search.py --query "total assets" --debug
  ```

## Output Format

The script returns JSON output with the following structure:

```json
{
  "search term": {
    "account": "acct_010",
    "tablename": "FS220",
    "acctname": "TOTAL ASSETS",
    "score": 0.87654321
  }
}
```

In compact mode:

```json
{
  "search term": {
    "account": "acct_010",
    "tablename": "FS220",
    "acctname": "TOTAL ASSETS"
  }
}
```

If no match is found:

```json
{
  "search term": {
    "error": "No matching account found"
  }
}
```

## Integration with Chatbot

This script is designed to be integrated with a chatbot system. The chatbot should:

1. Process user queries for financial information
2. Extract relevant search terms and enclose them in curly braces: `{term}`
3. Pass this formatted text to the script
4. Use the returned account, tablename, and acctname information to query and format financial data

## Example Workflow

1. User asks: "What is Navy Federal's asset size?"
2. Chatbot extracts the financial concept: "asset size"
3. Chatbot calls: `python accounts_vector_search.py --query "asset size"`
4. Script returns: `{"asset size": {"account": "acct_010", "tablename": "FS220", "acctname": "TOTAL ASSETS"}}`
5. Chatbot uses this information to query the financial data and format the response with the human-readable field name

## License

This project is licensed under the MIT License - see the LICENSE file for details. 