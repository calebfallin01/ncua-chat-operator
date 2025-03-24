#!/usr/bin/env python3
"""
Accounts Vector Search - Script for querying the Pinecone database for account information

This script:
1. Takes parenthetical input from a user's query (format: {search_term})
2. First matches the query to a category
3. Searches the Pinecone vector database in the 'Accounts' namespace within that category
4. Returns the 'Code' and 'Description' fields for each match

Example usage:
  python accounts_vector_search.py --query "total assets"
  python accounts_vector_search.py --input "What is the {asset size} of Navy Federal?"
  echo "What is their {net income}?" | python accounts_vector_search.py
  
The script will:
- Extract queries in curly braces: {query}
- First categorize each query
- Find matching accounts in the vector database within that category
- Return exactly one result per query (the best match)
- Return 'Code' and 'Description' fields in the output JSON
"""

import os
import sys
import re
import json
import logging
import traceback
from typing import List, Dict, Any, Optional, Tuple
import argparse

# Import our local utilities
from utils.openai_client import OpenAIClient
from utils.pinecone_client import PineconeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("accounts_search.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define available categories
AVAILABLE_CATEGORIES = [
    "Loans",
    "Other Assets",
    "Net Worth",
    "Shares/Deposits",
    "Equity",
    "Delinquency",
    "Specialized Lending",
    "Liquidity and Commitments",
    "Miscellaneous Information",
    "Investments",
    "Income",
    "Charge Offs and Recoveries",
    "Expenses",
    "Cost of Funds",
    "Cash and Cash Equivalents",
    "Cash and Other Deposits",
    "Liabilities",
    "CUSO",
    "Risk Based Capital"
]

class AccountsSearcher:
    """Class for searching the Accounts namespace in the Pinecone database"""
    
    def __init__(self):
        """Initialize the searcher with necessary clients"""
        try:
            # Initialize the OpenAI client for generating embeddings
            self.openai_client = OpenAIClient()
            
            # Initialize the Pinecone client for vector search
            self.pinecone_client = PineconeClient()
            
            # Get index information to verify connection
            try:
                stats = self.pinecone_client.index.describe_index_stats()
                logger.info(f"Connected to Pinecone index with stats: {stats}")
                
                # Log available namespaces
                if hasattr(stats, 'namespaces') and stats.namespaces:
                    namespaces = list(stats.namespaces.keys())
                    logger.info(f"Available namespaces: {namespaces}")
                    if 'Accounts' in namespaces:
                        logger.info(f"Found 'Accounts' namespace with vector count: {stats.namespaces.get('Accounts').vector_count}")
                    else:
                        logger.warning(f"'Accounts' namespace not found! Available namespaces: {namespaces}")
                else:
                    logger.warning("No namespaces found in index stats")
            except Exception as e:
                logger.error(f"Error getting index stats: {str(e)}")
                logger.error(traceback.format_exc())
            
            logger.info("Initialized AccountsSearcher")
        except Exception as e:
            logger.error(f"Error initializing AccountsSearcher: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def extract_queries(self, text: str) -> List[str]:
        """
        Extract queries enclosed in curly braces from the input text.
        
        Args:
            text: The input text containing parenthetical queries
            
        Returns:
            List of extracted query strings
        """
        # Use regex to find all content within curly braces
        pattern = r'\{([^{}]*)\}'
        matches = re.findall(pattern, text)
        
        # Log the extraction results
        if matches:
            logger.info(f"Extracted {len(matches)} queries: {matches}")
        else:
            logger.info(f"No queries found in input: {text}")
            
        return matches
    
    def determine_category(self, query: str) -> str:
        """
        Determine the most appropriate category for the given query.
        
        Args:
            query: The search query string
            
        Returns:
            The most likely category for the query
        """
        try:
            # Use OpenAI to categorize the query
            messages = [
                {
                    "role": "system",
                    "content": f"""You are an expert in financial categorization for credit unions.
Your task is to determine which category a financial query belongs to.
The available categories are:
{', '.join(AVAILABLE_CATEGORIES)}

Return ONLY the most appropriate category from the list above, with no explanation or additional text.
"""
                },
                {
                    "role": "user",
                    "content": f"Determine the category for this financial query: '{query}'"
                }
            ]
            
            response = self.openai_client.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=0.1,
                max_tokens=20
            )
            
            category = response.choices[0].message.content.strip()
            
            # Verify the category is valid
            if category in AVAILABLE_CATEGORIES:
                logger.info(f"Determined category '{category}' for query: {query}")
                return category
            else:
                # Try to match to the closest category
                for available_cat in AVAILABLE_CATEGORIES:
                    if available_cat.lower() in category.lower():
                        logger.info(f"Matched approximate category '{available_cat}' for query: {query}")
                        return available_cat
                
                # Default to a general category if no match
                logger.warning(f"Category '{category}' not in available categories, defaulting to 'Miscellaneous Information'")
                return "Miscellaneous Information"
                
        except Exception as e:
            logger.error(f"Error determining category: {str(e)}")
            # Default to general category
            return "Miscellaneous Information"
    
    def search_account_direct(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Search for an account with category-based refinement and direct filter queries.
        
        Args:
            query: The search query string
            
        Returns:
            Dictionary with 'Code', 'Description', 'Category', 'Type' if found, None otherwise
        """
        try:
            # First determine the category for the query
            category = self.determine_category(query)
            
            # Generate an embedding for the query
            logger.info(f"Generating embedding for query: {query}")
            embedding = self.openai_client.create_embedding(query)
            
            if not embedding:
                logger.error("Failed to generate embedding")
                return None
            
            # Log embedding details
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            
            logger.info(f"Searching 'Accounts' namespace for query in category '{category}': {query}")
            
            try:
                # Use the namespace parameter directly with category filter - using lowercase 'category' field
                results = self.pinecone_client.index.query(
                    vector=embedding,
                    namespace="Accounts",
                    top_k=10,  # Return more results for reranking
                    include_metadata=True,
                    filter={"category": {"$eq": category}}  # Filter by category - lowercase to match actual metadata
                )
                
                # Log the raw results
                logger.info(f"Query response: {results}")
                
                matches = results.matches
                logger.info(f"Found {len(matches)} matches in 'Accounts' namespace with category '{category}'")
                
                # If no matches found, try without category filter (debugging)
                if not matches:
                    logger.info(f"No matches found with category filter. Trying without category filter.")
                    results_no_filter = self.pinecone_client.index.query(
                        vector=embedding,
                        namespace="Accounts",
                        top_k=10,  # Return more results for reranking
                        include_metadata=True
                    )
                    
                    matches = results_no_filter.matches
                    logger.info(f"Found {len(matches)} matches without category filter")
                    
                    # Log full details of matches for debugging
                    if matches:
                        logger.info("First match details:")
                        first_match = matches[0]
                        logger.info(f"ID: {first_match.id}")
                        logger.info(f"Score: {first_match.score}")
                        
                        # Check if metadata exists
                        if hasattr(first_match, 'metadata'):
                            logger.info(f"Metadata: {first_match.metadata}")
                            if first_match.metadata:
                                for key, value in first_match.metadata.items():
                                    logger.info(f"Metadata field '{key}': {value}")
                            else:
                                logger.info("Metadata is empty dictionary")
                        else:
                            logger.info("No metadata attribute found")
                            
                        # Log info about all matches
                        for i, match in enumerate(matches):
                            logger.info(f"Match {i+1} ID: {match.id}, Score: {match.score}")
                            if hasattr(match, 'metadata') and match.metadata:
                                logger.info(f"Match {i+1} metadata keys: {match.metadata.keys()}")
                    
                    # If we found matches, log the categories to help diagnose category issues
                    if matches:
                        categories = set()
                        for match in matches:
                            if hasattr(match, 'metadata') and match.metadata:
                                cat = match.metadata.get('category')  # Use lowercase 'category'
                                if cat:
                                    categories.add(cat)
                        
                        logger.info(f"Available categories in matches: {list(categories)}")
                
                # If we have matches, apply hybrid ranking
                if matches:
                    # Normalize the query for text matching
                    query_normalized = query.lower().strip()
                    
                    # Store the reranked matches
                    reranked_matches = []
                    
                    # For debugging
                    all_scores = []
                    
                    # Parse the query into individual terms for more granular matching
                    query_terms = query_normalized.split()
                    
                    # Define specialized boosting keywords that indicate measurement types or calculation methods
                    measurement_keywords = [
                        'total', 'sum', 'average', 'avg', 'mean', 'median', 'dollar amount', 'dollars', 
                        'percentage', 'percent', 'ratio', 'number of', 'count', 'quantity', 'aggregate',
                        'net', 'gross', 'balance', '#', 'per', 'amount', 'value', 'volume'
                    ]
                    
                    # Define keywords that indicate the nature of financial metrics
                    financial_keywords = [
                        'assets', 'liabilities', 'equity', 'capital', 'deposit', 'deposits', 'loan', 'loans',
                        'interest', 'income', 'expense', 'revenue', 'cost', 'profit', 'loss', 'earnings',
                        'share', 'shares', 'member', 'members', 'fee', 'fees', 'reserve', 'reserves',
                        'investment', 'investments', 'debt', 'credit', 'debit', 'cash', 'liquidity',
                        'mortgage', 'mortgages', 'account', 'accounts', 'agriculture', 'farm', 'cuso',
                        'delinquent', 'delinquency', 'allowance', 'asset quality', 'mbl', 'charge-off'
                    ]
                    
                    for match in matches:
                        metadata = match.metadata
                        orig_score = match.score
                        
                        # Get text fields for matching - use lowercase field names to match the database
                        description = metadata.get('description', '').lower().strip()
                        code = metadata.get('code', '').lower().strip()
                        category_value = metadata.get('category', '').lower().strip()
                        type_value = metadata.get('type', '').lower().strip()
                        
                        # Start with the original embedding similarity score
                        final_score = orig_score
                        score_explanation = []
                        
                        # --- DESCRIPTION PRIMARY MATCHING (Highest priority) ---
                        
                        # 1. Exact match with Description (highest possible boost) 
                        if description == query_normalized:
                            # Very strong boost for exact description match
                            final_score += 1.0
                            score_explanation.append(f"Exact Description match: +1.0")
                        
                        # 2. Description contains the entire query (strong boost)
                        elif query_normalized in description:
                            # Strong boost for full query in description
                            final_score += 0.8
                            score_explanation.append(f"Description contains full query: +0.8")
                            
                        # 3. query contains a significant portion of description
                        elif len(description) > 20 and description[:30] in query_normalized:
                            # Strong boost for beginning of description in query
                            final_score += 0.7
                            score_explanation.append(f"query contains description beginning: +0.7")
                            
                        # --- MEASUREMENT KEYWORD MATCHING (High priority) ---
                        
                        # Count measurement keywords that appear in both query and description
                        measurement_matches = set()
                        for keyword in measurement_keywords:
                            if keyword in query_normalized and keyword in description:
                                measurement_matches.add(keyword)
                        
                        # Boost based on measurement keyword matches
                        if len(measurement_matches) >= 3:
                            # Very strong boost for multiple measurement keyword matches
                            final_score += 0.6
                            score_explanation.append(f"Multiple ({len(measurement_matches)}) measurement keywords matched: +0.6")
                        elif len(measurement_matches) == 2:
                            # Strong boost for two measurement keyword matches
                            final_score += 0.5
                            score_explanation.append(f"Two measurement keywords matched ({', '.join(measurement_matches)}): +0.5")
                        elif len(measurement_matches) == 1:
                            # Medium boost for one measurement keyword match
                            final_score += 0.4
                            score_explanation.append(f"One measurement keyword matched ({next(iter(measurement_matches))}): +0.4")
                            
                        # --- FINANCIAL KEYWORD MATCHING ---
                        
                        # Count financial keywords that appear in both query and description
                        financial_matches = set()
                        for keyword in financial_keywords:
                            if keyword in query_normalized and keyword in description:
                                financial_matches.add(keyword)
                        
                        # Boost based on financial keyword matches
                        if len(financial_matches) >= 3:
                            # Strong boost for multiple financial keyword matches
                            final_score += 0.5
                            score_explanation.append(f"Multiple ({len(financial_matches)}) financial keywords matched: +0.5")
                        elif len(financial_matches) == 2:
                            # Medium boost for two financial keyword matches
                            final_score += 0.4
                            score_explanation.append(f"Two financial keywords matched ({', '.join(financial_matches)}): +0.4")
                        elif len(financial_matches) == 1:
                            # Small boost for one financial keyword match
                            final_score += 0.3
                            score_explanation.append(f"One financial keyword matched ({next(iter(financial_matches))}): +0.3")
                            
                        # --- TOKEN MATCHING FOR DESCRIPTION ---
                        
                        # Parse tokens for comparison
                        query_tokens = set(query_normalized.split())
                        description_tokens = set(description.split())
                        
                        # Calculate token overlap metrics for description
                        if query_tokens and description_tokens:
                            # Calculate token overlap for description
                            desc_overlap = len(query_tokens.intersection(description_tokens))
                            desc_overlap_ratio = desc_overlap / len(query_tokens) if query_tokens else 0
                            
                            # Strong boost for high token overlap ratio with description
                            if desc_overlap_ratio >= 0.8:
                                final_score += 0.7
                                score_explanation.append(f"High token overlap ({desc_overlap_ratio:.2f}) with Description: +0.7")
                            elif desc_overlap_ratio >= 0.6:
                                final_score += 0.6
                                score_explanation.append(f"Good token overlap ({desc_overlap_ratio:.2f}) with Description: +0.6")
                            elif desc_overlap_ratio >= 0.4:
                                final_score += 0.5
                                score_explanation.append(f"Medium token overlap ({desc_overlap_ratio:.2f}) with Description: +0.5")
                            elif desc_overlap_ratio >= 0.2 and desc_overlap >= 2:
                                final_score += 0.3
                                score_explanation.append(f"Some token overlap ({desc_overlap_ratio:.2f}) with Description: +0.3")
                        
                        # --- CODE MATCHING (Lower priority) ---
                        
                        # Exact match with Code
                        if code == query_normalized:
                            # Strong boost for exact match
                            final_score += 0.6
                            score_explanation.append(f"Exact Code match: +0.6")
                        # Contains match with Code
                        elif query_normalized in code:
                            # Medium boost for partial match in code
                            final_score += 0.4
                            score_explanation.append(f"Code contains query: +0.4")
                        # Code contains in query (reverse match)
                        elif len(code) >= 2 and code in query_normalized:
                            # Smaller boost for code found in query
                            final_score += 0.3
                            score_explanation.append(f"query contains Code: +0.3")
                        
                        # --- TYPE AND CATEGORY MATCHING (Supplemental) ---
                        
                        # Extra boost if Type is relevant
                        if type_value and type_value in query_normalized:
                            final_score += 0.3
                            score_explanation.append(f"Type '{type_value}' found in query: +0.3")
                            
                        # Store the reranked score and explanation
                        match._reranked_score = final_score
                        match._score_explanation = " | ".join(score_explanation)
                        reranked_matches.append(match)
                        
                        # Log the scoring details
                        score_detail = {
                            'id': match.id,
                            'Code': code,
                            'Description': description[:50] + "..." if len(description) > 50 else description,
                            'original_score': orig_score,
                            'reranked_score': final_score,
                            'explanation': match._score_explanation
                        }
                        all_scores.append(score_detail)
                    
                    # Log all scoring details for transparency
                    logger.info(f"Reranking details: {all_scores}")
                    
                    # Sort by reranked score
                    reranked_matches.sort(key=lambda x: x._reranked_score, reverse=True)
                    
                    # Select the top match after reranking
                    top_match = reranked_matches[0]
                    metadata = top_match.metadata
                    
                    # Log detailed match information
                    logger.info(f"Best match after reranking: {top_match.id}")
                    logger.info(f"Original score: {top_match.score}, Reranked score: {top_match._reranked_score}")
                    logger.info(f"Match metadata: {metadata}")
                    
                    # Extract code, description, category, and type from metadata
                    code = metadata.get('code')
                    description = metadata.get('description')
                    category_value = metadata.get('category')
                    type_value = metadata.get('type')
                    
                    if code and description:
                        logger.info(f"Top match: Code={code}, Description={description}, Category={category_value}, Type={type_value}, score={top_match._reranked_score}")
                        return {
                            'Code': code,
                            'Description': description,
                            'Category': category_value,
                            'Type': type_value,
                            'score': top_match._reranked_score
                        }
                    else:
                        logger.warning(f"Top match missing Code or Description!")
                        logger.warning(f"Available metadata fields: {list(metadata.keys())}")
                else:
                    # If no matches with status=true, try to use the top match anyway
                    if matches:
                        top_match = matches[0]
                        metadata = top_match.metadata
                        
                        # Extract code, description, category, and type from metadata
                        code = metadata.get('code')
                        description = metadata.get('description')
                        category_value = metadata.get('category')
                        type_value = metadata.get('type')
                        
                        if code and description:
                            logger.info(f"Using top match despite status: Code={code}, Description={description}, Category={category_value}, Type={type_value}, score={top_match.score}")
                            return {
                                'Code': code,
                                'Description': description,
                                'Category': category_value,
                                'Type': type_value,
                                'score': top_match.score
                            }
                
                # Log if no suitable matches were found
                logger.info(f"No matching accounts found for query: {query}")
                return None
                
            except Exception as e:
                logger.error(f"Error during Pinecone query: {str(e)}")
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"Error in direct account search: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    def process_input(self, input_text: str) -> Dict[str, Any]:
        """
        Process input text, extract queries, and search for accounts.
        
        Args:
            input_text: The input text containing parenthetical queries
            
        Returns:
            Dictionary with query results
        """
        # Extract queries from input text
        queries = self.extract_queries(input_text)
        
        if not queries:
            logger.warning(f"No queries found in input text: '{input_text}'")
            return {"error": "No queries found in input text"}
        
        # Process each query and collect results
        results = {}
        for query in queries:
            logger.info(f"Processing query: '{query}'")
            # Use the direct filtered search method instead
            result = self.search_account_direct(query)
            if result:
                results[query] = result
                logger.info(f"Found result for '{query}': {result}")
            else:
                error_msg = {"error": "No matching account found"}
                results[query] = error_msg
                logger.warning(f"No result found for '{query}': {error_msg}")
        
        return results

def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Search for accounts in the vector database")
    parser.add_argument("--input", "-i", type=str, help="Input text containing queries in curly braces")
    parser.add_argument("--query", "-q", type=str, help="Direct query (will be wrapped in curly braces)")
    parser.add_argument("--file", "-f", type=str, help="File containing input text")
    parser.add_argument("--compact", "-c", action="store_true", help="Output compact results without scores")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Get input text from arguments
    input_text = None
    if args.input:
        input_text = args.input
    elif args.query:
        input_text = f"{{{args.query}}}"
    elif args.file:
        try:
            with open(args.file, 'r') as f:
                input_text = f.read()
        except Exception as e:
            logger.error(f"Error reading input file: {str(e)}")
            return
    else:
        # If no input is provided, read from stdin
        logger.info("No input provided, reading from stdin...")
        input_text = sys.stdin.read()
    
    # Process input text
    try:
        # Initialize the searcher
        searcher = AccountsSearcher()
        
        # Process input text
        results = searcher.process_input(input_text)
        
        # Format the results
        if args.compact:
            # Compact output - just term, Code, Description, Category, and Type
            compact_results = {}
            for query, result in results.items():
                if "error" in result:
                    compact_results[query] = result
                else:
                    compact_results[query] = {
                        "Code": result["Code"],
                        "Description": result["Description"],
                        "Category": result.get("Category", ""),
                        "Type": result.get("Type", "")
                    }
            print(json.dumps(compact_results, indent=2))
        else:
            # Full output with scores
            print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        print(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    main() 