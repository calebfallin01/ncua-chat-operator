#!/usr/bin/env python3
"""
Script to query the vector database of credit unions in Pinecone.
This script takes a credit union name as input, converts it to a vector using OpenAI,
and searches for similar credit unions in the Pinecone vector database.
"""

import sys
import argparse
from dotenv import load_dotenv

from utils.openai_client import OpenAIClient
from utils.pinecone_client import PineconeClient
from utils.text_utils import preprocess_text

# Load environment variables
load_dotenv()

def query_vector_database(query_text, top_k=20, use_comprehensive=True, context=None):
    """
    Query the vector database for similar credit unions.
    
    Args:
        query_text (str): The query text.
        top_k (int): The number of results to return.
        use_comprehensive (bool): Whether to use comprehensive search.
        context (dict): Optional location context (city, state_code, state_full).
        
    Returns:
        list: A list of matches.
    """
    print(f"Searching for: '{query_text}'")
    
    # Print context if provided
    if context:
        context_str = ", ".join([f"{k}: {v}" for k, v in context.items()])
        print(f"With context: {context_str}")
    
    # Initialize clients
    openai_client = OpenAIClient()
    pinecone_client = PineconeClient()
    
    # Preprocess the query text - get both normalized and original versions
    normalized_text, original_text = preprocess_text(query_text)
    
    # Show the normalized text that will be used for searching
    if normalized_text != original_text.lower():
        print(f"Using normalized search term: '{normalized_text}'")
    
    # Create an embedding for the normalized query (used for sorting in comprehensive search)
    query_embedding = openai_client.create_embedding(normalized_text)
    
    if not query_embedding:
        print("Failed to create embedding for the query.")
        return []
    
    if use_comprehensive:
        # Use the comprehensive search across all fields
        matches = pinecone_client.comprehensive_search(
            query=normalized_text, 
            vector=query_embedding,
            top_k=top_k,
            context=context
        )
        
        # If matches were found, print a summary
        if matches:
            match_types = {}
            for match in matches:
                match_type = getattr(match, '_match_type', 'unknown')
                match_types[match_type] = match_types.get(match_type, 0) + 1
            
            print(f"Found {len(matches)} matches with the following types:")
            for match_type, count in match_types.items():
                print(f"  - {match_type}: {count}")
                
        return matches
    else:
        # Use standard vector search
        matches = pinecone_client.query(query_embedding, top_k=top_k)
        return matches

def print_results(matches, query_text, min_score=0.75, show_all_results=False, force_min_score=False):
    """
    Print the search results.
    
    Args:
        matches (list): A list of matches from Pinecone.
        query_text (str): The original query text.
        min_score (float): Minimum similarity score to display (default: 0.75).
        show_all_results (bool): Whether to show all results above min_score (default: False).
        force_min_score (bool): Whether to force the use of min_score without prompting (default: False).
    """
    if not matches:
        print("No matches found.")
        return
    
    # Get the normalized version of the query for comparison
    normalized_query, _ = preprocess_text(query_text)
    
    # Sort by adjusted score if available, otherwise by regular score
    sorted_matches = sorted(
        matches, 
        key=lambda m: getattr(m, '_adjusted_score', m.score), 
        reverse=True
    )
    
    # Get the top match
    top_match = sorted_matches[0]
    
    # Get the score (adjusted if available)
    top_score = getattr(top_match, '_adjusted_score', top_match.score)
    top_match_type = getattr(top_match, '_match_type', 'STANDARD')
    top_field = getattr(top_match, '_field_matched', None)
    match_reason = getattr(top_match, '_match_reason', None)
    
    # Check if the best score is below the minimum threshold
    if top_score < min_score:
        cu_name = top_match.metadata.get('cu_name', 'Unknown')
        
        print(f"\nNo matches found with similarity score >= {min_score}.")
        print(f"The highest match was '{cu_name}' with a score of {top_score:.4f}")
        if match_reason:
            print(f"Match reason: {match_reason}")
        
        # Ask the user if they meant this credit union
        response = input(f"Were you looking for '{cu_name}'? (y/n): ").lower()
        
        if response.startswith('y'):
            # Display the match and return
            print("\nShowing details for your confirmed match:")
            print("-" * 50)
            
            metadata = top_match.metadata
            
            # Get field-specific match label
            match_label = f" ({top_match_type} MATCH)" if top_match_type != "STANDARD" else ""
            
            print(f"Credit Union: {metadata.get('cu_name', 'Unknown')}{match_label}")
            if top_field:
                print(f"Match Field: {top_field}")
            print(f"Score: {top_score:.4f}")
            if match_reason:
                print(f"Match Reason: {match_reason}")
            
            # Show original vector score if adjusted
            if hasattr(top_match, '_adjusted_score') and top_match._adjusted_score != top_match.score:
                print(f"Original Vector Score: {top_match.score:.4f}")
                
            # Show pre-context score if context was applied and exists
            if hasattr(top_match, '_pre_context_score') and top_match._pre_context_score is not None:
                print(f"Pre-Context Score: {top_match._pre_context_score:.4f}")
            
            print("-" * 50)
            
            # Print all metadata fields
            print("All Available Metadata:")
            for key, value in sorted(metadata.items()):
                # Skip normalized_name as it's used internally
                if key == 'normalized_name':
                    continue
                print(f"{key}: {value}")
            
            print("-" * 50)
            return
        else:
            print("No match confirmed. Try refining your search query.")
            return
    
    # Filter matches by minimum score
    filtered_matches = [match for match in sorted_matches if 
                        getattr(match, '_adjusted_score', match.score) >= min_score]
    
    if not filtered_matches:
        print(f"No matches found with similarity score >= {min_score}.")
        return
    
    # If we're only showing the top match
    if not show_all_results:
        # Only show top result
        top_match = filtered_matches[0]
        
        print("\nBest match found:")
        print("-" * 50)
        
        metadata = top_match.metadata
        score = getattr(top_match, '_adjusted_score', top_match.score)
        match_type = getattr(top_match, '_match_type', 'STANDARD')
        field = getattr(top_match, '_field_matched', None)
        match_reason = getattr(top_match, '_match_reason', None)
        
        # Get field-specific match label
        match_label = f" ({match_type} MATCH)" if match_type != "STANDARD" else ""
        
        print(f"Credit Union: {metadata.get('cu_name', 'Unknown')}{match_label}")
        if field:
            print(f"Match Field: {field}")
        print(f"Score: {score:.4f}")
        if match_reason:
            print(f"Match Reason: {match_reason}")
        
        # Show original vector score if adjusted
        if hasattr(top_match, '_adjusted_score') and top_match._adjusted_score != top_match.score:
            print(f"Original Vector Score: {top_match.score:.4f}")
            
        # Show pre-context score if context was applied and exists
        if hasattr(top_match, '_pre_context_score') and top_match._pre_context_score is not None:
            print(f"Pre-Context Score: {top_match._pre_context_score:.4f}")
        
        print("-" * 50)
        
        # Print all metadata fields
        print("All Available Metadata:")
        for key, value in sorted(metadata.items()):
            # Skip normalized_name as it's used internally
            if key == 'normalized_name':
                continue
            print(f"{key}: {value}")
            
        print("-" * 50)
        
        # Optionally show the number of other high-scoring matches
        other_matches = len(filtered_matches) - 1
        if other_matches > 0:
            print(f"There are {other_matches} additional matches with scores >= {min_score}.")
            print(f"Use --all-results to see all matches.")
    else:
        # Show all matches
        print(f"\nFound {len(filtered_matches)} matches with similarity score >= {min_score}:")
        
        for i, match in enumerate(filtered_matches):
            metadata = match.metadata
            score = getattr(match, '_adjusted_score', match.score)
            match_type = getattr(match, '_match_type', 'STANDARD') 
            field = getattr(match, '_field_matched', None)
            match_reason = getattr(match, '_match_reason', None)
            
            # Get field-specific match label
            match_label = f" ({match_type} MATCH)" if match_type != "STANDARD" else ""
            
            print("-" * 50)
            print(f"{i+1}. Credit Union: {metadata.get('cu_name', 'Unknown')}{match_label}")
            if field:
                print(f"   Match Field: {field}")
            print(f"   Score: {score:.4f}")
            if match_reason:
                print(f"   Match Reason: {match_reason}")
            
            # Show original vector score if adjusted
            if hasattr(match, '_adjusted_score') and match._adjusted_score != match.score:
                print(f"   Original Vector Score: {match.score:.4f}")
                
            # Show pre-context score if context was applied and exists
            if hasattr(match, '_pre_context_score') and match._pre_context_score is not None:
                print(f"   Pre-Context Score: {match._pre_context_score:.4f}")
            
            print("   " + "-" * 46)
            
            # Print all metadata fields
            print("   All Available Metadata:")
            for key, value in sorted(metadata.items()):
                # Skip normalized_name as it's used internally
                if key == 'normalized_name':
                    continue
                print(f"   {key}: {value}")
        
        print("-" * 50)

def interactive_mode():
    """Run the script in interactive mode."""
    print("Credit Union Vector Database Query Tool")
    print("Enter 'quit' or 'exit' to exit the program")
    print("-" * 50)
    
    # Default settings
    min_score = 0.75
    show_all_results = False
    top_k = 20  # Increase default to ensure we catch more potential matches
    use_comprehensive = True  # Default to comprehensive search
    context = None  # No context by default
    
    # Ask if user wants to customize settings
    response = input("Would you like to customize search settings? (y/n): ").lower()
    if response.startswith('y'):
        try:
            min_score_input = input(f"Minimum similarity score (0.0-1.0) [default: {min_score}]: ")
            if min_score_input:
                min_score = float(min_score_input)
                if min_score < 0 or min_score > 1:
                    print("Invalid score. Using default 0.75.")
                    min_score = 0.75
                    
            show_all_input = input("Show all results above minimum score? (y/n) [default: n]: ").lower()
            show_all_results = show_all_input.startswith('y')
            
            top_k_input = input(f"Number of candidates to retrieve (5-100) [default: {top_k}]: ")
            if top_k_input:
                top_k = int(top_k_input)
                if top_k < 5 or top_k > 100:
                    print("Invalid value. Using default 20.")
                    top_k = 20
                    
            comp_input = input(f"Use comprehensive search? (y/n) [default: y]: ").lower()
            if comp_input and not comp_input.startswith('y'):
                use_comprehensive = False
            
            # Ask about context
            use_context_input = input("Use location context for search prioritization? (y/n) [default: n]: ").lower()
            if use_context_input.startswith('y'):
                context = {}
                
                city_input = input("City (press Enter to skip): ")
                if city_input:
                    context['city'] = city_input
                    
                state_input = input("State name (press Enter to skip): ")
                if state_input:
                    context['state_full'] = state_input
                    
                state_code_input = input("State code (e.g., PA) (press Enter to skip): ")
                if state_code_input:
                    context['state_code'] = state_code_input
                
                # If no context was entered, set back to None
                if not context:
                    context = None
                else:
                    print(f"Using location context: {context}")
                
        except ValueError:
            print("Invalid input. Using default settings.")
    
    print(f"Using minimum similarity score: {min_score}")
    print(f"Show all results: {'Yes' if show_all_results else 'No - showing only top match'}")
    print(f"Number of candidates to retrieve: {top_k}")
    print(f"Using comprehensive search: {'Yes' if use_comprehensive else 'No - vector only'}")
    print("-" * 50)
    
    while True:
        query = input("\nEnter a credit union name to search for: ")
        
        if query.lower() in ['quit', 'exit']:
            print("Exiting program. Goodbye!")
            break
        
        if not query:
            print("No query provided. Please try again.")
            continue
        
        matches = query_vector_database(query, top_k=top_k, use_comprehensive=use_comprehensive, context=context)
        print_results(matches, query, min_score=min_score, show_all_results=show_all_results)

def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Query the credit union vector database.")
    parser.add_argument("query", nargs="*", help="The credit union name to search for")
    parser.add_argument("-k", "--top-k", type=int, default=20, help="Number of results to return from database")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("-m", "--min-score", type=float, default=0.75, 
                        help="Minimum similarity score (0.0-1.0), default: 0.75")
    parser.add_argument("-a", "--all-results", action="store_true", 
                        help="Show all results above minimum score instead of just the top match")
    parser.add_argument("-f", "--force-min-score", action="store_true",
                        help="Don't prompt for confirmation on low-scoring matches")
    parser.add_argument("-c", "--comprehensive", action="store_true", default=True,
                        help="Use comprehensive search across all fields (default: True)")
    parser.add_argument("-v", "--vector-only", action="store_true",
                        help="Use only vector similarity search (turns off comprehensive search)")
    
    # Location context arguments
    parser.add_argument("--city", type=str, help="City context for search prioritization")
    parser.add_argument("--state", type=str, help="State name context for search prioritization")
    parser.add_argument("--state-code", type=str, help="State code context for search prioritization (e.g., PA)")
    
    args = parser.parse_args()
    
    # Build context dictionary if any location parameters provided
    context = {}
    if args.city:
        context['city'] = args.city
    if args.state:
        context['state_full'] = args.state
    if args.state_code:
        context['state_code'] = args.state_code
    
    # Use empty context if no location params provided
    context = context if context else None
    
    if args.interactive:
        interactive_mode()
        return
    
    if args.query:
        # Use command line argument as query
        query = " ".join(args.query)
    else:
        # Prompt for input
        query = input("Enter a credit union name to search for: ")
    
    # If no query provided, exit
    if not query:
        print("No query provided. Exiting.")
        return
    
    # Use comprehensive search unless vector-only is specified
    use_comprehensive = not args.vector_only
    
    matches = query_vector_database(
        query, 
        top_k=args.top_k, 
        use_comprehensive=use_comprehensive,
        context=context
    )
    
    print_results(matches, query, min_score=args.min_score, 
                 show_all_results=args.all_results,
                 force_min_score=args.force_min_score)

if __name__ == "__main__":
    main() 