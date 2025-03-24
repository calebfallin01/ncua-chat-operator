#!/usr/bin/env python3
"""
Pinecone Vector Database Diagnostic Script

A direct diagnostic tool to test the Pinecone connection and examine
what data is available in the vector database.
"""

import os
import sys
import json
import logging
from utils.pinecone_client import PineconeClient
from utils.openai_client import OpenAIClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pinecone_diagnostic.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_pinecone_connection():
    """Test the basic Pinecone connection and print index information"""
    try:
        logger.info("=== TESTING PINECONE CONNECTION ===")
        
        # Initialize Pinecone client
        logger.info("Initializing Pinecone client...")
        pc = PineconeClient()
        
        # Get index information
        logger.info("Getting index information...")
        stats = pc.index.describe_index_stats()
        logger.info(f"Index name: {pc.index_name}")
        logger.info(f"Vector dimension: {pc.vector_dimension}")
        logger.info(f"Total vector count: {stats.total_vector_count}")
        
        # Check namespaces
        if hasattr(stats, 'namespaces') and stats.namespaces:
            namespaces = list(stats.namespaces.keys())
            logger.info(f"Available namespaces: {namespaces}")
            
            # Count vectors in each namespace
            for ns in namespaces:
                count = stats.namespaces[ns].vector_count
                logger.info(f"  Namespace '{ns}' has {count} vectors")
        else:
            logger.warning("No namespaces found!")
            
        logger.info("Pinecone connection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error testing Pinecone connection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_basic_query():
    """Test a basic query without filters to see if any results can be found"""
    try:
        logger.info("=== TESTING BASIC PINECONE QUERY ===")
        
        # Initialize Pinecone client
        pc = PineconeClient()
        
        # Create a dummy vector for metadata-only search
        dummy_vector = [0.0] * pc.vector_dimension
        
        # Get all namespaces first
        stats = pc.index.describe_index_stats()
        namespaces = list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') and stats.namespaces else []
        
        # Try querying each namespace
        for ns in namespaces:
            logger.info(f"Testing query in namespace: {ns}")
            
            # Try with a namespace filter
            filter_dict = {"namespace": ns}
            
            logger.info(f"Using filter: {filter_dict}")
            
            # Query with namespace filter
            results = pc.index.query(
                vector=dummy_vector,
                filter=filter_dict,
                top_k=5,
                include_metadata=True
            )
            
            matches = results.matches
            logger.info(f"Found {len(matches)} matches in namespace '{ns}'")
            
            # Log details of each match
            for i, match in enumerate(matches):
                logger.info(f"Match {i+1} ID: {match.id}, Score: {match.score}")
                if hasattr(match, 'metadata') and match.metadata:
                    metadata = match.metadata
                    logger.info(f"  Metadata keys: {list(metadata.keys())}")
                    
                    # Look for specific fields we're interested in
                    fields_of_interest = ['namespace', 'status', 'account', 'tablename']
                    for field in fields_of_interest:
                        value = metadata.get(field, 'NOT FOUND')
                        logger.info(f"  {field}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in basic query test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_account_search():
    """Test searching for financial account information specifically"""
    try:
        logger.info("=== TESTING ACCOUNT SEARCH ===")
        
        # Initialize clients
        pc = PineconeClient()
        openai = OpenAIClient()
        
        # List of financial terms to search for
        search_terms = [
            "total assets",
            "net income",
            "shares",
            "loans"
        ]
        
        # Try searching for each term
        for term in search_terms:
            logger.info(f"Testing search for term: '{term}'")
            
            # Generate embedding for the term
            embedding = openai.create_embedding(term)
            if not embedding:
                logger.error(f"Could not generate embedding for '{term}'")
                continue
                
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            
            # Search in the Accounts namespace if it exists
            filter_dict = {"namespace": "Accounts"}
            
            logger.info(f"Searching with filter: {filter_dict}")
            
            # Query with the embedding
            results = pc.index.query(
                vector=embedding,
                filter=filter_dict,
                top_k=5,
                include_metadata=True
            )
            
            matches = results.matches
            logger.info(f"Found {len(matches)} matches for '{term}'")
            
            # Log details of each match
            for i, match in enumerate(matches):
                logger.info(f"Match {i+1} ID: {match.id}, Score: {match.score}")
                if hasattr(match, 'metadata') and match.metadata:
                    metadata = match.metadata
                    account = metadata.get('account', 'NOT FOUND')
                    tablename = metadata.get('tablename', 'NOT FOUND')
                    status = metadata.get('status', 'NOT FOUND')
                    
                    logger.info(f"  account: {account}")
                    logger.info(f"  tablename: {tablename}")
                    logger.info(f"  status: {status}")
                    
                    # Check if this has our required fields
                    if account != 'NOT FOUND' and tablename != 'NOT FOUND':
                        logger.info(f"  ✓ FOUND VALID ACCOUNT MATCH: {account} in {tablename}")
                    else:
                        logger.info(f"  ✗ MISSING REQUIRED FIELDS")
            
            # If no matches found with the filter, try without filter
            if not matches:
                logger.info(f"No matches found with filter. Trying without filter...")
                
                # Query without filter
                results = pc.index.query(
                    vector=embedding,
                    top_k=5,
                    include_metadata=True
                )
                
                matches = results.matches
                logger.info(f"Found {len(matches)} unfiltered matches for '{term}'")
                
                # Log details of unfiltered matches
                for i, match in enumerate(matches):
                    logger.info(f"Unfiltered match {i+1} ID: {match.id}, Score: {match.score}")
                    if hasattr(match, 'metadata') and match.metadata:
                        metadata = match.metadata
                        logger.info(f"  Metadata keys: {list(metadata.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in account search test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_status_formats():
    """Test different status field formats (string vs boolean)"""
    try:
        logger.info("=== TESTING STATUS FIELD FORMATS ===")
        
        # Initialize Pinecone client
        pc = PineconeClient()
        
        # Create a dummy vector for metadata-only search
        dummy_vector = [0.0] * pc.vector_dimension
        
        # Different formats to try
        status_formats = [
            {"status": "true"},
            {"status": True},
            {"status": 1},
            {"status": "TRUE"},
            {"status": "1"},
            {"status": "active"}
        ]
        
        # Try each status format
        for format_dict in status_formats:
            logger.info(f"Testing status filter: {format_dict}")
            
            # Query with the status filter
            results = pc.index.query(
                vector=dummy_vector,
                filter=format_dict,
                top_k=5,
                include_metadata=True
            )
            
            matches = results.matches
            logger.info(f"Found {len(matches)} matches with status filter: {format_dict}")
            
            # Log detailed info about the first match if any are found
            if matches:
                match = matches[0]
                logger.info(f"First match ID: {match.id}, Score: {match.score}")
                if hasattr(match, 'metadata') and match.metadata:
                    metadata = match.metadata
                    logger.info(f"  Metadata: {metadata}")
                    logger.info(f"  Status field: {metadata.get('status', 'NOT FOUND')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in status format test: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all diagnostic tests"""
    print("PINECONE VECTOR DATABASE DIAGNOSTIC TOOL")
    print("This script tests connectivity and searches the database")
    print("See the log file for detailed results: pinecone_diagnostic.log\n")
    
    # Run all tests
    tests = [
        ("Testing Pinecone connection", test_pinecone_connection),
        ("Testing basic query", test_basic_query),
        ("Testing account search", test_account_search),
        ("Testing status field formats", test_status_formats)
    ]
    
    for description, test_func in tests:
        print(f"Running: {description}...")
        success = test_func()
        if success:
            print(f"✓ {description} completed successfully\n")
        else:
            print(f"✗ {description} failed! Check the log for details.\n")
    
    print("All diagnostic tests complete!")
    print("Check the pinecone_diagnostic.log file for detailed results")

if __name__ == "__main__":
    main() 