#!/usr/bin/env python3
"""
Direct Pinecone API Test Script

Tests different ways of querying the Pinecone database to find working filter methods.
"""

import os
import sys
import json
import logging
from utils.pinecone_client import PineconeClient

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pinecone_direct_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def test_filter_variations():
    """Test different filter syntax variations to find what works"""
    
    try:
        # Initialize the Pinecone client
        pc = PineconeClient()
        logger.info(f"Connected to Pinecone index: {pc.index_name}")
        
        # Get index stats to verify we're working with the right index
        stats = pc.index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        # Create a dummy vector for testing (using the correct dimension)
        vector_dim = pc.vector_dimension
        dummy_vector = [0.0] * vector_dim
        logger.info(f"Using dummy vector with dimension: {vector_dim}")
        
        # Log all available namespaces
        if hasattr(stats, 'namespaces') and stats.namespaces:
            logger.info(f"Available namespaces: {list(stats.namespaces.keys())}")
            for ns, info in stats.namespaces.items():
                logger.info(f"  Namespace '{ns}' has {info.vector_count} vectors")
        
        # Try different filter formats for the namespace
        logger.info("\n\n==== TESTING NAMESPACE FILTER VARIATIONS ====")
        namespace_filters = [
            {"namespace": "Accounts"},  # Standard equality
            {"namespace": "accounts"},  # Lowercase
            {"namespace": {"$eq": "Accounts"}},  # Explicit equality operator
            {"namespace": {"$in": ["Accounts"]}},  # In list operator
            {"$text": {"$search": "Accounts", "$path": "namespace"}},  # Text search format
            # No filter - baseline
            None
        ]
        
        for filter_dict in namespace_filters:
            filter_name = str(filter_dict) if filter_dict else "None (no filter)"
            logger.info(f"\nTesting namespace filter: {filter_name}")
            
            try:
                # Query with this filter
                results = pc.index.query(
                    vector=dummy_vector,
                    filter=filter_dict,
                    top_k=5,
                    include_metadata=True
                )
                
                matches = results.matches
                match_count = len(matches)
                logger.info(f"Filter {filter_name} returned {match_count} matches")
                
                # If we found matches, examine the first one
                if match_count > 0:
                    logger.info(f"First match ID: {matches[0].id}, Score: {matches[0].score}")
                    logger.info(f"First match metadata: {matches[0].metadata}")
                    
                    # Log all metadata keys
                    if hasattr(matches[0], 'metadata') and matches[0].metadata:
                        keys = list(matches[0].metadata.keys())
                        logger.info(f"Metadata keys: {keys}")
                        
                        # Check if our expected fields exist
                        if 'account' in keys and 'tablename' in keys:
                            logger.info("SUCCESS! Found 'account' and 'tablename' fields")
                            logger.info(f"account: {matches[0].metadata['account']}")
                            logger.info(f"tablename: {matches[0].metadata['tablename']}")
                        elif 'Account' in keys and 'TableName' in keys:
                            logger.info("Found capitalized 'Account' and 'TableName' fields")
                            logger.info(f"Account: {matches[0].metadata['Account']}")
                            logger.info(f"TableName: {matches[0].metadata['TableName']}")
                        else:
                            logger.info("'account' and 'tablename' fields not found")
                            
                            # Check for any fields that might contain the info we need
                            for key, value in matches[0].metadata.items():
                                logger.info(f"  {key}: {value}")
            except Exception as e:
                logger.error(f"Error with filter {filter_name}: {str(e)}")
        
        # Try targeted query with known account terms
        logger.info("\n\n==== TESTING DIRECT QUERIES FOR KNOWN FINANCIAL TERMS ====")
        financial_terms = [
            "total assets", 
            "net income", 
            "loans", 
            "shares", 
            "deposits"
        ]
        
        from utils.openai_client import OpenAIClient
        openai = OpenAIClient()
        
        for term in financial_terms:
            logger.info(f"\nTrying direct query for: '{term}'")
            
            try:
                # Generate embedding for the term
                embedding = openai.create_embedding(term)
                if not embedding:
                    logger.error(f"Failed to generate embedding for '{term}'")
                    continue
                
                logger.info(f"Generated embedding with {len(embedding)} dimensions")
                
                # Try different variations of the namespace filter
                working_filter = None
                best_match_count = 0
                
                for filter_dict in namespace_filters:
                    filter_name = str(filter_dict) if filter_dict else "None (no filter)"
                    logger.info(f"  Using filter: {filter_name}")
                    
                    try:
                        # Query with this filter
                        results = pc.index.query(
                            vector=embedding,
                            filter=filter_dict,
                            top_k=5,
                            include_metadata=True
                        )
                        
                        matches = results.matches
                        match_count = len(matches)
                        logger.info(f"  Filter {filter_name} returned {match_count} matches")
                        
                        # If this filter yielded more matches than previous ones, record it
                        if match_count > best_match_count:
                            best_match_count = match_count
                            working_filter = filter_dict
                            
                            # If we found matches, examine the first one
                            if match_count > 0:
                                logger.info(f"  First match ID: {matches[0].id}, Score: {matches[0].score}")
                                
                                # Log all metadata keys
                                if hasattr(matches[0], 'metadata') and matches[0].metadata:
                                    keys = list(matches[0].metadata.keys())
                                    logger.info(f"  Metadata keys: {keys}")
                    except Exception as e:
                        logger.error(f"  Error with filter {filter_name}: {str(e)}")
                
                # Log the best filter for this term
                if working_filter:
                    logger.info(f"Best filter for '{term}': {working_filter}")
                else:
                    logger.info(f"No working filter found for '{term}'")
                    
            except Exception as e:
                logger.error(f"Error processing term '{term}': {str(e)}")
        
        # Try examining raw vectors directly from the database
        logger.info("\n\n==== EXAMINING RAW VECTORS FROM THE ACCOUNTS NAMESPACE ====")
        
        try:
            # Try to fetch some vectors directly with various namespace filters
            for filter_variant in [
                {"namespace": "Accounts"},
                {"namespace": "accounts"},
                None  # No filter
            ]:
                filter_str = str(filter_variant) if filter_variant else "None"
                logger.info(f"Fetching vectors with filter: {filter_str}")
                
                try:
                    results = pc.index.query(
                        vector=dummy_vector,
                        filter=filter_variant,
                        top_k=10,
                        include_metadata=True
                    )
                    
                    matches = results.matches
                    logger.info(f"Found {len(matches)} vectors")
                    
                    # Examine each match
                    for i, match in enumerate(matches):
                        logger.info(f"Vector {i+1}:")
                        logger.info(f"  ID: {match.id}")
                        logger.info(f"  Score: {match.score}")
                        
                        # Check metadata structure
                        if hasattr(match, 'metadata') and match.metadata:
                            metadata = match.metadata
                            keys = list(metadata.keys())
                            logger.info(f"  Metadata keys: {keys}")
                            
                            # Look for any fields that might contain account/tablename info
                            account_candidates = [k for k in keys if 'account' in k.lower()]
                            table_candidates = [k for k in keys if 'table' in k.lower()]
                            
                            if account_candidates:
                                logger.info(f"  Potential account fields: {account_candidates}")
                                for field in account_candidates:
                                    logger.info(f"    {field}: {metadata[field]}")
                            
                            if table_candidates:
                                logger.info(f"  Potential table fields: {table_candidates}")
                                for field in table_candidates:
                                    logger.info(f"    {field}: {metadata[field]}")
                            
                            # Look at status field if it exists
                            if 'status' in keys:
                                logger.info(f"  status: {metadata['status']}")
                            elif 'Status' in keys:
                                logger.info(f"  Status: {metadata['Status']}")
                        else:
                            logger.info("  No metadata available")
                            
                except Exception as e:
                    logger.error(f"Error fetching vectors with filter {filter_str}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Error examining raw vectors: {str(e)}")
            
        logger.info("\n\n==== TESTING COMPLETE ====")
        print("Testing complete. See pinecone_direct_test.log for detailed results.")
        
    except Exception as e:
        logger.error(f"Error in test_filter_variations: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    print("Running direct Pinecone API tests...")
    test_filter_variations()
    print("All tests complete. Check pinecone_direct_test.log for detailed results.") 