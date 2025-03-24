#!/usr/bin/env python3
"""
Query Accounts Namespace Directly

This script uses the Pinecone API to directly query the Accounts namespace.
"""

import os
import sys
import json
import logging
from pinecone import Pinecone

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("query_namespace.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def query_namespace_directly():
    """Query the Accounts namespace directly using the Pinecone API"""
    try:
        # Get API key from environment variable
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            logger.error("PINECONE_API_KEY environment variable not set")
            return
        
        # Get index name from environment variable
        index_name = os.environ.get("PINECONE_INDEX")
        if not index_name:
            logger.error("PINECONE_INDEX environment variable not set")
            return
        
        logger.info(f"Using API key: {api_key[:5]}... and index: {index_name}")
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        logger.info("Initialized Pinecone client")
        
        # Connect to the specified index
        index = pc.Index(index_name)
        logger.info(f"Connected to index: {index_name}")
        
        # Get index stats to verify namespaces
        stats = index.describe_index_stats()
        logger.info(f"Index stats: {stats}")
        
        # Log available namespaces
        namespaces = list(stats.namespaces.keys()) if hasattr(stats, 'namespaces') and stats.namespaces else []
        logger.info(f"Available namespaces: {namespaces}")
        
        # Try direct querying from each namespace
        # Create a dummy vector for querying
        dummy_vector = [0.0] * stats.dimension
        
        for namespace in namespaces:
            ns_name = namespace if namespace else "default (empty string)"
            logger.info(f"\nQuerying namespace: {ns_name}")
            
            # Try different API approaches
            
            # Approach 1: Using namespace parameter directly
            try:
                logger.info("Approach 1: Using namespace parameter")
                results = index.query(
                    vector=dummy_vector,
                    namespace=namespace,  # Using dedicated namespace parameter
                    top_k=5,
                    include_metadata=True
                )
                
                matches = results.matches
                logger.info(f"Approach 1 returned {len(matches)} matches")
                
                # Log first match if available
                if matches:
                    logger.info(f"First match ID: {matches[0].id}")
                    logger.info(f"First match metadata: {matches[0].metadata}")
            except Exception as e:
                logger.error(f"Error with approach 1: {str(e)}")
            
            # Approach 2: Using namespace in filter
            try:
                logger.info("Approach 2: Using namespace in filter")
                results = index.query(
                    vector=dummy_vector,
                    filter={"namespace": namespace},
                    top_k=5,
                    include_metadata=True
                )
                
                matches = results.matches
                logger.info(f"Approach 2 returned {len(matches)} matches")
            except Exception as e:
                logger.error(f"Error with approach 2: {str(e)}")
        
        # Try a direct fetch from the Accounts namespace
        # Query known financial terms
        from utils.openai_client import OpenAIClient
        openai = OpenAIClient()
        
        logger.info("\n\nTesting direct financial term queries with namespace parameter")
        terms = ["total assets", "net income", "loans", "shares"]
        
        for term in terms:
            logger.info(f"\nQuerying for term: {term}")
            
            # Generate embedding
            embedding = openai.create_embedding(term)
            if not embedding:
                logger.error(f"Failed to generate embedding for {term}")
                continue
                
            logger.info(f"Generated embedding with {len(embedding)} dimensions")
            
            # Try querying with namespace parameter
            for namespace in namespaces:
                ns_name = namespace if namespace else "default (empty string)"
                logger.info(f"Querying namespace: {ns_name}")
                
                try:
                    results = index.query(
                        vector=embedding,
                        namespace=namespace,
                        top_k=5,
                        include_metadata=True
                    )
                    
                    matches = results.matches
                    logger.info(f"Found {len(matches)} matches")
                    
                    # Log first match if available
                    if matches:
                        logger.info(f"Best match ID: {matches[0].id}, Score: {matches[0].score}")
                        logger.info(f"Best match metadata: {matches[0].metadata}")
                except Exception as e:
                    logger.error(f"Error querying namespace {ns_name}: {str(e)}")
        
        logger.info("\n\nTesting complete")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    print("Querying Accounts namespace directly...")
    query_namespace_directly()
    print("Complete. Check query_namespace.log for results.") 