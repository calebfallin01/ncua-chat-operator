"""Pinecone client wrapper for vector search operations."""

import logging
import sys
import os
import json
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add the parent directory to the path so we can import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.settings import settings

# Configure logger
logger = logging.getLogger(__name__)

# Initialize global client variable
pinecone_client = None

class PineconeClient:
    """Client for interacting with Pinecone vector database."""
    
    def __init__(self):
        """Initialize the Pinecone client."""
        try:
            # Initialize Pinecone client with the newer v6 API
            logger.info("Initializing Pinecone client...")
            import pkg_resources
            version = pkg_resources.get_distribution("pinecone").version
            logger.info(f"Pinecone client version: {version}")
            
            # Initialize Pinecone with the new API
            self.pc = Pinecone(api_key=settings.pinecone_api_key)
            
            # Get the index
            self.index_name = settings.pinecone_index
            
            # Log available indexes
            try:
                indexes = self.pc.list_indexes()
                logger.info(f"Available indexes: {indexes}")
                
                if not indexes or self.index_name not in [idx.name for idx in indexes]:
                    logger.warning(f"Index '{self.index_name}' not found in available indexes: {[idx.name for idx in indexes]}")
            except Exception as e:
                logger.error(f"Error listing indexes: {str(e)}")
            
            # Try to connect to the index directly
            try:
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Pinecone client initialized for index: {self.index_name}")
                
                # Test connection with a describe_index_stats call
                stats = self.index.describe_index_stats()
                logger.info(f"Index stats: {stats}")
                self.vector_dimension = stats.dimension
                logger.info(f"Vector dimension: {self.vector_dimension}")
                logger.info(f"Connected to Pinecone index: {self.index_name}")
            except Exception as e:
                logger.error(f"Failed to connect to index: {str(e)}")
                raise ValueError(f"Index '{self.index_name}' does not exist or is not accessible.")
                
        except Exception as e:
            logger.error(f"Error initializing Pinecone client: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def fetch_vector(self, id: str) -> dict:
        """
        Fetch a vector directly by ID.
        
        Args:
            id: The vector ID to fetch
            
        Returns:
            Dictionary with vector data including metadata
        """
        try:
            # Using the newer Pinecone v6 API
            fetch_response = self.index.fetch(ids=[id])
            
            if id in fetch_response.vectors:
                vector_data = fetch_response.vectors[id]
                result = {
                    "id": id,
                    "metadata": vector_data.metadata
                }
                logger.info(f"Successfully fetched vector for ID: {id}")
                return result
            else:
                logger.warning(f"No vector found for ID: {id}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching vector from Pinecone: {str(e)}")
            # Log full exception details
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def search(self, embedding: list, top_k: int = 5) -> list:
        """
        Search for the most similar vectors to the provided embedding.
        
        Args:
            embedding: The query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of dictionary results with id, score, and metadata
        """
        try:
            # Make sure embedding matches the correct dimension
            if len(embedding) != self.vector_dimension:
                logger.warning(f"Input embedding dimension {len(embedding)} does not match index dimension {self.vector_dimension}")
                # Pad or truncate as needed
                if len(embedding) < self.vector_dimension:
                    # Pad with zeros
                    embedding = embedding + [0.0] * (self.vector_dimension - len(embedding))
                else:
                    # Truncate
                    embedding = embedding[:self.vector_dimension]
                logger.info(f"Adjusted embedding to dimension {len(embedding)}")
            
            # Using the newer Pinecone v6 API
            query_response = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            # Log the response
            logger.info(f"Query response received with {len(query_response.matches)} matches")
            
            # Transform results to a more usable format
            transformed_results = []
            for match in query_response.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata
                }
                transformed_results.append(result)
            
            return transformed_results
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            # Log full exception details
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def search_by_name(self, name: str, location: str = None, top_k: int = 20) -> list:
        """
        Search for credit unions by name and optionally location using metadata filtering.
        
        Args:
            name: Credit union name to search for
            location: Optional location filter
            top_k: Number of results to return
            
        Returns:
            List of dictionary results with id, score, and metadata
        """
        try:
            if not name:
                return []

            # Create a dummy vector matching the index dimension
            dummy_vector = [0.0] * self.vector_dimension
            
            # First try exact match
            filter_dict = {}
            if name:
                # Convert to lowercase for consistency
                filter_dict["cu_name"] = name.lower()
            
            if location:
                # Add location filter if provided
                filter_dict["state"] = location.upper()
            
            # Log the filter for debugging
            logger.info(f"Exact match filter: {json.dumps(filter_dict)}")
            
            # Query with exact match
            exact_response = self.index.query(
                vector=dummy_vector,
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            results = []
            
            # Process exact matches
            if exact_response.matches and len(exact_response.matches) > 0:
                logger.info(f"Found {len(exact_response.matches)} exact matches")
                for match in exact_response.matches:
                    result = {
                        "id": match.id,
                        "score": 1.0,  # Perfect match
                        "metadata": match.metadata,
                        "match_type": "exact"
                    }
                    results.append(result)
                return results
                
            # If no exact match, do a broad search and filter manually
            logger.info("No exact matches, trying broad search")
            
            # Get a larger set of potential matches
            broad_response = self.index.query(
                vector=dummy_vector,
                top_k=100,  # Get more to filter
                include_metadata=True
            )
            
            # Manually filter and score based on the name
            name_parts = name.lower().split()
            filtered_matches = []
            
            for match in broad_response.matches:
                cu_name = match.metadata.get('cu_name', '').lower()
                
                # Skip if no name
                if not cu_name:
                    continue
                    
                # Calculate similarity based on word overlap
                common_words = sum(1 for part in name_parts if part in cu_name)
                name_length = len(name_parts)
                
                # Basic similarity score
                similarity = common_words / name_length if name_length > 0 else 0
                
                # Boost for better matches
                # Full string contains
                if name.lower() in cu_name:
                    similarity += 0.3
                # First word match
                elif name_parts and cu_name.startswith(name_parts[0]):
                    similarity += 0.2
                # Any significant word match (4+ chars)
                for part in name_parts:
                    if len(part) >= 4 and part in cu_name:
                        similarity += 0.1
                
                # Cap at 1.0
                similarity = min(1.0, similarity)
                
                # Only include reasonable matches
                if similarity >= 0.2:
                    filtered_matches.append({
                        "id": match.id,
                        "score": similarity,
                        "metadata": match.metadata,
                        "match_type": "broad"
                    })
            
            # Sort by score
            filtered_matches.sort(key=lambda x: x["score"], reverse=True)
            logger.info(f"Broad search found {len(filtered_matches)} potential matches")
            
            # Return the top matches
            return filtered_matches[:top_k]
                
        except Exception as e:
            logger.error(f"Error searching Pinecone by name: {str(e)}")
            # Log full exception details
            import traceback
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise

# Try to initialize the Pinecone client
try:
    logger.info("Attempting to initialize Pinecone client...")
    pinecone_client = PineconeClient()
    logger.info("Pinecone client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {str(e)}")
    pinecone_client = None
    logger.warning("Application will continue with Pinecone functionality disabled.") 