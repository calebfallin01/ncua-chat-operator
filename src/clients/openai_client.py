"""OpenAI client wrapper for the NCUA Credit Union Query System."""

import logging
import sys
import os
import openai
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Add the parent directory to the path so we can import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config.settings import settings

# Configure logger
logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        # Set the API key
        openai.api_key = settings.openai_api_key
        self.model = settings.model_name
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def analyze_query(self, query: str) -> dict:
        """
        Analyze a user query to determine the intent and extraction parameters.
        
        Args:
            query: The user's natural language query
            
        Returns:
            A dictionary containing extracted entities and query type
        """
        system_prompt = """
        You are a specialized AI trained to analyze queries about credit unions. 
        Your task is to extract relevant search parameters from the query.
        
        Extract the following information:
        1. Credit union name (if provided)
        2. State or location information (if provided)
        3. Any specific attributes the user is asking about
        4. The data being requested (e.g., "total assets", "asset size", "loan amount")
        
        Return a JSON with the following structure:
        {
            "credit_union_name": "extracted name or null if not provided",
            "location": "extracted location or null if not provided",
            "attributes": ["list", "of", "attributes", "user", "is", "asking", "about"],
            "query_type": "general_info|financial_data|location|contact|services",
            "data_queries": ["list", "of", "specific", "data", "being", "requested"]
        }
        
        The "data_queries" field is critical - these are the specific data points the user is asking about,
        which will be used to search in the account description table (acctdesc_2024_12). Examples:
        - For "What is the asset size of Navy Federal Credit Union?", include "asset size" and "total assets"
        - For "How many loans did XYZ Credit Union issue?", include "loans", "total loans", "loan amount"
        """
        
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.0,
                max_tokens=500
            )
            
            # Extract and parse the content
            content = response['choices'][0]['message']['content']
            logger.debug(f"OpenAI query analysis: {content}")
            
            return content
        except Exception as e:
            logger.error(f"Error analyzing query with OpenAI: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def generate_embedding(self, text: str) -> list:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A list of floats representing the embedding vector
        """
        try:
            # Use text-embedding-3-large which generates 3072-dimensional vectors
            # This matches the dimension of our Pinecone index
            response = await openai.Embedding.acreate(
                model="text-embedding-3-large",
                input=text,
            )
            
            embedding = response['data'][0]['embedding']
            logger.info(f"Generated embedding with dimension: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Fallback to create a dummy vector with the right dimensions
            logger.warning("Using fallback dummy vector with dimension 3072")
            return [0.0] * 3072

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    async def generate_response(self, query: str, cu_data: dict) -> str:
        """
        Generate a natural language response to the user's query based on retrieved data.
        
        Args:
            query: The original user query
            cu_data: Credit union data retrieved from databases
            
        Returns:
            A natural language response to the user's query
        """
        system_prompt = """
        You are a helpful assistant that provides information about credit unions.
        Use only the information provided to answer the question.
        If the information is not available in the provided data, say so clearly.
        Do not make up or infer information not present in the data.
        
        Format your response in a clear, concise manner. Be brief and direct.
        
        When handling financial data:
        1. Format currency values in a readable way (e.g., "$180,813,031,049" instead of "180813031049")
        2. Format percentages with two decimal places when appropriate
        3. If you see a value in a nested dictionary under "financial_data.*", this is the PRIMARY data
           to answer the query with. Prioritize this information.
        
        The data is structured as follows:
        - Metadata fields are prefixed with "metadata."
        - Financial data fields are prefixed with "financial_data."
        - Within financial data objects, the "value" field contains the actual numeric value
        """
        
        try:
            # Pre-process the data to make it more usable for the model
            processed_data = {}
            financial_values = {}
            
            # Extract metadata and organize it
            for key, value in cu_data.items():
                if key.startswith("metadata."):
                    # Clean up metadata keys
                    clean_key = key.replace("metadata.", "")
                    processed_data[clean_key] = value
                elif key.startswith("financial_data."):
                    # Process financial data more carefully
                    if isinstance(value, dict) and "value" in value:
                        # Extract the key name and value for financial data
                        data_key = key.replace("financial_data.", "")
                        financial_values[data_key] = value["value"]
                        # Also add the full object for context
                        processed_data[key] = value
            
            # Ensure cu_name is available
            if "cu_name" in processed_data:
                cu_name = processed_data["cu_name"]
            else:
                # Try to find cu_name in metadata
                cu_name = "the requested credit union"
                for key, value in cu_data.items():
                    if key.endswith("cu_name") or key.endswith("name"):
                        cu_name = value
                        break
            
            # Format financial values for easier consumption
            formatted_financial = {}
            for key, value in financial_values.items():
                # Format currency values
                if isinstance(value, (int, float)) and value > 1000:
                    formatted_financial[key] = "${:,}".format(value)
                else:
                    formatted_financial[key] = value
            
            # Create a simplified data structure for the model
            simplified_data = {
                "credit_union_name": cu_name,
                "metadata": {k: v for k, v in processed_data.items() if not k.startswith("financial_data.")},
                "financial_data": financial_values,
                "formatted_values": formatted_financial
            }
            
            # Convert to a clear string representation with explicit sections
            data_str = f"""
            CREDIT UNION: {cu_name}
            
            BASIC INFO:
            {json.dumps({k: v for k, v in simplified_data["metadata"].items() if k != "cu_name"}, indent=2)}
            
            FINANCIAL DATA:
            {json.dumps(simplified_data["formatted_values"], indent=2)}
            
            RAW DATA (if needed):
            {json.dumps(cu_data, indent=2)}
            """
            
            user_message = f"""
            Question: {query}
            
            Available data:
            {data_str}
            """
            
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.5,  # Slightly lower temperature for more focused responses
                max_tokens=settings.max_tokens
            )
            
            return response['choices'][0]['message']['content']
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

# Create a singleton instance
openai_client = OpenAIClient() 