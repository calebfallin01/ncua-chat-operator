import os
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for interacting with OpenAI API."""
    
    def __init__(self):
        """Initialize the OpenAI client."""
        # Get API key from environment variable
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize the OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = os.environ.get("OPENAI_MODEL", "text-embedding-3-large")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def create_embedding(self, text: str) -> list:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to generate an embedding for
            
        Returns:
            A list of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            
            embedding = response.data[0].embedding
            logger.info(f"Generated embedding with dimension: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None 