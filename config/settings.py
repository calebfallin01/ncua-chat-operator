"""Configuration settings for the NCUA Credit Union Data Query System."""

import os
from dotenv import load_dotenv
from pydantic import BaseSettings

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    
    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_api_key: str = os.getenv("SUPABASE_API_KEY", "")
    
    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_environment: str = os.getenv("PINECONE_ENVIRONMENT", "")
    pinecone_index: str = os.getenv("PINECONE_INDEX", "")
    
    # OpenAI model settings
    model_name: str = "gpt-3.5-turbo-0125"
    temperature: float = 0.0
    max_tokens: int = 1000
    
    # Application settings
    max_results: int = 5
    similarity_threshold: float = 0.7
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"


# Create settings instance
settings = Settings()

# Validate required settings
def validate_settings():
    """Validate that all required settings are provided."""
    missing = []
    
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.supabase_url:
        missing.append("SUPABASE_URL")
    if not settings.supabase_api_key:
        missing.append("SUPABASE_API_KEY")
    if not settings.pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if not settings.pinecone_environment:
        missing.append("PINECONE_ENVIRONMENT")
    if not settings.pinecone_index:
        missing.append("PINECONE_INDEX")
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# Export settings
__all__ = ["settings", "validate_settings"] 