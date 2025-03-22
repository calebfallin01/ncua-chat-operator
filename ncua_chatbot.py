#!/usr/bin/env python3
"""
NCUA Chatbot - An interactive chatbot for querying NCUA credit union data

This chatbot:
1. Interprets natural language queries about credit unions
2. Finds the right credit union using vector search
3. Retrieves financial data using the credit union number
4. Extracts the relevant information from the database results
5. Returns a concise answer to the user
"""

import os
import sys
import json
import asyncio
import subprocess
import logging
import argparse
import traceback
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
import openai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
import re
from utils import MetricsCache, cached

# Parse command line arguments
parser = argparse.ArgumentParser(description="NCUA Credit Union Chatbot")
parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
parser.add_argument("--demo", action="store_true", help="Run in demo mode with example questions")
parser.add_argument("--test-query", type=str, help="Test the query_financial_data method with the given cu_number")
args = parser.parse_args()

# Set debug mode
DEBUG_MODE = args.debug
DEMO_MODE = args.demo
TEST_QUERY_MODE = args.test_query is not None

# Print immediate feedback only in debug mode
if DEBUG_MODE:
    print("Starting NCUA Chatbot script...")
    print("Debug mode: ENABLED")
if DEMO_MODE:
    print("Demo mode: ENABLED")

# Configure logging to file and console
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ncua_chatbot.log"),
        # Only output to console in debug mode
        logging.StreamHandler() if DEBUG_MODE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Logger initialized")

# Set up Rich console for pretty output
console = Console()

# Check for environment file quietly
env_file = ".env"
if not os.path.exists(env_file) and DEBUG_MODE:
    print(f"Warning: .env file not found at {env_file}. Make sure you have the required API keys set.")

class ConversationContext:
    """Class to maintain conversation context and state across interactions."""
    
    def __init__(self):
        """Initialize the conversation context."""
        # Currently active credit union info
        self.current_cu_info = None
        
        # Currently active financial data
        self.current_financial_data = None
        
        # Track all credit unions mentioned in the conversation
        self.mentioned_credit_unions = {}
        
        # Track the last query about each credit union
        self.last_queries = {}
        
        # Track time when last mentioned to implement context decay
        self.last_mentioned_timestamp = {}
    
    def set_active_credit_union(self, cu_info: Dict[str, Any], financial_data: Optional[Dict[str, Any]] = None):
        """
        Set the currently active credit union in the conversation.
        
        Args:
            cu_info: Credit union information including name and cu_number
            financial_data: Optional financial data for the credit union
        """
        if not cu_info or "cu_number" not in cu_info:
            return False
            
        # Store current credit union info
        self.current_cu_info = cu_info
        
        # Store cu_number for easier access
        cu_number = cu_info.get("cu_number")
        
        # Store financial data if provided
        if financial_data:
            self.current_financial_data = financial_data
        
        # Add to mentioned credit unions
        self.mentioned_credit_unions[cu_number] = cu_info
        
        # Update last mentioned timestamp
        import time
        self.last_mentioned_timestamp[cu_number] = time.time()
        
        return True
    
    def get_active_credit_union(self) -> Optional[Dict[str, Any]]:
        """Get the currently active credit union info."""
        return self.current_cu_info
        
    def get_active_financial_data(self) -> Optional[Dict[str, Any]]:
        """Get the currently active financial data."""
        return self.current_financial_data
    
    def get_credit_union_by_number(self, cu_number: str) -> Optional[Dict[str, Any]]:
        """Get credit union info by cu_number if it was previously mentioned."""
        return self.mentioned_credit_unions.get(cu_number)
    
    def has_context(self) -> bool:
        """Check if there is an active conversation context."""
        return self.current_cu_info is not None
    
    def record_query(self, cu_number: str, query: str):
        """Record a query for a specific credit union."""
        self.last_queries[cu_number] = query
        
        # Update last mentioned timestamp
        import time
        self.last_mentioned_timestamp[cu_number] = time.time()
    
    def clear_context(self):
        """Clear the current conversation context."""
        self.current_cu_info = None
        self.current_financial_data = None
    
    def get_most_recent_credit_union(self) -> Optional[Dict[str, Any]]:
        """Get the most recently mentioned credit union based on timestamp."""
        if not self.last_mentioned_timestamp:
            return None
            
        # Find credit union with the most recent timestamp
        most_recent_cu_number = max(
            self.last_mentioned_timestamp.keys(),
            key=lambda cu_num: self.last_mentioned_timestamp[cu_num]
        )
        
        return self.mentioned_credit_unions.get(most_recent_cu_number)

class NCUAChatbot:
    """Interactive chatbot for querying NCUA credit union data."""
    
    def __init__(self):
        """Initialize the chatbot with API keys and configuration."""
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variable
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Set the API key
        openai.api_key = self.openai_api_key
        
        # Configuration
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4-turbo")
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize conversation context tracker
        self.context = ConversationContext()
        
        # Initialize cache for financial data
        self.metrics_cache = MetricsCache(cache_dir=".cache/metrics", ttl=3600)
        logger.info("Initialized metrics cache")
        
        # Map from tablename field to actual table name (e.g., "FS220A" to "fs220a_2024_12")
        self.tablename_to_table = {
            "FS220": "fs220_2024_12",
            "FS220A": "fs220a_2024_12",
            "FS220B": "fs220b_2024_12", 
            "FS220C": "fs220c_2024_12",
            "FS220D": "fs220d_2024_12",
            "FS220G": "fs220g_2024_12",
            "FS220H": "fs220h_2024_12",
            "FS220I": "fs220i_2024_12",
            "FS220J": "fs220j_2024_12",
            "FS220K": "fs220k_2024_12",
            "FS220L": "fs220l_2024_12",
            "FS220M": "fs220m_2024_12",
            "FS220N": "fs220n_2024_12",
            "FS220P": "fs220p_2024_12",
            "FS220Q": "fs220q_2024_12",
            "FS220R": "fs220r_2024_12",
            "FS220S": "fs220s_2024_12"
        }
        
        # System message defining the chatbot's role
        self.system_message = {
            "role": "system",
            "content": """
You are a financial assistant specialized in credit union data from the National Credit Union Administration (NCUA).
Your job is to help users find information about credit unions by querying the NCUA database.

You have access to:
1. A vector search to find credit unions by name (query_credit_unions.py)
2. A database query tool to get financial data for a specific credit union (query_all_tables.py)

You maintain conversation context about which credit union the user is discussing. This means:
- When a user first asks about a specific credit union, you store that credit union's information
- For follow-up questions about the same credit union, you can use pronouns like "they", "them", "their", "it", etc.
- If a user asks "what is their asset size?" or similar questions, you know which credit union "their" refers to
- You don't need to re-query the database for each question about the same credit union

When a user asks about a specific credit union, you need to:
1. Identify the credit union name and any location context from the user's question
2. Find the credit union using vector search
3. Get its financial data using the credit union number (cu_number)
4. Extract the specific information requested from the financial data
5. Provide a concise, informative answer

For follow-up questions about the same credit union, you can use the existing context.

DO NOT make up information. If you cannot find the data, tell the user.
"""
        }
        
        # Create status indicators for debug and demo modes - only show in debug mode
        if DEBUG_MODE:
            mode_indicators = []
            if DEBUG_MODE:
                mode_indicators.append("[bold yellow]DEBUG MODE[/bold yellow]")
            if DEMO_MODE:
                mode_indicators.append("[bold blue]DEMO MODE[/bold blue]")
            
            mode_status = "\n".join(mode_indicators) + "\n" if mode_indicators else ""
            
            console.print(Panel.fit(
                f"{mode_status}[bold blue]NCUA Credit Union Chatbot[/bold blue]\n"
                "Ask questions about credit unions and their financial data.",
                title="Welcome"
            ))
    
    def extract_credit_union_info(self, user_message: str) -> Tuple[str, Optional[Dict[str, str]]]:
        """
        Extract credit union name and location context from user message.
        
        Args:
            user_message: User's query
            
        Returns:
            Tuple of (credit_union_name, location_context)
        """
        # Using OpenAI to help extract entities from the user's query
        messages = [
            {
                "role": "system",
                "content": """
You are an expert in extracting information from text. 
Extract the credit union name and any location information from the user's query.
Return your response as a JSON object with these fields:
- credit_union_name: The name of the credit union (or null if not present)
- city: The city name (or null if not present)
- state_code: The two-letter state code like "PA" (or null if not present)
- state_full: The full state name like "Pennsylvania" (or null if not present)

Do not include explanations, just return the JSON object.
"""
            },
            {
                "role": "user",
                "content": user_message
            }
        ]
        
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-0125",  # Using a simpler model for entity extraction
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Extract credit union name
            credit_union_name = result.get("credit_union_name")
            
            # Extract location context if available
            location_context = {}
            if result.get("city"):
                location_context["city"] = result["city"]
            if result.get("state_code"):
                location_context["state_code"] = result["state_code"]
            if result.get("state_full"):
                location_context["state_full"] = result["state_full"]
            
            if DEBUG_MODE and credit_union_name:
                console.print(f"[dim]Extracted credit union name: {credit_union_name}[/dim]")
                if location_context:
                    console.print(f"[dim]Extracted location context: {location_context}[/dim]")
                
            return credit_union_name, location_context if location_context else None
            
        except Exception as e:
            logger.error(f"Error extracting credit union info with OpenAI: {str(e)}")
            logger.info("Using fallback method for entity extraction")
            
            # Fallback method: Use rule-based extraction for common patterns
            credit_union_name = self._fallback_extract_cu_name(user_message)
            location_context = self._fallback_extract_location(user_message)
            
            if credit_union_name:
                logger.info(f"Fallback extraction found credit union: {credit_union_name}")
            
            return credit_union_name, location_context if location_context else None
    
    def _fallback_extract_cu_name(self, text: str) -> Optional[str]:
        """
        Extract credit union name using rule-based patterns.
        
        Args:
            text: Text to extract from
            
        Returns:
            Credit union name if found, None otherwise
        """
        # Common credit union names and abbreviations
        common_cu_patterns = [
            r"navy\s*f(?:ederal)?(?:\s*c(?:redit)?\s*u(?:nion)?)?",  # Navy Federal Credit Union
            r"(?:navy|navyfcu|navy\s*fcu)",  # Navy FCU abbreviations
            r"pentagon\s*f(?:ederal)?(?:\s*c(?:redit)?\s*u(?:nion)?)?",  # Pentagon Federal Credit Union
            r"(?:penfed|pentagon\s*fcu)",  # PenFed abbreviations
            r"members\s*(?:1st|first)(?:\s*c(?:redit)?\s*u(?:nion)?)?",  # Members 1st Credit Union
            r"state\s*employees(?:\s*c(?:redit)?\s*u(?:nion)?)?",  # State Employees Credit Union
            r"secu",  # SECU abbreviation
            r"(\w+(?:\s+\w+){0,4})\s+(?:credit\s+union|cu|c\.u\.|federal\s+credit\s+union|fcu|f\.c\.u\.)",  # Generic pattern
            r"(\w+(?:\s+\w+){0,4})\s+(?:federal\s+savings)",  # Federal savings
        ]
        
        # Look for matches in the text
        text_lower = text.lower()
        
        # First try the common patterns
        for pattern in common_cu_patterns:
            match = re.search(pattern, text_lower)
            if match:
                # If the pattern has a capture group, use that, otherwise use the whole match
                if match.groups():
                    result = match.group(1).strip()
                else:
                    result = match.group(0).strip()
                
                # Clean up and format the result
                result = result.strip()
                
                # Convert to title case for proper names
                if len(result) > 5:  # Only titlecase longer names, not abbreviations
                    result = ' '.join(word.capitalize() for word in result.split())
                else:
                    result = result.upper()
                
                return result
                
        # Fallback for "of X" patterns (e.g., "What is the asset size of Navy Federal?")
        of_pattern = r"(?:of|for|about|at|from)\s+([A-Za-z0-9\s&']+(?:\s*(?:credit\s*union|cu|c\.u\.|federal|fcu|f\.c\.u\.))?)"
        match = re.search(of_pattern, text_lower)
        if match:
            result = match.group(1).strip()
            if result and len(result) > 2:  # Avoid capturing "of the" etc.
                return ' '.join(word.capitalize() for word in result.split())
            
        return None
    
    def _fallback_extract_location(self, text: str) -> Optional[Dict[str, str]]:
        """
        Extract location information using rule-based patterns.
        
        Args:
            text: Text to extract from
            
        Returns:
            Dictionary with location information if found, None otherwise
        """
        # Dictionary mapping state codes to full names
        state_map = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia'
        }
        
        # Create reverse map from full names to codes
        full_to_code = {v.lower(): k for k, v in state_map.items()}
        
        location_context = {}
        
        # Look for state codes (e.g., "PA", "CA")
        state_code_pattern = r'\b(' + '|'.join(state_map.keys()) + r')\b'
        state_code_match = re.search(state_code_pattern, text.upper())
        if state_code_match:
            location_context["state_code"] = state_code_match.group(1)
        
        # Look for full state names (e.g., "Pennsylvania", "California")
        state_full_pattern = r'\b(' + '|'.join(re.escape(name) for name in state_map.values()) + r')\b'
        state_full_match = re.search(state_full_pattern, text, re.IGNORECASE)
        if state_full_match:
            state_full = state_full_match.group(1)
            location_context["state_full"] = state_full
            # Add state code if not already present
            if "state_code" not in location_context:
                location_context["state_code"] = full_to_code.get(state_full.lower())
            
        # Look for major cities - this could be expanded with more cities
        city_pattern = r'\b(?:in|at|of|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        city_match = re.search(city_pattern, text)
        if city_match:
            potential_city = city_match.group(1)
            # Make sure it's not a state name or other non-city word
            if potential_city.lower() not in full_to_code and potential_city not in ["Bank", "Credit", "Union", "Federal"]:
                location_context["city"] = potential_city
        
        return location_context if location_context else None
    
    def find_credit_union(self, credit_union_name: str, location_context: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Find a credit union using vector search.
        
        Args:
            credit_union_name: Name of the credit union to search for
            location_context: Optional location context (city, state_code, state_full)
            
        Returns:
            Credit union data including cu_number if found, None otherwise
        """
        try:
            cmd = ["python", "query_credit_unions.py"]
            
            # Clean the credit union name - remove quotes, special chars that might interfere
            if credit_union_name:
                # Split the credit union name into individual words
                words = credit_union_name.split()
                # Add each word as a separate argument
                cmd.extend(words)
            else:
                logger.error("No credit union name provided to find_credit_union")
                return None
            
            # Add location context parameters if provided
            state_code = None
            if location_context:
                if location_context.get("city"):
                    cmd.extend(["--city", location_context["city"]])
                if location_context.get("state_code"):
                    state_code = location_context["state_code"]
                    cmd.extend(["--state-code", location_context["state_code"]])
                if location_context.get("state_full"):
                    cmd.extend(["--state", location_context["state_full"]])
            
            # Add other parameters - using --all-results gives us the full output
            cmd.extend(["--min-score", "0.7", "--force-min-score", "--all-results"])
            
            if DEBUG_MODE:
                console.print(f"[bold]Running:[/bold] {' '.join(cmd)}")
            
            # Run with output capture
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error running query_credit_unions.py: {result.stderr}")
                return None
            
            # Process the output to extract the credit union details
            output = result.stdout
            
            # Parse all matches from the output
            matches = []
            current_match = {}
            in_match = False
            
            for line in output.split('\n'):
                if line.startswith('--') and '--' in line:
                    if current_match and in_match:
                        matches.append(current_match)
                        current_match = {}
                    in_match = False
                    
                elif line.strip().startswith(str(len(matches) + 1) + '.'):
                    in_match = True
                    current_match = {'rank': len(matches) + 1}
                    # Extract credit union name
                    cu_name_match = re.search(r'Credit Union:\s*(.*?)(?:\s*\(|$)', line)
                    if cu_name_match:
                        current_match['cu_name'] = cu_name_match.group(1).strip()
                    # Extract match score
                    score_match = re.search(r'Score:\s*([\d.]+)', line)
                    if score_match:
                        current_match['score'] = float(score_match.group(1))
                        
                elif in_match and ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower().replace(' ', '_')
                    value = value.strip()
                    current_match[key] = value
            
            # Add the last match if there is one
            if current_match and in_match:
                matches.append(current_match)
            
            # Select best match, prioritizing state if provided
            best_match = None
            
            if state_code and matches:
                # First try to find a match with the correct state
                state_matches = []
                for match in matches:
                    # Check if there's a state_code or state in the match
                    match_state = match.get('state') or match.get('state_code')
                    if match_state and state_code.upper() in match_state.upper():
                        state_matches.append(match)
                
                if state_matches:
                    # Sort by score
                    best_match = sorted(state_matches, key=lambda x: x.get('score', 0), reverse=True)[0]
                    logger.info(f"Found best match with correct state: {best_match.get('cu_name')}")
            
            # If no state match or no state provided, use highest score
            if not best_match and matches:
                best_match = sorted(matches, key=lambda x: x.get('score', 0), reverse=True)[0]
                logger.info(f"Using highest score match: {best_match.get('cu_name')}")
            
            if not best_match:
                logger.warning("No matches found in vector search results")
                return None
            
            # Extract credit union metadata from best match
            cu_metadata = {}
            
            # Get basic info from match
            cu_metadata["cu_name"] = best_match.get('cu_name')
            
            # Extract cu_number if available
            cu_number = best_match.get('cu_number')
            if cu_number:
                # Normalize the credit union number
                if '.' in cu_number:
                    try:
                        cu_metadata["cu_number"] = str(int(float(cu_number)))
                    except ValueError:
                        cu_metadata["cu_number"] = cu_number
                else:
                    cu_metadata["cu_number"] = cu_number
            
            # Check if we have a cu_number, if not, check search results again
            if "cu_number" not in cu_metadata:
                # Look for cu_number pattern in the output
                cu_number_match = re.search(r"cu_number:\s*(\d+(?:\.\d+)?)", output)
                if cu_number_match:
                    raw_cu_number = cu_number_match.group(1)
                    # Normalize the credit union number by removing decimal point
                    if '.' in raw_cu_number:
                        try:
                            cu_metadata["cu_number"] = str(int(float(raw_cu_number)))
                        except ValueError:
                            cu_metadata["cu_number"] = raw_cu_number
                    else:
                        cu_metadata["cu_number"] = raw_cu_number
            
            # Final check - do we have what we need?
            if "cu_number" in cu_metadata and "cu_name" in cu_metadata:
                logger.info(f"Successfully found credit union: {cu_metadata.get('cu_name', 'Unknown')} with number {cu_metadata['cu_number']}")
                return cu_metadata
            else:
                logger.warning("No credit union number found in the selected match")
                return None
                
        except Exception as e:
            logger.error(f"Error finding credit union: {str(e)}")
            return None

    def query_financial_data(self, cu_number: str) -> Optional[Dict[str, Any]]:
        """
        Query financial data for a credit union.
        
        Args:
            cu_number: Credit union number
            
        Returns:
            Dictionary of financial data if found, None otherwise
        """
        try:
            # Normalize the credit union number - remove any decimal part since it's always .0
            # This ensures compatibility with query_all_tables.py which expects integers
            normalized_cu_number = cu_number
            if isinstance(cu_number, str) and '.' in cu_number:
                try:
                    # Convert string with decimal like "5536.0" to integer string "5536"
                    normalized_cu_number = str(int(float(cu_number)))
                    logger.info(f"Normalized cu_number from {cu_number} to {normalized_cu_number}")
                except ValueError:
                    logger.warning(f"Could not normalize cu_number: {cu_number}")
            
            # Check cache first
            cached_data = self.metrics_cache.get(normalized_cu_number)
            if cached_data is not None:
                logger.info(f"Using cached financial data for CU #{normalized_cu_number}")
                return cached_data
            
            # Add debug logging for the credit union number
            logger.info(f"Querying financial data for CU #{normalized_cu_number}, original: {cu_number}")
            
            # Create a temporary file for output
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
                output_file = tmp_file.name
                logger.info(f"Created temporary output file: {output_file}")
            
            # First, try to get a readable format to better understand the data
            if DEBUG_MODE:
                console.print("[dim]First querying in readable format to understand the data structure...[/dim]")
                readable_cmd = ["python", "query_all_tables.py", normalized_cu_number, "--format", "readable"]
                logger.info(f"Running readable query command: {' '.join(readable_cmd)}")
                readable_result = subprocess.run(readable_cmd, capture_output=True, text=True)
                if readable_result.returncode == 0 and readable_result.stdout:
                    logger.info("Readable query succeeded")
                    console.print(Panel(readable_result.stdout[:2000] + ("..." if len(readable_result.stdout) > 2000 else ""), 
                                      title=f"Sample Readable Data for CU #{normalized_cu_number}"))
                else:
                    logger.warning(f"Readable query failed: {readable_result.stderr}")
            
            # Now get the actual JSON data
            cmd = ["python", "query_all_tables.py", normalized_cu_number, "--output", output_file]
            
            logger.info(f"Running database query command: {' '.join(cmd)}")
            
            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error running query_all_tables.py: {result.stderr}")
                logger.error(f"STDOUT: {result.stdout[:1000]}")
                return None
            
            # Read the output file
            try:
                # Check if the file exists and has content
                if not os.path.exists(output_file):
                    logger.error(f"Output file {output_file} does not exist")
                    return None
                
                file_size = os.path.getsize(output_file)
                logger.info(f"Output file size: {file_size} bytes")
                
                if file_size == 0:
                    logger.error(f"Output file {output_file} is empty")
                    return None
                
                with open(output_file, 'r') as f:
                    file_content = f.read()
                    logger.info(f"Read {len(file_content)} characters from output file")
                    
                    if not file_content.strip():
                        logger.error("Output file content is empty or whitespace only")
                        return None
                    
                    try:
                        financial_data = json.loads(file_content)
                        logger.info(f"Successfully parsed JSON from output file")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse JSON: {str(e)}")
                        logger.error(f"First 500 chars of content: {file_content[:500]}")
                        return None
                
                # Clean up the temporary file
                os.unlink(output_file)
                logger.info(f"Deleted temporary output file")
                
                # Check if we have data
                results = financial_data.get("results", {})
                logger.info(f"Found {len(results)} tables in results")
                
                # Store the results in the cache
                self.metrics_cache.set(normalized_cu_number, financial_data)
                logger.info(f"Cached financial data for CU #{normalized_cu_number}")
                
                # Just return the raw data - do not enrich or process it
                return financial_data
                
            except Exception as e:
                logger.error(f"Error reading financial data: {str(e)}")
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"Error querying financial data: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    def identify_required_metrics(self, user_query: str) -> Dict[str, List[str]]:
        """
        Identify which financial metrics and fields are needed to answer the query.
        
        Args:
            user_query: The user's query about a credit union
            
        Returns:
            Dictionary mapping metric categories to lists of field names to look for
        """
        query_lower = user_query.lower()
        required_metrics = {}
        
        # Common financial metric field patterns
        metric_patterns = {
            'assets': {
                'fields': ['acct_010', 'ACCT_010', '010', 'acct_799', 'ACCT_799', '799'],
                'tables': ['fs220_2024_12'],
                'description': 'Total Assets'
            },
            'net_worth': {
                'fields': ['acct_997', 'ACCT_997', '997', 'acct_998', 'ACCT_998', '998'],
                'tables': ['fs220_2024_12'],
                'description': 'Net Worth'
            },
            'net_income': {
                'fields': ['acct_661a', 'ACCT_661A', '661a', '661A', 'acct_661', 'ACCT_661', '661'],
                'tables': ['fs220a_2024_12', 'fs220_2024_12'],
                'description': 'Net Income'
            },
            'shares': {
                'fields': ['acct_018', 'ACCT_018', '018', 'acct_013', 'ACCT_013', '013'],
                'tables': ['fs220_2024_12'],
                'description': 'Total Shares and Deposits'
            },
            'loans': {
                'fields': ['acct_025b', 'ACCT_025B', '025b', '025B', 'acct_025', 'ACCT_025', '025'],
                'tables': ['fs220_2024_12', 'fs220b_2024_12'],
                'description': 'Total Loans'
            },
            'members': {
                'fields': ['acct_083', 'ACCT_083', '083', 'acct_084', 'ACCT_084', '084'],
                'tables': ['fs220_2024_12'],
                'description': 'Number of Members'
            }
        }
        
        # Check for mentions of specific metrics
        if any(term in query_lower for term in ['asset', 'assets', 'size', 'how big', 'how large']):
            required_metrics['assets'] = metric_patterns['assets']
            
        if any(term in query_lower for term in ['net worth', 'networth', 'equity', 'capital']):
            required_metrics['net_worth'] = metric_patterns['net_worth']
            
        if any(term in query_lower for term in ['income', 'earnings', 'profit', 'make', 'earn', 'revenue']):
            required_metrics['net_income'] = metric_patterns['net_income']
            
        if any(term in query_lower for term in ['share', 'shares', 'deposit', 'deposits', 'saving', 'savings']):
            required_metrics['shares'] = metric_patterns['shares']
            
        if any(term in query_lower for term in ['loan', 'loans', 'lend', 'lending', 'borrow', 'debt']):
            required_metrics['loans'] = metric_patterns['loans']
            
        if any(term in query_lower for term in ['member', 'members', 'customer', 'customers', 'membership']):
            required_metrics['members'] = metric_patterns['members']
        
        # If we couldn't identify specific metrics, include all common ones
        if not required_metrics:
            # For general questions, include main financial metrics
            required_metrics = {
                'assets': metric_patterns['assets'],
                'net_income': metric_patterns['net_income'],
                'members': metric_patterns['members']
            }
        
        return required_metrics
    
    def extract_required_fields(self, financial_data: Dict[str, Any], required_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract only the required fields from the financial data based on identified metrics.
        
        Args:
            financial_data: The full financial data dictionary
            required_metrics: Dictionary of required metrics with fields to look for
            
        Returns:
            Filtered financial data with only the required fields
        """
        results = financial_data.get("results", {})
        filtered_data = {
            "credit_union_info": financial_data.get("credit_union_info", {}),
            "results": {}
        }
        
        # Check if this is an uncommon query that might need extra or all data
        is_uncommon_query = financial_data.get("query_type") == "uncommon_information"
        if is_uncommon_query:
            # For uncommon queries, include all data but limit records per table for token constraints
            filtered_data["query_type"] = "uncommon_information"
            filtered_data["detected_keywords"] = financial_data.get("detected_keywords", [])
            
            # Set a more reasonable maximum number of records based on the table type
            for table_name, records in results.items():
                if not records:
                    continue
                    
                # Include more records for tables likely to contain the information
                if any(keyword in table_name.lower() for keyword in ["branch", "atm", "location", "tradename"]):
                    # These tables typically have multiple records that are all relevant
                    filtered_data["results"][table_name] = records[:20]  # Include up to 20 records
                elif "foicu" in table_name.lower() or "foicudes" in table_name.lower():
                    # Important tables for credit union metadata
                    filtered_data["results"][table_name] = records
                else:
                    # For other tables, include fewer records
                    filtered_data["results"][table_name] = records[:3]  # Just include up to 3 records
            
            return filtered_data
        
        # For each metric, get the required tables and fields
        needed_tables = set()
        for metric_info in required_metrics.values():
            needed_tables.update(metric_info.get('tables', []))
        
        # Always include these key tables if available
        core_tables = ['fs220_2024_12', 'fs220a_2024_12', 'foicu_2024_12']
        for table in core_tables:
            if table in results:
                needed_tables.add(table)
                
        # Include any table that has the word "branch" in its name for location questions
        if any("branch" in metric for metric in required_metrics.keys()):
            for table_name in results.keys():
                if "branch" in table_name.lower():
                    needed_tables.add(table_name)
        
        # Extract only the tables we need
        for table_name in needed_tables:
            if table_name in results:
                # Get the records from this table
                records = results[table_name]
                filtered_records = []
                
                for record in records:
                    # Create a filtered record with only specific fields
                    filtered_record = {
                        'id': record.get('id'),
                        'cu_number': record.get('cu_number'),
                        'cycle_date': record.get('cycle_date'),
                        'join_number': record.get('join_number')
                    }
                    
                    # Add the specific fields needed for each metric
                    for metric_name, metric_info in required_metrics.items():
                        field_patterns = metric_info.get('fields', [])
                        for field in field_patterns:
                            if field in record:
                                filtered_record[field] = record[field]
                    
                    # Add any field that starts with 'acct_' and has a non-null, non-zero value
                    # This ensures we don't miss important account fields
                    for field, value in record.items():
                        if (field.lower().startswith('acct_') or field.isdigit()) and value is not None and value != 0:
                            filtered_record[field] = value
                            
                        # For non-financial tables like branch information, include all fields
                        if table_name.lower() in ['credit_union_branch_information_2024_12', 'atm_locations_2024_12', 'tradenames_2024_12']:
                            filtered_record[field] = value
                    
                    filtered_records.append(filtered_record)
                
                # Add the filtered records to the result
                filtered_data['results'][table_name] = filtered_records
        
        # Add metric information to help the model
        filtered_data['required_metrics'] = {
            metric: info.get('description', metric) 
            for metric, info in required_metrics.items()
        }
        
        return filtered_data

    def interpret_results(self, user_query: str, credit_union_info: Dict[str, Any], financial_data: Dict[str, Any]) -> str:
        """
        Interpret the financial data to answer the user's query.
        Extract key data points locally and send only those to OpenAI rather than the full dataset.
        
        Args:
            user_query: The original user query
            credit_union_info: Information about the credit union
            financial_data: Financial data for the credit union
            
        Returns:
            Answer to the user's query
        """
        try:
            # Check if financial_data is valid
            if not financial_data or not financial_data.get("results"):
                logger.error(f"No financial data found for interpretation: {financial_data}")
                cu_name = self.format_credit_union_name(credit_union_info.get("cu_name", "this credit union"))
                return f"I found information about {cu_name}, but couldn't retrieve any financial data from the database."
            
            # Check if any tables have actual data
            results = financial_data.get("results", {})
            if not any(results.get(table_name) for table_name in results):
                logger.error(f"Financial data contains tables but all are empty")
                cu_name = self.format_credit_union_name(credit_union_info.get("cu_name", "this credit union"))
                return f"I found {cu_name} in our database, but there's no financial data available. This could be because the credit union hasn't reported recent data."
            
            # Format the credit union name properly
            cu_name = self.format_credit_union_name(credit_union_info.get("cu_name", credit_union_info.get("credit_union_name", "this credit union")))
            
            # Extract key metrics from the data
            extracted_data = self.extract_key_financial_metrics(financial_data, user_query)
            logger.info(f"Extracted {len(extracted_data)} key data points locally")
            
            # Whether it's an uncommon query or not, we'll use a consistent approach for concise answers
            system_content = f"""
You are a financial assistant providing information about {cu_name}.

IMPORTANT INSTRUCTIONS:
1. Be extremely concise - answer in 1-2 sentences maximum
2. Do NOT explain which tables or fields you used to find the information
3. Do NOT explain your reasoning or process
4. Simply state the answer directly and nothing more
5. Include dollar signs and commas for financial amounts
6. Focus ONLY on answering the specific question asked

The user does not need to know how you found the information or what tables contained it.
Just provide the direct answer as if you already knew it.

Example format:
Question: "What is the asset size of Navy Federal?"
Good answer: "Navy Federal Credit Union has total assets of $175,893,456,789."
Bad answer: "After examining the fs220_2024_12 table, I found that Navy Federal Credit Union has assets of $175,893,456,789 as reported in the acct_010 field."
"""

            # Create a simple, focused user content that includes only key data points
            user_content = f"""
Question: {user_query}

Here are the relevant financial metrics for {cu_name}:

{extracted_data}
"""
            
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
            
            logger.info("Sending query with extracted data points to OpenAI")
            
            response = openai.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0,  # Reduced temperature for more direct answers
                max_tokens=100  # Limit token count to encourage brevity
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Received answer from OpenAI: {answer}")
            
            return answer
            
        except Exception as e:
            logger.error(f"Error interpreting results: {str(e)}")
            logger.error(traceback.format_exc())
            # Give a helpful error message that explains the core issue
            return "I encountered a technical problem analyzing the financial data. Please try again with a more specific question."
            
    def extract_key_financial_metrics(self, financial_data: Dict[str, Any], user_query: str) -> str:
        """
        Extract key financial metrics from the data locally rather than sending
        the entire dataset to OpenAI for analysis.
        
        Args:
            financial_data: The financial data dictionary
            user_query: The user's query to help determine relevant metrics
            
        Returns:
            String with key metrics in a structured format
        """
        extracted_metrics = []
        results = financial_data.get("results", {})
        
        # Define common account codes and their descriptions
        common_metrics = {
            # Assets and Size
            "acct_010": "Total Assets",
            "010": "Total Assets",
            "acct_799": "Total Other Assets",
            "799": "Total Other Assets",
            
            # Net Worth/Equity
            "acct_997": "Net Worth",
            "997": "Net Worth",
            "acct_998": "Regular Reserves",
            "998": "Regular Reserves",
            "acct_996": "Undivided Earnings",
            "996": "Undivided Earnings",
            
            # Income and Earnings
            "acct_661a": "Net Income",
            "661a": "Net Income",
            "acct_661A": "Net Income",
            "661A": "Net Income",
            "acct_115": "Income from Investments",
            "115": "Income from Investments",
            
            # Shares/Deposits
            "acct_018": "Total Shares and Deposits",
            "018": "Total Shares and Deposits",
            "acct_013": "Total Shares and Deposits",
            "013": "Total Shares and Deposits",
            
            # Loans
            "acct_025b": "Total Loans",
            "025b": "Total Loans",
            "acct_025B": "Total Loans",
            "025B": "Total Loans",
            "acct_025": "Total Loans",
            "025": "Total Loans",
            "acct_703": "Consumer Loans",
            "703": "Consumer Loans",
            "acct_704": "Real Estate Loans",
            "704": "Real Estate Loans",
            
            # Member Information
            "acct_083": "Number of Current Members",
            "083": "Number of Current Members",
            "acct_084": "Potential Members",
            "084": "Potential Members"
        }
        
        # Search all tables for the metrics
        for table_name, table_data in results.items():
            if not table_data:
                continue
                
            # Get the first record from each table
            record = table_data[0]
            
            # Check for common metrics in this record
            for field, value in record.items():
                if field in common_metrics and value is not None and value != 0:
                    metric_name = common_metrics[field]
                    
                    # Format financial amounts with $ and commas
                    if isinstance(value, (int, float)) and "number" not in metric_name.lower():
                        formatted_value = f"${value:,.2f}"
                    else:
                        formatted_value = f"{value:,}" if isinstance(value, (int, float)) else str(value)
                    
                    # Add to our list of extracted metrics
                    extracted_metrics.append(f"{metric_name}: {formatted_value}")
        
        # If it's a branch or location query, extract branch information
        if any(term in user_query.lower() for term in ["branch", "location", "office", "address"]):
            for table_name, table_data in results.items():
                if "branch" in table_name.lower() and table_data:
                    # Get branch count
                    branch_count = len(table_data)
                    extracted_metrics.append(f"Number of Branches: {branch_count}")
                    
                    # Include a sample of branch locations
                    if branch_count > 0:
                        branch_locations = []
                        for i, branch in enumerate(table_data[:3]):  # Get first 3 branches
                            address = []
                            if branch.get("braddr1"):
                                address.append(branch.get("braddr1"))
                            if branch.get("brcity") and branch.get("brstate"):
                                address.append(f"{branch.get('brcity')}, {branch.get('brstate')}")
                            if address:
                                branch_locations.append(", ".join(address))
                                
                        if branch_locations:
                            branches_text = "; ".join(branch_locations)
                            if branch_count > 3:
                                branches_text += f" (plus {branch_count - 3} more)"
                            extracted_metrics.append(f"Branch Locations: {branches_text}")
        
        # If we found no metrics but have specific tables, extract key facts
        if not extracted_metrics and results:
            for table_name, table_data in results.items():
                if table_data and "foicu" in table_name.lower():
                    # Extract charter year, charter type if available
                    record = table_data[0]
                    if record.get("charteryr"):
                        extracted_metrics.append(f"Charter Year: {record.get('charteryr')}")
                    if record.get("chartertype"):
                        charter_type = "Federal" if record.get("chartertype") == "F" else "State"
                        extracted_metrics.append(f"Charter Type: {charter_type}")
        
        # If we still have no data, check for any non-zero values in first record
        if not extracted_metrics:
            for table_name, table_data in results.items():
                if table_data:
                    record = table_data[0]
                    for field, value in record.items():
                        # Skip metadata fields
                        if field.lower() in ["id", "cu_number", "cycle_date", "join_number"]:
                            continue
                            
                        # Include non-zero financial values
                        if isinstance(value, (int, float)) and value != 0:
                            # Format the field name in a more readable way
                            readable_field = field.replace("_", " ").replace("acct", "Account").title()
                            
                            # Format the value
                            formatted_value = f"${value:,.2f}" if isinstance(value, (int, float)) else str(value)
                            
                            # Add to metrics
                            extracted_metrics.append(f"{readable_field}: {formatted_value}")
                            
                            # Limit to 5 metrics from this fallback method
                            if len(extracted_metrics) >= 5:
                                break
                                
                    # Break after checking one table to avoid too many metrics
                    if extracted_metrics:
                        break
        
        # If still no data, return a message about lacking specific data
        if not extracted_metrics:
            return "No specific financial metrics found in the data."
        
        # Join all the metrics with newlines for better readability
        return "\n".join(extracted_metrics)

    def format_credit_union_name(self, cu_name: str) -> str:
        """
        Format a credit union name properly for display.
        
        Args:
            cu_name: Raw credit union name, often in all caps
            
        Returns:
            Properly formatted credit union name
        """
        if not cu_name:
            return "this credit union"
            
        # Handle all caps names
        if cu_name.isupper():
            # Convert to title case but handle special cases
            words = cu_name.lower().split()
            
            # Words to keep lowercase
            lowercase_words = {'of', 'the', 'and', 'in', 'at', 'by', 'for', 'with', 'a', 'an'}
            
            # Special abbreviations to preserve
            abbreviations = {'fcu': 'FCU', 'cu': 'CU', 'n.c.': 'N.C.', 'n.h.': 'N.H.', 's.c.': 'S.C.', 
                             's.d.': 'S.D.', 'n.y.': 'N.Y.', 'n.j.': 'N.J.', 'f.c.u.': 'F.C.U.', 'c.u.': 'C.U.'}
            
            # Format each word
            formatted_words = []
            for i, word in enumerate(words):
                # Check if it's an abbreviation
                if word.lower() in abbreviations:
                    formatted_words.append(abbreviations[word.lower()])
                # Keep lowercase words lowercase, except at the beginning
                elif word in lowercase_words and i > 0:
                    formatted_words.append(word)
                # Handle hyphenated words
                elif '-' in word:
                    formatted_words.append('-'.join(w.capitalize() for w in word.split('-')))
                # Handle apostrophes correctly
                elif "'" in word:
                    parts = word.split("'")
                    formatted = parts[0].capitalize() + "'" + parts[1]
                    formatted_words.append(formatted)
                # Default case - capitalize
                else:
                    formatted_words.append(word.capitalize())
            
            formatted_name = ' '.join(formatted_words)
            
            # Special case for "Federal Credit Union" - expand FCU if it appears at the end
            if formatted_name.endswith("Fcu") or formatted_name.endswith("FCU"):
                formatted_name = formatted_name[:-3] + "Federal Credit Union"
            # Also check for other common abbreviations that we might want to expand
            elif formatted_name.endswith("Cu") or formatted_name.endswith("CU"):
                formatted_name = formatted_name[:-2] + "Credit Union"
                
            return formatted_name
        
        # If not all caps, assume it's already formatted properly
        return cu_name

    def extract_credit_union_from_query(self, user_query: str) -> Optional[Dict[str, Any]]:
        """
        Extract credit union name from query and find the matching credit union.
        If no credit union is found but there's current context, use that instead.
        Prioritizes exact state matches when location context is provided.
        
        Args:
            user_query: The user's query about a credit union
            
        Returns:
            Dictionary with credit union information or None if not found
        """
        try:
            # Check if this is a follow-up question using pronouns (they, them, their, it, etc.)
            has_pronoun = any(pronoun in user_query.lower() for pronoun in [
                " they ", " them ", " their ", " it ", " its ", 
                "they ", "them ", "their ", "it ", "its ",
                " they", " them", " their", " it", " its"
            ])
            
            # Look for patterns like "what's their asset size" or "how many members do they have"
            contextual_phrases = [
                "what is their", "what's their", "how many", "how much", 
                "asset size", "net worth", "income", "revenue", "profit",
                "loans", "deposits", "members", "branches"
            ]
            
            has_contextual_phrase = any(phrase in user_query.lower() for phrase in contextual_phrases)
            
            # If we have pronouns or contextual phrases and active context, use that
            if (has_pronoun or has_contextual_phrase) and self.context.has_context():
                if DEBUG_MODE:
                    console.print("[dim]Using active conversation context for credit union[/dim]")
                return self.context.get_active_credit_union()
            
            # Extract credit union name and location information from query
            credit_union_name, location_context = self.extract_credit_union_info(user_query)
            
            if not credit_union_name:
                # If no explicit credit union name and we have context, use the active context
                if self.context.has_context():
                    if DEBUG_MODE:
                        console.print("[dim]No credit union name found, using existing context[/dim]")
                    return self.context.get_active_credit_union()
                
                return None
            
            # Find the credit union in the database with special handling for state context
            # State matching is particularly important for credit unions with similar names
            cu_info = self.find_credit_union(credit_union_name, location_context)
            
            if not cu_info:
                return None
            
            # Update conversation context with the new credit union
            self.context.set_active_credit_union(cu_info)
            
            # Display confirmation of the found credit union (in debug mode only)
            if DEBUG_MODE:
                cu_name = self.format_credit_union_name(cu_info.get("cu_name", "Unknown"))
                console.print(f"[dim]Found credit union: {cu_name} (ID: {cu_info.get('cu_number', 'Unknown')})[/dim]")
                
            return cu_info
            
        except Exception as e:
            logger.error(f"Error extracting credit union from query: {str(e)}")
            return None

    async def process_query(self, user_query: str) -> str:
        """
        Process a user query about a credit union using parallel operations.
        Uses conversation context to optimize retrieval and handle follow-up questions.
        Employs targeted queries based on account descriptions for efficiency.
        
        Args:
            user_query: The user's query about a credit union
            
        Returns:
            Answer to the user's query
        """
        try:
            logger.info(f"Processing query: '{user_query}'")
            
            # Extract credit union name from query, using context for follow-up questions
            logger.info("Extracting credit union from query...")
            cu_info = self.extract_credit_union_from_query(user_query)
            
            if not cu_info:
                logger.warning("No credit union identified in the query")
                return "I couldn't identify a specific credit union in your question. Please specify which credit union you're asking about, or ask a follow-up question about the same credit union."
            
            # Ensure consistent naming
            if "credit_union_name" in cu_info and "cu_name" not in cu_info:
                cu_info["cu_name"] = cu_info["credit_union_name"]
            elif "cu_name" in cu_info and "credit_union_name" not in cu_info:
                cu_info["credit_union_name"] = cu_info["cu_name"]
            
            if "cu_name" not in cu_info and "credit_union_name" not in cu_info:
                cu_info["cu_name"] = "Unknown Credit Union"
                cu_info["credit_union_name"] = "Unknown Credit Union"
            
            logger.info(f"Found credit union: {cu_info.get('cu_name', 'Unknown')} with number {cu_info.get('cu_number', 'Unknown')}")
            
            if DEBUG_MODE:
                console.print(f"[dim]Processing query for {cu_info.get('cu_name', 'Unknown Credit Union')} (ID: {cu_info.get('cu_number', 'Unknown')})[/dim]")
            
            # Extract relevant information from the query to determine what metrics to search for
            # This will be used for targeted querying
            relevant_terms = self.extract_query_terms(user_query)
            logger.info(f"Extracted relevant terms from query: {relevant_terms}")
            
            # Check if we already have financial data in the context
            financial_data = None
            need_full_data = False
            context_has_data = False
            use_targeted_query = True  # Default to using targeted query
            
            # Check if there may be a request for information outside of our common metrics
            # Look for keywords that suggest detailed or uncommon information
            uncommon_info_keywords = [
                # Branch and location information
                "branch", "branches", "location", "address", "street", "avenue", "drive", "city", "state", "zip", "postal",
                
                # Historical information
                "founded", "charter", "year", "history", "establish", "start", "begin", "origin", "found", "date", "since",
                
                # Services and products
                "products", "services", "rates", "fees", "routing", "offer", "provide", "available", "account type", 
                "checking", "savings", "auto loan", "mortgage", "credit card", "business",
                
                # People and governance
                "board", "executive", "leadership", "ceo", "president", "chair", "director", "manager", "officers",
                
                # Contact and support
                "insurance", "compliance", "atm", "phone", "contact", "website", "email", "call", "reach", 
                "support", "customer service", "help", "assistance",
                
                # Operations
                "hours", "open", "close", "time", "schedule", "holiday", "weekend", "operate", "operating",
                
                # Membership
                "field of membership", "join", "eligibility", "qualify", "requirement", "community", "sponsor",
                
                # Other queries that require full context
                "how many", "when", "where", "information about", "details on", "tell me about", "describe"
            ]
            
            is_uncommon_query = any(keyword in user_query.lower() for keyword in uncommon_info_keywords)
            if is_uncommon_query:
                use_targeted_query = False  # Use full data query for uncommon queries
                logger.info(f"Using full data query for uncommon information request")
            
            if self.context.has_context() and self.context.get_active_credit_union() and \
               self.context.get_active_credit_union().get('cu_number') == cu_info.get('cu_number'):
                context_data = self.context.get_active_financial_data()
                if context_data:
                    context_has_data = True
                    # If this is a query for uncommon information, we might need to get full data
                    if is_uncommon_query:
                        logger.info(f"Query may require additional data beyond cached metrics")
                        need_full_data = True
                    else:
                        # Check if the context data was a targeted query and if it contains what we need
                        if context_data.get("targeted_query") and context_data.get("queried_metrics"):
                            # Check if all our current terms were in the previous query
                            previous_metrics = context_data.get("queried_metrics", [])
                            if all(term in previous_metrics for term in relevant_terms):
                                logger.info(f"Context already has data for all requested metrics")
                                financial_data = context_data
                                use_targeted_query = False  # Don't need another targeted query
                            else:
                                logger.info(f"Need to run targeted query for additional metrics")
                                # We'll run a new targeted query with the combined metrics
                                relevant_terms = list(set(relevant_terms + previous_metrics))
                        else:
                            # Use the cached data for common financial metrics
                            financial_data = context_data
                            logger.info(f"Using financial data from conversation context for {cu_info.get('cu_name')}")
                            if DEBUG_MODE:
                                console.print(f"[dim]Using cached financial data from conversation context[/dim]")
            
            # Perform the appropriate query based on the situation
            if not financial_data:
                if use_targeted_query and relevant_terms:
                    # Use the more efficient targeted query approach
                    logger.info(f"Using targeted query for metrics: {relevant_terms}")
                    financial_data = self.targeted_financial_query(cu_info["cu_number"], relevant_terms)
                else:
                    # Fall back to full data query
                    logger.info(f"Using full data query for cu_number: {cu_info['cu_number']}")
                    
                    # Run the standard query asynchronously
                    financial_data = await self.query_financial_data_async(cu_info["cu_number"])
                
                # Update conversation context with the financial data
                if financial_data:
                    self.context.set_active_credit_union(cu_info, financial_data)
                    logger.info(f"Updated conversation context with financial data for {cu_info.get('cu_name')}")
            
            # Record the query in the conversation context
            if cu_info and "cu_number" in cu_info:
                self.context.record_query(cu_info["cu_number"], user_query)
            
            if not financial_data:
                logger.warning(f"No financial data found for CU #{cu_info.get('cu_number', 'Unknown')}")
                return f"I couldn't find any financial data for {cu_info.get('cu_name', 'this credit union')}. Please check the credit union name and try again."
            
            # Extract results from financial_data
            results = financial_data.get("results", {})
            
            # Check how many tables have data
            num_tables_with_data = len(results)
            logger.info(f"Retrieved data from {num_tables_with_data} tables")
            
            if DEBUG_MODE:
                console.print(f"[dim]Retrieved data from {num_tables_with_data} tables[/dim]")
                console.print(f"[dim]Credit union info: {cu_info}[/dim]")
                console.print(f"[dim]Financial data has {len(results)} tables[/dim]")
                
                # List the tables we found data in
                if results:
                    tables_list = ", ".join(results.keys())
                    console.print(f"[dim]Tables with data: {tables_list}[/dim]")
            
            # We don't need to filter for targeted queries since they're already filtered
            if financial_data.get("targeted_query"):
                filtered_data = financial_data
                # Make sure credit union info is included
                filtered_data["credit_union_info"] = cu_info
            # For uncommon queries, use the complete financial data rather than filtering
            elif is_uncommon_query:
                logger.info(f"Using complete financial data for uncommon query")
                filtered_data = financial_data
                # Ensure credit union info is included
                filtered_data["credit_union_info"] = cu_info
                
                # Add some context about what was detected
                filtered_data["query_type"] = "uncommon_information"
                filtered_data["detected_keywords"] = [kw for kw in uncommon_info_keywords if kw in user_query.lower()]
            else:
                # For regular queries without targeting, extract required fields
                # Identify required metrics for filtering
                required_metrics = self.identify_required_metrics(user_query)
                
                # Now proceed with filtering for standard financial metrics
                financial_data["credit_union_info"] = cu_info
                filtered_data = self.extract_required_fields(financial_data, required_metrics)
            
            # Interpret results - include context about which credit union this is
            cu_name = self.format_credit_union_name(cu_info.get("cu_name", cu_info.get("credit_union_name", "this credit union")))
            answer = self.interpret_results(user_query, cu_info, filtered_data)
            
            # If the original query didn't specify the credit union but we used context,
            # make sure the answer clarifies which credit union we're talking about
            if (any(pronoun in user_query.lower() for pronoun in ["they", "them", "their", "it", "its"])) and \
               not any(cu_name.lower() in user_query.lower() for cu_name in [
                   cu_info.get("cu_name", "").lower(), 
                   cu_info.get("credit_union_name", "").lower()
               ]):
                # But check if the answer already mentions the credit union name
                if cu_name.lower() not in answer.lower():
                    answer = f"For {cu_name}, {answer[0].lower() + answer[1:]}"
            
            return answer
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(traceback.format_exc())
            return "I encountered an error processing your question. Please try again."
            
    def extract_query_terms(self, query: str) -> List[str]:
        """
        Extract relevant terms from a user query for targeted database queries.
        
        Args:
            query: The user's query
            
        Returns:
            List of relevant terms for database lookups
        """
        # Common financial terms to look for
        financial_terms = {
            "assets": ["asset", "assets", "size", "total assets", "asset size"],
            "net worth": ["net worth", "networth", "equity", "capital", "net capital"],
            "net income": ["income", "earnings", "profit", "revenue", "net income", "make", "earn"],
            "loans": ["loan", "loans", "lending", "mortgage", "credit", "borrow"],
            "shares": ["share", "shares", "deposit", "deposits", "savings", "saving"],
            "members": ["member", "members", "membership", "customer", "customers"],
            "branches": ["branch", "branches", "location", "office", "offices"],
            "rates": ["rate", "rates", "interest", "yield", "apr"],
            "fees": ["fee", "fees", "charge", "charges", "cost", "costs"]
        }
        
        # Convert query to lowercase for matching
        query_lower = query.lower()
        
        # Find matching terms
        matched_terms = []
        for category, terms in financial_terms.items():
            if any(term in query_lower for term in terms):
                matched_terms.append(category)
                
        # If we found specific terms, return those
        if matched_terms:
            return matched_terms
            
        # If we couldn't identify specific terms, include general financial metrics
        # This ensures we get reasonable data for general questions
        return ["assets", "net income", "net worth", "members"]

    async def find_credit_union_async(self, credit_union_name: str, location_context: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """
        Asynchronous version of find_credit_union that uses asyncio for subprocess operations.
        
        Args:
            credit_union_name: Name of the credit union to search for
            location_context: Optional location information to refine the search
            
        Returns:
            Credit union information if found, None otherwise
        """
        try:
            logger.info(f"Searching for credit union: '{credit_union_name}'")
            
            # Build base command
            cmd = ["python", "query_credit_unions.py", credit_union_name, "--min-score", "0.65", "--force-min-score", "--all-results"]
            
            # Add location context if provided
            if location_context:
                if location_context.get("state_code"):
                    cmd.extend(["--state-code", location_context["state_code"]])
                elif location_context.get("state_full"):
                    cmd.extend(["--state", location_context["state_full"]])
                if location_context.get("city"):
                    cmd.extend(["--city", location_context["city"]])
            
            logger.info(f"Running vector search command: {' '.join(cmd)}")
            
            # Use asyncio to run the subprocess
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            # Process output
            output = stdout.decode('utf-8')
            
            if process.returncode != 0:
                logger.error(f"Error running query_credit_unions.py: {stderr.decode('utf-8')}")
                return None
                
            # Run again with the same command to capture output for processing
            logger.info("Re-running to capture output for processing...")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            raw_output = stdout.decode('utf-8')
            logger.info(f"Vector search raw output: {raw_output[:500]}...")
            
            if DEBUG_MODE:
                # Display the formatted output
                console.print(Panel(
                    raw_output[:2000] + ("..." if len(raw_output) > 2000 else ""),
                    title="Vector Search Output"
                ))
            
            # Parse the output to extract cu_number
            cu_number = None
            credit_union_name = None
            
            # Try to extract cu_number using regex patterns
            # Pattern 1: Look for cu_number in metadata section
            pattern1 = r"cu_number:\s*(\d+\.?\d*)"
            pattern2 = r"cu_number.*?(\d+\.?\d*)"
            
            match = re.search(pattern1, raw_output)
            if match:
                cu_number = match.group(1)
                logger.info(f"Normalized cu_number from {cu_number} to {str(int(float(cu_number)))}")
                cu_number = str(int(float(cu_number)))
                logger.info(f"Found cu_number via regex 1: {cu_number}")
            else:
                # Try the second pattern if the first one didn't work
                match = re.search(pattern2, raw_output)
                if match:
                    cu_number = match.group(1)
                    logger.info(f"Found cu_number via regex 2: {cu_number}")
            
            # Extract the credit union name using a regex pattern
            name_pattern = r"Credit Union:\s*(.*?)(?:\s*\(|$)"
            name_match = re.search(name_pattern, raw_output)
            if name_match:
                credit_union_name = name_match.group(1).strip()
                logger.info(f"Found credit union name: {credit_union_name}")
            
            # Extract metadata section
            metadata_marker = "All Available Metadata:"
            metadata = {}
            
            metadata_section_start = raw_output.find(metadata_marker)
            if metadata_section_start != -1:
                logger.info("Found metadata section marker")
                metadata_section = raw_output[metadata_section_start:]
                
                # Find the end of the metadata section
                end_markers = ["--------------------------------------------------", "Found", "No matches"]
                metadata_section_end = len(metadata_section)
                
                for marker in end_markers:
                    pos = metadata_section.find(marker, len(metadata_marker))
                    if pos != -1 and pos < metadata_section_end:
                        metadata_section_end = pos
                
                metadata_section = metadata_section[:metadata_section_end]
                logger.info("Reached end of metadata section")
                
                # Parse metadata lines
                lines = metadata_section.split('\n')
                for line in lines[1:]:  # Skip the "All Available Metadata:" line
                    line = line.strip()
                    if line and ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
            
            # If we found a cu_number, construct the result
            if cu_number:
                # Normalize the cu_number
                normalized_cu_number = str(int(float(cu_number)))
                logger.info(f"Stored normalized_cu_number: {normalized_cu_number}")
                
                # Add additional information if available
                result = {
                    "cu_number": cu_number
                }
                
                # Add other metadata if available
                for key, value in metadata.items():
                    result[key] = value
                
                # Add the credit union name if we found it
                if credit_union_name:
                    result["credit_union_name"] = credit_union_name
                
                # Add normalized cu_number
                result["normalized_cu_number"] = normalized_cu_number
                
                logger.info(f"Successfully found credit union: {credit_union_name} with number {cu_number}")
                return result
            else:
                logger.warning("Could not extract cu_number from vector search output")
                return None
            
        except Exception as e:
            logger.error(f"Error in find_credit_union_async: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def query_financial_data_async(self, cu_number: str) -> Optional[Dict[str, Any]]:
        """
        Asynchronous version of query_financial_data that uses asyncio for subprocess operations.
        
        Args:
            cu_number: Credit union number
            
        Returns:
            Dictionary of financial data if found, None otherwise
        """
        try:
            # Normalize the credit union number
            normalized_cu_number = cu_number
            if isinstance(cu_number, str) and '.' in cu_number:
                try:
                    normalized_cu_number = str(int(float(cu_number)))
                    logger.info(f"Normalized cu_number from {cu_number} to {normalized_cu_number}")
                except ValueError:
                    logger.warning(f"Could not normalize cu_number: {cu_number}")
            
            # Check cache first
            cached_data = self.metrics_cache.get(normalized_cu_number)
            if cached_data is not None:
                logger.info(f"Using cached financial data for CU #{normalized_cu_number}")
                return cached_data
            
            logger.info(f"Querying financial data for CU #{normalized_cu_number}, original: {cu_number}")
            
            # Create a temporary file for output
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_file:
                output_file = tmp_file.name
                logger.info(f"Created temporary output file: {output_file}")
            
            # Skip the readable query for speed in async mode
            # Build and run command asynchronously
            cmd = ["python", "query_all_tables.py", normalized_cu_number, "--output", output_file]
            logger.info(f"Running database query command: {' '.join(cmd)}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Error running query_all_tables.py: {stderr.decode('utf-8')}")
                logger.error(f"STDOUT: {stdout.decode('utf-8')[:1000]}")
                return None
            
            # Read the output file
            try:
                # Check if the file exists and has content
                if not os.path.exists(output_file):
                    logger.error(f"Output file {output_file} does not exist")
                    return None
                
                file_size = os.path.getsize(output_file)
                logger.info(f"Output file size: {file_size} bytes")
                
                if file_size == 0:
                    logger.error(f"Output file {output_file} is empty")
                    return None
                
                with open(output_file, 'r') as f:
                    file_content = f.read()
                    logger.info(f"Read {len(file_content)} characters from output file")
                    
                    if not file_content.strip():
                        logger.error("Output file content is empty or whitespace only")
                        return None
                    
                    financial_data = json.loads(file_content)
                    logger.info(f"Successfully parsed JSON from output file")
                
                # Clean up the temporary file
                os.unlink(output_file)
                logger.info(f"Deleted temporary output file")
                
                # Check if we have data
                results = financial_data.get("results", {})
                logger.info(f"Found {len(results)} tables in results")
                
                # Store the results in the cache
                self.metrics_cache.set(normalized_cu_number, financial_data)
                logger.info(f"Cached financial data for CU #{normalized_cu_number}")
                
                # Just return the raw data - do not enrich or process it
                return financial_data
                
            except Exception as e:
                logger.error(f"Error reading financial data: {str(e)}")
                logger.error(traceback.format_exc())
                return None
                
        except Exception as e:
            logger.error(f"Error in query_financial_data_async: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def identify_required_metrics_async(self, user_query: str) -> Dict[str, Any]:
        """
        Asynchronous version of identify_required_metrics.
        Since this is a CPU-bound operation, we run it in a thread pool.
        
        Args:
            user_query: The user's query
            
        Returns:
            Dictionary of required metrics
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.identify_required_metrics, user_query)

    async def chat_loop(self):
        """Run the interactive chat loop with context management."""
        # Clean welcome message without technical details
        console.print(Panel.fit(
            "[bold blue]NCUA Credit Union Chatbot[/bold blue]\n"
            "Ask me questions about credit unions and their financial data.",
            title="Welcome",
            border_style="blue"
        ))
        
        while True:
            try:
                # Display active context if available, but only show the credit union name
                prompt_text = "\nYour question"
                if self.context.has_context():
                    active_cu = self.context.get_active_credit_union()
                    if active_cu:
                        cu_name = self.format_credit_union_name(active_cu.get("cu_name", active_cu.get("credit_union_name", "Unknown")))
                        prompt_text = f"\nYour question about [cyan]{cu_name}[/cyan]"
                        
                        # Only show technical details in debug mode
                        if DEBUG_MODE:
                            cu_number = active_cu.get("cu_number", "Unknown")
                            console.print(f"[dim]Active context: {cu_name} (ID: {cu_number})[/dim]")
                
                user_query = Prompt.ask(prompt_text, default="quit")
                
                # Check for exit command
                if user_query.lower() in ["quit", "exit", "bye", "goodbye"]:
                    console.print("[green]Thank you for using the NCUA Credit Union Chatbot. Goodbye![/green]")
                    break
                
                # Check for context commands
                if user_query.lower() in ["clear context", "reset context", "forget"]:
                    self.context.clear_context()
                    console.print("[green]Context cleared. I've forgotten the current credit union.[/green]")
                    continue
                
                # Show a subtle loading indicator
                with console.status("[cyan]Searching for information...[/cyan]", spinner="dots"):
                    # Process the query
                    answer = await self.process_query(user_query)
                
                # Display the answer
                if answer:
                    # Check if current context is present
                    has_context = False
                    context_info = ""
                    
                    if self.context.has_context():
                        active_cu = self.context.get_active_credit_union()
                        if active_cu:
                            cu_name = self.format_credit_union_name(active_cu.get("cu_name", active_cu.get("credit_union_name", "Unknown")))
                            context_info = f" [dim](Discussing {cu_name})[/dim]"
                            has_context = True
                    
                    title = "Answer"
                    if has_context and DEBUG_MODE:
                        title += context_info
                    
                    console.print(Panel(Markdown(answer), title=title, border_style="green"))
                else:
                    console.print("[yellow]Sorry, I couldn't find an answer to your question.[/yellow]")
                    
            except KeyboardInterrupt:
                console.print("\n[green]Goodbye![/green]")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                console.print("[red]An error occurred. Please try again.[/red]")

    def query_account_descriptions(self, search_terms: List[str]) -> Dict[str, Any]:
        """
        Query the acctdesc_2024_12 table to find account mapping information based on search terms.
        This is used to navigate the database schema and build targeted queries.
        
        Args:
            search_terms: List of terms to search for in account descriptions
            
        Returns:
            Dictionary mapping account codes to tables and descriptions
        """
        try:
            account_mappings = {}
            
            # Build command to query the account description table
            cmd = ["python", "interactive_query.py", "--account-mapping"]
            
            # Add search terms as arguments
            for term in search_terms:
                cmd.extend(["--search-term", term])
                
            logger.info(f"Querying account descriptions with command: {' '.join(cmd)}")
            
            # Run the command to get account mappings
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error querying account descriptions: {result.stderr}")
                return {}
                
            # Process the output to extract account mappings
            try:
                account_mappings = json.loads(result.stdout)
                logger.info(f"Found {len(account_mappings)} account mappings")
                
                # Log a sample of the mappings for debugging
                if account_mappings and DEBUG_MODE:
                    sample_keys = list(account_mappings.keys())[:3]
                    for key in sample_keys:
                        logger.info(f"Account mapping: {key} -> {account_mappings[key]}")
                        
                return account_mappings
            except json.JSONDecodeError:
                # Fallback: parse the text output
                logger.warning("Could not parse JSON output, using text parsing")
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if ':' in line:
                        parts = line.split(':', 1)
                        acct_code = parts[0].strip()
                        details = parts[1].strip()
                        
                        # Try to parse details as a dictionary
                        if '{' in details and '}' in details:
                            try:
                                details_dict = json.loads(details)
                                account_mappings[acct_code] = details_dict
                            except:
                                # Just store as string if can't parse
                                account_mappings[acct_code] = {'description': details}
                        else:
                            account_mappings[acct_code] = {'description': details}
                
                return account_mappings
                
        except Exception as e:
            logger.error(f"Error in query_account_descriptions: {str(e)}")
            logger.error(traceback.format_exc())
            return {}
            
    def targeted_financial_query(self, cu_number: str, metrics: List[str]) -> Dict[str, Any]:
        """
        Perform targeted queries based on account mappings for efficiency.
        
        Args:
            cu_number: Credit union number to query
            metrics: List of metrics to query (like "net income", "assets", etc.)
            
        Returns:
            Dictionary of query results with just the requested data
        """
        try:
            # Normalize the cu_number
            if isinstance(cu_number, str) and '.' in cu_number:
                try:
                    normalized_cu_number = str(int(float(cu_number)))
                    logger.info(f"Normalized cu_number from {cu_number} to {normalized_cu_number}")
                except ValueError:
                    normalized_cu_number = cu_number
            else:
                normalized_cu_number = cu_number
                
            # Get account mappings for the requested metrics
            logger.info(f"Getting account mappings for metrics: {metrics}")
            account_mappings = self.query_account_descriptions(metrics)
            
            if not account_mappings:
                logger.warning(f"No account mappings found for metrics: {metrics}")
                # Fall back to standard query approach
                return self.query_financial_data(normalized_cu_number)
                
            # Build targeted queries based on the account mappings
            targeted_results = {
                "credit_union_info": {
                    "cu_number": normalized_cu_number
                },
                "results": {},
                "targeted_query": True,
                "queried_metrics": metrics
            }
            
            # Track which tables we need to query
            tables_to_query = {}
            for acct_code, mapping in account_mappings.items():
                table_name = mapping.get("tablename")
                if not table_name:
                    continue
                    
                # Convert to actual table name format with year/month
                actual_table = self.tablename_to_table.get(table_name)
                if not actual_table:
                    # Try lowercase
                    actual_table = self.tablename_to_table.get(table_name.lower())
                    
                if not actual_table:
                    # If we can't map it, use a guessed format
                    actual_table = f"{table_name.lower()}_2024_12"
                
                # Add this account to the list for this table
                if actual_table not in tables_to_query:
                    tables_to_query[actual_table] = []
                
                # Store the account code format correctly
                if acct_code.lower().startswith("acct_"):
                    tables_to_query[actual_table].append(acct_code)
                else:
                    tables_to_query[actual_table].append(f"acct_{acct_code}")
            
            logger.info(f"Tables to query: {tables_to_query}")
            
            # Query each table for the specific columns we need
            for table_name, account_codes in tables_to_query.items():
                # Build a comma-separated list of columns to query
                query_columns = ["cu_number", "id"]
                query_columns.extend(account_codes)
                columns_str = ",".join(query_columns)
                
                logger.info(f"Querying {table_name} for columns: {columns_str}")
                
                # Run a direct query to get just these specific columns
                cmd = ["python", "interactive_query.py", 
                        "--table", table_name, 
                        "--cu-number", normalized_cu_number,
                        "--columns", columns_str,
                        "--output-json"]
                
                logger.info(f"Running direct query command: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.error(f"Error querying {table_name}: {result.stderr}")
                    continue
                
                # Parse the results
                try:
                    table_results = json.loads(result.stdout)
                    if table_results:
                        targeted_results["results"][table_name] = table_results
                        logger.info(f"Retrieved {len(table_results)} records from {table_name}")
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse results from {table_name}")
                    # Try to extract any structured data
                    if "{" in result.stdout and "}" in result.stdout:
                        try:
                            import re
                            json_objects = re.findall(r'\{[^{}]*\}', result.stdout)
                            table_results = [json.loads(obj) for obj in json_objects]
                            targeted_results["results"][table_name] = table_results
                            logger.info(f"Extracted {len(table_results)} records from {table_name} using regex")
                        except:
                            logger.error(f"Failed to extract data using regex")
            
            # If we didn't get any results, fall back to the standard approach
            if not targeted_results["results"]:
                logger.warning("No targeted results found, falling back to standard query")
                return self.query_financial_data(normalized_cu_number)
                
            return targeted_results
            
        except Exception as e:
            logger.error(f"Error in targeted_financial_query: {str(e)}")
            logger.error(traceback.format_exc())
            # Fall back to standard query
            logger.info("Falling back to standard query due to error")
            return self.query_financial_data(normalized_cu_number)

async def main():
    try:
        # Only show these startup messages in debug mode
        if DEBUG_MODE:
            print("Starting main function...")
            logger.info("Starting NCUA Chatbot main function")
            
            # Check for required Python packages
            print("Checking required packages...")
            try:
                import openai
                print("OpenAI package version:", openai.__version__)
            except ImportError:
                print("Error: OpenAI package not installed. Please run: pip install openai>=1.0.0")
                sys.exit(1)
            
            # Verify environment variables
            print("Checking environment variables...")
            required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_API_KEY", "PINECONE_API_KEY"]
            missing_vars = [var for var in required_vars if not os.getenv(var)]
            
            if missing_vars:
                print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
                print("Please make sure these are set in your .env file")
                sys.exit(1)
        
        # Initialize chatbot without verbose output
        chatbot = NCUAChatbot()
        
        # Special test mode for query_financial_data
        if TEST_QUERY_MODE:
            print(f"\n[TEST MODE] Testing query_financial_data with cu_number: {args.test_query}")
            financial_data = await chatbot.query_financial_data_async(args.test_query)
            
            if not financial_data:
                print("No financial data found!")
                return
                
            print(f"Retrieved financial data with {len(financial_data.get('results', {}))} tables")
            
            # Check if any tables have actual data
            results = financial_data.get("results", {})
            tables_with_data = [table for table, records in results.items() if records]
            
            print(f"Tables with data: {len(tables_with_data)}")
            for table in tables_with_data:
                record_count = len(results[table])
                print(f"  - {table}: {record_count} records")
                
                # Print a sample of the first record
                if record_count > 0:
                    first_record = results[table][0]
                    sample_fields = dict(list(first_record.items())[:5])
                    print(f"    Sample fields: {sample_fields}")
            
            # Done with test
            return
        
        # Only show tips in regular mode (not debug or demo)
        if not DEBUG_MODE and not DEMO_MODE:
            # Remove the tip message - it clutters the interface
            pass
        
        # Demo mode
        if DEMO_MODE:
            console.print("\n[bold]Demo Mode:[/bold] Here are some example questions you can ask:")
            demo_questions = [
                "What is Navy Federal Credit Union's total assets?",
                "How many members does PenFed have?",
                "What is the net income of Digital Federal Credit Union?",
                "Tell me about Golden 1 Credit Union in California",
                "What are the total loans for State Employees' Credit Union?"
            ]
            for i, question in enumerate(demo_questions, 1):
                console.print(f"[cyan]{i}.[/cyan] {question}")
        
        # Start the chat loop
        await chatbot.chat_loop()
        
    except Exception as e:
        if DEBUG_MODE:
            print(f"Error in main function: {str(e)}")
            import traceback
            traceback.print_exc()
        else:
            console.print("[red]An error occurred while starting the chatbot. Please try again.[/red]")
        sys.exit(1)

# Check if we're running in an interactive environment
if __name__ == "__main__":
    # Run the main function asynchronously without technical messages
    asyncio.run(main())