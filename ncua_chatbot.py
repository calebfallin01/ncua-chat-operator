#!/usr/bin/env python3
"""
NCUA Chatbot - A simplified chatbot for querying credit unions by name
"""

import os
import json
import asyncio
import subprocess
import logging
import argparse
import re
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv
import openai
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

# Parse command line arguments
parser = argparse.ArgumentParser(description="NCUA Credit Union Chatbot")
parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
args = parser.parse_args()

# Set debug mode
DEBUG_MODE = args.debug

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ncua_chatbot.log"),
        logging.StreamHandler() if DEBUG_MODE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set up Rich console for pretty output
console = Console()

# Check for environment file
env_file = ".env"
if not os.path.exists(env_file) and DEBUG_MODE:
    print(f"Warning: .env file not found at {env_file}. Make sure you have the required API keys set.")

class ConversationContext:
    """Class to maintain conversation context and state across interactions."""
    
    def __init__(self):
        """Initialize the conversation context."""
        # Currently active credit union info
        self.current_cu_info = None
        
    def set_active_credit_union(self, cu_info: Dict[str, Any]):
        """Set the currently active credit union in the conversation."""
        if not cu_info:
            return False
            
        # Store current credit union info
        self.current_cu_info = cu_info
        return True
    
    def get_active_credit_union(self) -> Optional[Dict[str, Any]]:
        """Get the currently active credit union info."""
        return self.current_cu_info
        
    def has_context(self) -> bool:
        """Check if there is an active conversation context."""
        return self.current_cu_info is not None
    
    def clear_context(self):
        """Clear the current conversation context."""
        self.current_cu_info = None

class NCUAChatbot:
    """Interactive chatbot for querying credit unions by name."""
    
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
        
        # Initialize conversation context tracker
        self.context = ConversationContext()
        
        # Create welcome message
        console.print(Panel.fit(
            "[bold blue]NCUA Credit Union Chatbot[/bold blue]\n"
            "Ask questions about credit unions.",
            title="Welcome"
        ))
    
    def generate_account_search_query(self, user_query: str) -> str:
        """
        Generate an enriched parenthetical search query for accounts_vector_search.py
        
        Args:
            user_query: User's original query about financial metrics
            
        Returns:
            Enriched search query with synonyms and related terms
        """
        try:
            # Using OpenAI to generate an enriched search query
            messages = [
                {
                    "role": "system",
                    "content": """
You are an expert financial analyst specializing in credit union call report data and NCUA terminology. Your task is to convert user questions into focused search queries that will find the most relevant financial metrics in the NCUA database.

Your goal is to create PRECISE search queries optimized for matching against account descriptions (the 'Description' field) which contain detailed explanations of what each financial metric represents.

The search query will be used to:
1. First determine the financial category the query belongs to (e.g., Assets, Loans, Income, etc.)
2. Then search within that category for the specific metric

For effective matching against account descriptions:
1. Focus on the core financial concept (e.g., "total assets", "net income", "delinquent loans")  
2. Include only the most relevant related terms
3. Use common measurement terms found in description fields (e.g., "total", "amount", "number of")
4. Keep the query concise and focused (10-15 terms maximum)

Return a space-separated list of relevant terms without explanations.
"""
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]
            
            response = openai.chat.completions.create(
                model="gpt-4-turbo",  # Using GPT-4 for better query understanding
                messages=messages,
                temperature=0.3,  # Lowered for more consistent output
                max_tokens=100    # Reduced significantly to ensure concise output
            )
            
            search_query = response.choices[0].message.content.strip()
            logger.info(f"Generated account search query: '{search_query}'")
            
            return search_query
            
        except Exception as e:
            logger.error(f"Error generating account search query: {str(e)}")
            # Fallback to simple keyword extraction
            return self._extract_basic_search_terms(user_query)
    
    def _extract_basic_search_terms(self, query: str) -> str:
        """
        Basic fallback method to extract search terms from a user query,
        enhanced to include likely descriptive terms used in Description field.
        
        Args:
            query: User's query
            
        Returns:
            Simple search terms based on common financial keywords
        """
        query_lower = query.lower()
        
        # Check for common financial metrics with descriptive terms
        if any(term in query_lower for term in ["asset", "assets", "size"]):
            return "total assets asset size value of assets report the dollar amount of assets Other Assets"
        elif any(term in query_lower for term in ["member", "members", "membership"]):
            return "number of members total members member count membership Miscellaneous Information"
        elif any(term in query_lower for term in ["loan", "loans", "lending"]):
            return "total loans dollar amount of loans loan balance Loans Specialized Lending"
        elif any(term in query_lower for term in ["income", "profit", "earnings", "revenue"]):
            return "net income earnings profit total income Income"
        elif any(term in query_lower for term in ["deposit", "deposits", "share", "shares"]):
            return "total shares deposits dollar amount of deposits share balances Shares/Deposits"
        elif any(term in query_lower for term in ["agriculture", "farm", "farming"]):
            return "agriculture farm loans agricultural purpose Specialized Lending"
        elif any(term in query_lower for term in ["delinquent", "delinquency", "late", "overdue"]):
            return "delinquent delinquency number of delinquent loans amount of delinquent loans Delinquency"
        elif any(term in query_lower for term in ["equity", "capital", "net worth"]):
            return "equity capital net worth total equity Net Worth Equity"
        elif any(term in query_lower for term in ["investment", "investments", "securities"]):
            return "investments value of investments securities Investments"
        elif any(term in query_lower for term in ["cash", "liquidity"]):
            return "cash cash equivalents cash on hand total cash Cash and Cash Equivalents"
        elif any(term in query_lower for term in ["expense", "expenses", "cost", "costs"]):
            return "operating expenses total expenses Expenses Cost of Funds"
        elif any(term in query_lower for term in ["cuso", "service organization"]):
            return "credit union service organization CUSO investment in CUSO CUSO"
        elif any(term in query_lower for term in ["charge off", "charge-off", "writeoff", "write-off"]):
            return "charge-offs loans losses Charge Offs and Recoveries"
        else:
            # For unknown queries, use a broader set of financial terms
            return "total dollar amount financial metrics account Miscellaneous Information"
            
    def query_accounts_vector_search(self, search_query: str) -> Optional[Dict[str, Any]]:
        """
        Query the accounts_vector_search.py script with the given search query
        
        Args:
            search_query: Search query to use with accounts_vector_search.py
            
        Returns:
            Dictionary of search results or None if failed
        """
        try:
            # Format the search query in curly braces as expected by accounts_vector_search.py
            formatted_query = f"{{{search_query}}}"
            
            # Build and execute the command
            cmd = ["python", "accounts_vector_search.py", "--input", formatted_query]
            
            if DEBUG_MODE:
                console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
                
            # Run the command
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Error running accounts_vector_search.py: {result.stderr}")
                return None
                
            # Parse the JSON output
            try:
                output_data = json.loads(result.stdout)
                logger.info(f"accounts_vector_search.py returned: {output_data}")
                return output_data
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON returned by accounts_vector_search.py: {result.stdout}")
                return None
                
        except Exception as e:
            logger.error(f"Error querying accounts_vector_search.py: {str(e)}")
            return None

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
            abbreviations = {'fcu': 'FCU', 'cu': 'CU'}
            
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
        
        Args:
            user_query: The user's query about a credit union
            
        Returns:
            Dictionary with credit union information or None if not found
        """
        try:
            # Extract credit union name and location information from query
            credit_union_name, location_context = self.extract_credit_union_info(user_query)
            
            if not credit_union_name:
                # If no explicit credit union name and we have context, use the active context
                if self.context.has_context():
                    if DEBUG_MODE:
                        console.print("[dim]No credit union name found, using existing context[/dim]")
                    return self.context.get_active_credit_union()
                
                return None
            
            # Find the credit union in the database
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
        Process a user query about a credit union.
        
        Args:
            user_query: The user's query about a credit union
            
        Returns:
            Information about the found credit union and any requested financial data
        """
        try:
            logger.info(f"Processing query: '{user_query}'")
            
            # Extract credit union name from query
            cu_info = self.extract_credit_union_from_query(user_query)
            
            if not cu_info:
                logger.warning("No credit union identified in the query")
                return "I couldn't identify a specific credit union in your question. Please specify which credit union you're asking about."
            
            # Format credit union name
            cu_name = self.format_credit_union_name(cu_info.get("cu_name", "Unknown"))
            cu_number = cu_info.get("cu_number", "Unknown")
            
            # Check if the query is asking about financial metrics
            # First, generate an enriched search query for accounts_vector_search.py
            search_query = self.generate_account_search_query(user_query)
            
            if search_query:
                # Query the accounts vector search with the enriched query
                logger.info(f"Querying accounts_vector_search.py with: {search_query}")
                account_results = self.query_accounts_vector_search(search_query)
                
                if account_results:
                    # Format the account results
                    result_message = f"Found credit union: {cu_name} (ID: {cu_number})\n\n"
                    
                    # Check for errors in the response
                    if "error" in account_results:
                        result_message += f"Error searching for financial metrics: {account_results['error']}"
                    else:
                        # Process each query result
                        for query_term, data in account_results.items():
                            if "error" in data:
                                result_message += f"Could not find information for '{query_term}': {data['error']}\n"
                            else:
                                # Extract account information using the new schema
                                code = data.get("Code", "Unknown")
                                description = data.get("Description", "Unknown")
                                category = data.get("Category", "Unknown")
                                type_value = data.get("Type", "Unknown")
                                
                                # Add formatted result
                                result_message += f"**Financial Metric**: {description}\n"
                                result_message += f"**Code**: {code}\n"
                                result_message += f"**Category**: {category}\n"
                                if type_value and type_value.lower() != "unknown":
                                    result_message += f"**Type**: {type_value}\n"
                                result_message += "\n"
                    
                    return result_message
            
            # If we didn't get account results or didn't try, return basic information
            return f"Found credit union: {cu_name} (ID: {cu_number})"
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "I encountered an error processing your question. Please try again."

    async def chat_loop(self):
        """Run the interactive chat loop."""
        while True:
            try:
                # Display active context if available, but only show the credit union name
                prompt_text = "\nYour question"
                if self.context.has_context():
                    active_cu = self.context.get_active_credit_union()
                    if active_cu:
                        cu_name = self.format_credit_union_name(active_cu.get("cu_name", "Unknown"))
                        prompt_text = f"\nYour question about [cyan]{cu_name}[/cyan]"
                
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
                
                # Show a loading indicator
                with console.status("[cyan]Searching for information...[/cyan]", spinner="dots"):
                    # Process the query
                    answer = await self.process_query(user_query)
                
                # Display the answer
                console.print(Panel(Markdown(answer), title="Result", border_style="green"))
                    
            except KeyboardInterrupt:
                console.print("\n[green]Goodbye![/green]")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                console.print("[red]An error occurred. Please try again.[/red]")

async def main():
    """Main function"""
    try:
        # Initialize chatbot
        chatbot = NCUAChatbot()
        
        # Start the chat loop
        await chatbot.chat_loop()
        
    except Exception as e:
        console.print(f"[red]An error occurred while starting the chatbot: {str(e)}[/red]")

# Entry point
if __name__ == "__main__":
    # Run the main function asynchronously
    asyncio.run(main())