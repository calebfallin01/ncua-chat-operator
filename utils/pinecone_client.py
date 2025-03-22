import os
import logging
from pinecone import Pinecone
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class PineconeClient:
    """Client for interacting with Pinecone vector database."""
    
    def __init__(self):
        """Initialize the Pinecone client."""
        try:
            # Get API key from environment variable
            self.api_key = os.environ.get("PINECONE_API_KEY")
            if not self.api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set")
            
            # Initialize Pinecone client
            logger.info("Initializing Pinecone client...")
            self.pc = Pinecone(api_key=self.api_key)
            
            # Get the index
            self.index_name = os.environ.get("PINECONE_INDEX")
            if not self.index_name:
                raise ValueError("PINECONE_INDEX environment variable not set")
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
            # Get stats to verify connection
            stats = self.index.describe_index_stats()
            self.vector_dimension = stats.dimension
            logger.info(f"Vector dimension: {self.vector_dimension}")
                
        except Exception as e:
            logger.error(f"Error initializing Pinecone client: {str(e)}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def query(self, embedding: list, top_k: int = 5) -> list:
        """
        Search for the most similar vectors to the provided embedding.
        
        Args:
            embedding: The query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of match objects with id, score, and metadata
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
            
            # Using the Pinecone v6 API
            query_response = self.index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            logger.info(f"Query response received with {len(query_response.matches)} matches")
            return query_response.matches
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            return [] 

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def search_by_metadata(self, field, value, top_k=10, combine_with_vector=False):
        """
        Search for vectors by metadata field.
        
        Args:
            field (str): The metadata field to search in.
            value (str): The value to search for.
            top_k (int): Maximum number of results to return.
            combine_with_vector (bool): Whether to combine with a vector search.
            
        Returns:
            list: A list of matches.
        """
        try:
            # Log the search attempt
            logger.info(f"Searching for records with {field}={value}")
            
            # Create a filter dictionary
            filter_dict = {}
            
            # Handle array fields differently (tradename could be a list)
            if field == "tradename":
                # Array contains logic
                filter_dict = {
                    "$or": [
                        {field: value},  # Exact match (string case)
                        {f"{field}.$": value}  # Array contains (list case)
                    ]
                }
            else:
                # For standard fields, just use equality
                filter_dict[field] = value
            
            # Create a dummy vector for metadata-only search
            dummy_vector = [0.0] * self.vector_dimension
            
            # Log the filter
            logger.info(f"Using filter: {filter_dict}")
            
            # Query with metadata filter only
            matches = self.index.query(
                vector=dummy_vector,
                filter=filter_dict,
                top_k=top_k,
                include_metadata=True
            )
            
            logger.info(f"Found {len(matches.matches)} matches with {field}={value}")
            
            return matches.matches
            
        except Exception as e:
            logger.error(f"Error in metadata search: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def comprehensive_search(self, query, vector=None, top_k=20, min_score=0.0, context=None):
        """
        Search across all fields (cu_name, tradename, domain_root) and by vector similarity.
        
        Args:
            query (str): The search query string.
            vector (list): Optional embedding vector for similarity search.
            top_k (int): Maximum number of results to return.
            min_score (float): Minimum similarity score threshold.
            context (dict): Optional context information to prioritize results.
                           Can include: city, state_code, state_full
            
        Returns:
            list: List of matches with search metadata added.
        """
        try:
            # Normalize the query for consistent matching
            query_lower = query.lower().strip()
            logger.info(f"Performing comprehensive search for: '{query_lower}'")
            
            # Log context if provided
            if context:
                logger.info(f"Search context provided: {context}")
                # Normalize context values for consistent matching
                normalized_context = {k: v.lower().strip() if isinstance(v, str) else v 
                                     for k, v in context.items()}
            else:
                normalized_context = {}
                
            # Create a dummy vector for metadata-only search if none provided
            if vector:
                logger.info("Using provided vector for search")
                search_vector = vector
            else:
                logger.info("Using dummy vector for metadata-only search")
                search_vector = [0.0] * self.vector_dimension
            
            # FIRST ATTEMPT: Try looking for this query in the tradename field directly
            try:
                logger.info(f"Attempting direct tradename search for '{query_lower}'")
                
                # First approach - try exact match with the $in operator
                # This will look for tradename arrays containing the exact query
                tradename_filter = {
                    "tradename": {"$in": [query_lower, query.strip()]}  # Try both lowercase and original case
                }
                
                # Direct tradename match query
                direct_matches_result = self.index.query(
                    vector=search_vector,
                    filter=tradename_filter,
                    top_k=top_k,
                    include_metadata=True
                )
                
                direct_matches = direct_matches_result.matches
                logger.info(f"Found {len(direct_matches)} direct tradename matches for '{query_lower}'")
                
                # If found exact tradename matches, prioritize these
                if direct_matches:
                    # Add match information to these matches
                    for match in direct_matches:
                        metadata = match.metadata
                        cu_name = metadata.get('cu_name', 'N/A')
                        
                        match._match_type = "TRADENAME"
                        match._field_matched = "tradename"
                        match._match_reason = f"Direct tradename match for '{query_lower}'"
                        # Very high score for direct tradename matches
                        match._adjusted_score = 0.98
                        logger.info(f"Direct tradename match: {cu_name}")
                    
                    # Sort by adjusted score
                    direct_matches.sort(key=lambda m: m._adjusted_score, reverse=True)
                    logger.info(f"Returning {len(direct_matches)} direct tradename matches")
                    return direct_matches
                
                # SECOND ATTEMPT: Try looking for tradenames with case-insensitive matching
                # This is needed for cases like "caped" matching "CapEd"
                logger.info(f"Trying case-insensitive tradename search for '{query_lower}'")
                
                # We need to do this through post-processing with a larger result set
                initial_candidates = self.index.query(
                    vector=search_vector,
                    top_k=50,  # Get a larger set to find matches
                    include_metadata=True
                )
                
                # Debug - print ALL candidate credit unions
                logger.info(f"All candidates: {[match.metadata.get('cu_name', 'N/A') for match in initial_candidates.matches]}")
                
                # Special check for caped and Capital Educators
                for match in initial_candidates.matches:
                    cu_name = match.metadata.get('cu_name', 'N/A')
                    if cu_name == "CAPITAL EDUCATORS":
                        logger.info(f"Found CAPITAL EDUCATORS in candidates")
                        tradename_raw = match.metadata.get('tradename', None)
                        domain_root = match.metadata.get('domain_root', '')
                        logger.info(f"CAPITAL EDUCATORS tradename: {tradename_raw}")
                        logger.info(f"CAPITAL EDUCATORS domain: {domain_root}")
                
                # Post-process to find case-insensitive tradename matches
                tradename_matches = []
                
                # Check all candidates for potential matches
                for match in initial_candidates.matches:
                    metadata = match.metadata
                    cu_name = metadata.get('cu_name', 'N/A')
                    domain_root = metadata.get('domain_root', '').lower()
                    tradename_raw = metadata.get('tradename', None)
                    
                    # First check domains
                    if domain_root:
                        # If domain contains the query
                        if query_lower in domain_root:
                            match._match_type = "DOMAIN"
                            match._field_matched = "domain_root"
                            match._match_reason = f"Domain contains query: '{domain_root}' contains '{query_lower}'"
                            match._adjusted_score = 0.94
                            tradename_matches.append(match)
                            logger.info(f"Domain contains query: {cu_name} - {domain_root}")
                            continue
                    
                    # Check tradenames
                    if tradename_raw:
                        # Convert to list format
                        tradenames = []
                        if isinstance(tradename_raw, list):
                            # Keep original case for display, but use lowercase for comparison
                            tradenames = [(str(t), str(t).lower()) for t in tradename_raw if t]
                        else:
                            tradenames = [(str(tradename_raw), str(tradename_raw).lower())]
                        
                        # For each tradename (original, lowercase)
                        for original, lowercase in tradenames:
                            # Case-insensitive exact match - this catches "caped" == "CapEd"
                            if lowercase == query_lower:
                                match._match_type = "TRADENAME"
                                match._field_matched = "tradename"
                                match._match_reason = f"Exact tradename match (case-insensitive): '{query_lower}' = '{original}'"
                                match._adjusted_score = 0.97
                                tradename_matches.append(match)
                                logger.info(f"Case-insensitive tradename exact match: {cu_name} - {original}")
                                break
                
                # If we have matches, return them
                if tradename_matches:
                    # Sort by adjusted score
                    tradename_matches.sort(key=lambda m: m._adjusted_score, reverse=True)
                    logger.info(f"Returning {len(tradename_matches)} case-insensitive tradename matches")
                    return tradename_matches
                    
                # THIRD ATTEMPT: Try looking for this query in domain_root
                # This is critical for common abbreviations like p1fcu, capedcu, etc.
                logger.info(f"Attempting direct domain search for '{query_lower}'")
                
                # Try exact domain match first
                domain_exact_filter = {
                    "domain_root": query_lower  # Exact match on domain_root
                }
                
                domain_matches_result = self.index.query(
                    vector=search_vector,
                    filter=domain_exact_filter,
                    top_k=top_k,
                    include_metadata=True
                )
                
                domain_matches = domain_matches_result.matches
                logger.info(f"Found {len(domain_matches)} exact domain matches for '{query_lower}'")
                
                # Special case handling without hardcoding knowledge
                # For common patterns like "caped" → "capedcu", "abc" → "abcfcu", etc.
                # Look for common credit union domain patterns
                common_patterns = [f"{query_lower}cu", f"{query_lower}fcu", f"{query_lower}-cu"]
                
                for pattern in common_patterns:
                    logger.info(f"Trying domain pattern search for '{pattern}'")
                    pattern_filter = {
                        "domain_root": pattern
                    }
                    pattern_matches_result = self.index.query(
                        vector=search_vector,
                        filter=pattern_filter,
                        top_k=top_k,
                        include_metadata=True
                    )
                    
                    # If found, add to exact domain matches
                    if pattern_matches_result.matches:
                        logger.info(f"Found {len(pattern_matches_result.matches)} matches for pattern '{pattern}'")
                        # Process these matches same as exact matches
                        for match in pattern_matches_result.matches:
                            metadata = match.metadata
                            cu_name = metadata.get('cu_name', 'N/A')
                            domain_root = metadata.get('domain_root', '')
                            
                            match._match_type = "DOMAIN"
                            match._field_matched = "domain_root"
                            match._match_reason = f"Common pattern match: '{query_lower}' → '{domain_root}'"
                            match._adjusted_score = 0.96
                            domain_matches.append(match)
                            logger.info(f"Domain pattern match: {cu_name} - {domain_root}")
                
                # If found exact domain matches, prioritize these
                if domain_matches:
                    # Add match information to these matches
                    for match in domain_matches:
                        metadata = match.metadata
                        cu_name = metadata.get('cu_name', 'N/A')
                        domain_root = metadata.get('domain_root', '')
                        
                        match._match_type = "DOMAIN"
                        match._field_matched = "domain_root"
                        match._match_reason = f"Exact domain match: '{query_lower}' = '{domain_root}'"
                        # Very high score for direct domain matches
                        match._adjusted_score = 0.97
                        logger.info(f"Found exact domain match: {cu_name} - {domain_root}")
                    
                    # Sort by adjusted score
                    domain_matches.sort(key=lambda m: m._adjusted_score, reverse=True)
                    logger.info(f"Returning {len(domain_matches)} exact domain matches")
                    return domain_matches
                    
                # FOURTH ATTEMPT: Look for domains starting with the query
                # For cases like 'caped' matching 'capedcu'
                logger.info(f"Looking for domains starting with or containing '{query_lower}'")
                domain_matches = []
                
                # We need to do this through post-processing
                if not initial_candidates:  # If we haven't fetched candidates yet
                    initial_candidates = self.index.query(
                        vector=search_vector,
                        top_k=50,  # Get a larger set to find matches
                        include_metadata=True
                    )
                
                # Post-process to find domains that start with or contain the query
                for match in initial_candidates.matches:
                    metadata = match.metadata
                    cu_name = metadata.get('cu_name', 'N/A')
                    domain_root = metadata.get('domain_root', '').lower()
                    tradename_raw = metadata.get('tradename', None)
                    
                    # Process domain matches
                    if domain_root:
                        # Check if domain starts with query
                        if domain_root.startswith(query_lower):
                            match._match_type = "DOMAIN"
                            match._field_matched = "domain_root"
                            match._match_reason = f"Domain starts with query: '{domain_root}' starts with '{query_lower}'"
                            match._adjusted_score = 0.96
                            domain_matches.append(match)
                            logger.info(f"Domain prefix match: {cu_name} - {domain_root}")
                            continue
                            
                    # Process tradename matches
                    if tradename_raw:
                        tradenames = []
                        if isinstance(tradename_raw, list):
                            tradenames = [str(t).lower() for t in tradename_raw if t]
                        else:
                            tradenames = [str(tradename_raw).lower()]
                        
                        # Check if any tradename looks like the query (regardless of case)
                        for tradename in tradenames:
                            # Check if tradename matches query without case sensitivity
                            if tradename and tradename.lower() == query_lower.lower():
                                match._match_type = "TRADENAME"
                                match._field_matched = "tradename"
                                match._match_reason = f"Tradename match: '{tradename}' matches '{query_lower}'"
                                match._adjusted_score = 0.95
                                domain_matches.append(match)
                                logger.info(f"Case-insensitive tradename match: {cu_name} - {tradename}")
                                continue
                
                if domain_matches:
                    # Sort by adjusted score
                    domain_matches.sort(key=lambda m: m._adjusted_score, reverse=True)
                    logger.info(f"Returning {len(domain_matches)} domain/tradename prefix matches")
                    return domain_matches
                
                # FIFTH ATTEMPT: Comprehensive search for partial matches
                # Look for any partial matches in tradenames or domains
                logger.info(f"Attempting comprehensive partial match search for '{query_lower}'")
                
                post_process_matches = []
                
                # If we haven't loaded candidates yet, do it now
                if not initial_candidates:
                    initial_candidates = self.index.query(
                        vector=search_vector,
                        top_k=50,  # Get more candidates to find potential matches
                        include_metadata=True
                    )
                
                # Process all the matches for any kind of partial match
                for match in initial_candidates.matches:
                    metadata = match.metadata
                    cu_name = metadata.get('cu_name', 'N/A')
                    domain_root = metadata.get('domain_root', '').lower()
                    tradename_raw = metadata.get('tradename', None)
                    
                    # Check domain_root for potential matches
                    if domain_root:
                        # Check for exact domain match
                        if domain_root == query_lower:
                            match._match_type = "DOMAIN"
                            match._field_matched = "domain_root"
                            match._match_reason = f"Exact domain match: '{domain_root}' = '{query_lower}'"
                            match._adjusted_score = 0.97
                            post_process_matches.append(match)
                            logger.info(f"Domain exact match: {cu_name} - {domain_root}")
                            continue
                            
                        # Check if domain contains query
                        if query_lower in domain_root:
                            match._match_type = "DOMAIN"
                            match._field_matched = "domain_root"
                            match._match_reason = f"Domain contains query: '{domain_root}' contains '{query_lower}'"
                            match._adjusted_score = 0.96
                            post_process_matches.append(match)
                            logger.info(f"Domain contains match: {cu_name} - {domain_root}")
                            continue
                            
                        # Check if domain starts with query
                        if domain_root.startswith(query_lower):
                            match._match_type = "DOMAIN"
                            match._field_matched = "domain_root"
                            match._match_reason = f"Domain starts with query: '{domain_root}' starts with '{query_lower}'"
                            match._adjusted_score = 0.95
                            post_process_matches.append(match)
                            logger.info(f"Domain prefix match: {cu_name} - {domain_root}")
                            continue
                            
                        # Special handling for common cases like "caped" and "capedcu"
                        for domain_test in [f"{query_lower}cu", f"{query_lower}-cu", f"{query_lower}fcu"]:
                            if domain_root == domain_test:
                                match._match_type = "DOMAIN"
                                match._field_matched = "domain_root"
                                match._match_reason = f"Domain pattern match: '{domain_root}' = '{domain_test}'"
                                match._adjusted_score = 0.94
                                post_process_matches.append(match)
                                logger.info(f"Domain pattern match: {cu_name} - {domain_root}")
                                continue
                    
                    # Check tradenames for partial matches
                    if tradename_raw:
                        # Convert to list format
                        tradenames = []
                        if isinstance(tradename_raw, list):
                            tradenames = [str(t).lower() if t else '' for t in tradename_raw if t]
                        else:
                            tradenames = [str(tradename_raw).lower()]
                        
                        for tradename in tradenames:
                            # Exact tradename match
                            if tradename and tradename.lower() == query_lower:
                                match._match_type = "TRADENAME"
                                match._field_matched = "tradename"
                                match._match_reason = f"Exact tradename match: '{tradename}' = '{query_lower}'"
                                match._adjusted_score = 0.98
                                post_process_matches.append(match)
                                logger.info(f"Exact tradename match: {cu_name} - {tradename}")
                                break
                                
                            # Partial tradename match
                            if tradename and (query_lower in tradename or tradename in query_lower):
                                match._match_type = "TRADENAME"
                                match._field_matched = "tradename"
                                if query_lower in tradename:
                                    match._match_reason = f"Tradename contains query: '{tradename}' contains '{query_lower}'"
                                else:
                                    match._match_reason = f"Query contains tradename: '{query_lower}' contains '{tradename}'"
                                match._adjusted_score = 0.93
                                post_process_matches.append(match)
                                logger.info(f"Partial tradename match: {cu_name} - {tradename}")
                                break
                
                if post_process_matches:
                    # Sort by adjusted score
                    post_process_matches.sort(key=lambda m: m._adjusted_score, reverse=True)
                    logger.info(f"Returning {len(post_process_matches)} comprehensive partial matches")
                    return post_process_matches
                
            except Exception as e:
                logger.error(f"Error in direct search: {str(e)}")
                logger.error(f"Continuing with standard search")
            
            # If no direct matches or the search failed, fall back to regular search
            logger.info("Proceeding with standard vector search")
            
            # Query with no filter to get all potential matches
            response = self.index.query(
                vector=search_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            logger.info(f"Retrieved {len(response.matches)} potential matches")
            
            # For debugging purposes, log basic info about the top 5 matches
            logger.info("Top 5 raw matches:")
            for i, match in enumerate(response.matches[:5]):
                metadata = match.metadata
                cu_name = metadata.get('cu_name', 'N/A')
                cu_number = metadata.get('cu_number', 'N/A')
                tradename = metadata.get('tradename', 'N/A')
                domain = metadata.get('domain_root', 'N/A')
                logger.info(f"  {i+1}. {cu_name} (Score: {match.score:.3f}, CU#: {cu_number})")
                logger.info(f"     - Tradename: {tradename}")
                logger.info(f"     - Domain: {domain}")
            
            # If no matches were found at all, return empty list
            if not response.matches:
                logger.info("No matches found in vector search")
                return []
            
            # Post-process matches to add match type information
            enhanced_matches = []
            for match in response.matches:
                match_metadata = match.metadata
                match_score = match.score
                match_type = "unknown"
                field_matched = None
                match_reason = "No specific reason"
                
                # Calculate adjusted score based on which field matched
                adjusted_score = match_score
                
                # Get the values from the metadata for easier access
                cu_name = str(match_metadata.get('cu_name', '')).lower()
                
                # TRADENAME CHECK
                # Make tradename into a list for consistent processing
                tradename_raw = match_metadata.get('tradename', None)
                # Debug log tradename raw value
                logger.info(f"Raw tradename for {cu_name}: {tradename_raw} (type: {type(tradename_raw).__name__})")
                
                tradenames = []
                if isinstance(tradename_raw, list):
                    tradenames = [str(t).lower() if t else '' for t in tradename_raw]
                    logger.info(f"Processed tradename list for {cu_name}: {tradenames}")
                elif tradename_raw:
                    tradenames = [str(tradename_raw).lower()]
                    logger.info(f"Processed tradename string for {cu_name}: {tradenames}")
                
                # DOMAIN CHECK
                domain_root = str(match_metadata.get('domain_root', '')).lower()
                
                # Debug log
                logger.debug(f"Processing match: {cu_name}")
                logger.debug(f"  Tradenames: {tradenames}")
                logger.debug(f"  Domain: {domain_root}")
                
                # More lenient matching - match substrings in any direction
                
                # PRIMARY NAME MATCHING (Higher standards - requires exact substring match)
                if query_lower in cu_name:
                    match_type = "PRIMARY_NAME"
                    field_matched = "cu_name"
                    match_reason = f"Query '{query_lower}' found in cu_name '{cu_name}'"
                    # Boost score for primary name match
                    adjusted_score = max(adjusted_score, 0.95)
                
                # TRADENAME MATCHING (check all tradenames)
                for tradename in tradenames:
                    if tradename and (query_lower in tradename or 
                                     tradename in query_lower or  # This direction is also important
                                     any(query_part in tradename for query_part in query_lower.split()) or
                                     any(tname_part in query_lower for tname_part in tradename.split())):
                        match_type = "TRADENAME"
                        field_matched = "tradename"
                        match_reason = f"Match between query '{query_lower}' and tradename '{tradename}'"
                        # Boost score for tradename match - give it a high score to prioritize tradename matches
                        adjusted_score = max(adjusted_score, 0.95)
                        logger.info(f"Found tradename match: '{tradename}' for query '{query_lower}'")
                        break
                
                # DOMAIN MATCHING - much more aggressive matching
                # For many credit unions, their abbreviation is in the domain (e.g., p1fcu, capedcu)
                if domain_root:
                    # Full domain root match
                    if query_lower == domain_root:
                        match_type = "DOMAIN"
                        field_matched = "domain_root"
                        match_reason = f"Exact match between query '{query_lower}' and domain '{domain_root}'"
                        adjusted_score = max(adjusted_score, 0.97)  # Very high score for exact domain match
                        logger.info(f"Found exact domain match: '{domain_root}' for query '{query_lower}'")
                    # Domain root contains query (caped in capedcu)
                    elif query_lower in domain_root:
                        match_type = "DOMAIN"
                        field_matched = "domain_root"
                        match_reason = f"Query '{query_lower}' found in domain '{domain_root}'"
                        adjusted_score = max(adjusted_score, 0.95)  # High score
                        logger.info(f"Found domain contains query match: '{domain_root}' for query '{query_lower}'")
                    # Query contains domain root (this is less likely to be relevant)
                    elif domain_root in query_lower:
                        match_type = "DOMAIN"
                        field_matched = "domain_root"
                        match_reason = f"Domain '{domain_root}' found in query '{query_lower}'"
                        adjusted_score = max(adjusted_score, 0.85)  # Moderate score
                        logger.info(f"Found query contains domain match: '{domain_root}' in '{query_lower}'")
                    # Special handling for domain abbreviations
                    # Example: p1fcu is related to potlatch no 1 financial
                    elif domain_root.startswith(query_lower) or query_lower.startswith(domain_root):
                        match_type = "DOMAIN"
                        field_matched = "domain_root"
                        match_reason = f"Domain '{domain_root}' shares prefix with query '{query_lower}'"
                        adjusted_score = max(adjusted_score, 0.90)  # Good score for prefix match
                        logger.info(f"Found domain prefix match: '{domain_root}' with '{query_lower}'")
                
                # Add match info to the result
                match._match_type = match_type
                match._field_matched = field_matched
                match._match_reason = match_reason
                match._adjusted_score = adjusted_score
                
                # Include all matches but favor field matches by boosting their scores
                enhanced_matches.append(match)
            
            # Sort by adjusted score (highest first)
            enhanced_matches.sort(key=lambda m: m._adjusted_score, reverse=True)
            
            # Apply context-based prioritization if context is provided
            if normalized_context:
                logger.info("Applying context-based prioritization to results")
                
                # Process each match for context relevance
                for match in enhanced_matches:
                    # Get the match metadata
                    metadata = match.metadata
                    match_score = match._adjusted_score
                    context_score = 0.0
                    context_reason = []
                    
                    # Check city match
                    if 'city' in normalized_context and metadata.get('city'):
                        city_context = normalized_context['city']
                        match_city = metadata.get('city', '').lower()
                        
                        # Exact city match
                        if city_context == match_city:
                            context_score += 0.05  # Significant boost for exact city match
                            context_reason.append(f"Exact city match: {match_city}")
                            logger.info(f"Context match: {metadata.get('cu_name')} is in city '{match_city}'")
                    
                    # Check state match (state_code)
                    if 'state_code' in normalized_context and metadata.get('state_code'):
                        state_code_context = normalized_context['state_code']
                        match_state_code = metadata.get('state_code', '').lower()
                        
                        if state_code_context == match_state_code:
                            context_score += 0.03  # Boost for state code match
                            context_reason.append(f"State code match: {match_state_code}")
                            logger.info(f"Context match: {metadata.get('cu_name')} is in state code '{match_state_code}'")
                    
                    # Check state match (state_full)
                    if 'state_full' in normalized_context and metadata.get('state_full'):
                        state_full_context = normalized_context['state_full']
                        match_state_full = metadata.get('state_full', '').lower()
                        
                        if state_full_context == match_state_full:
                            context_score += 0.03  # Boost for full state name match
                            context_reason.append(f"State match: {match_state_full}")
                            logger.info(f"Context match: {metadata.get('cu_name')} is in state '{match_state_full}'")
                    
                    # Apply context score boost to the match
                    if context_score > 0:
                        # Store original score for reference
                        match._pre_context_score = match._adjusted_score
                        
                        # Apply the context boost
                        match._adjusted_score += context_score
                        
                        # Add context reasons
                        if hasattr(match, '_match_reason'):
                            match._match_reason += " | " + " & ".join(context_reason)
                        else:
                            match._match_reason = " & ".join(context_reason)
                            
                        # Update match type to indicate context was used
                        if hasattr(match, '_match_type'):
                            match._match_type += "_WITH_CONTEXT"
                        else:
                            match._match_type = "CONTEXT"
                            
                        logger.info(f"Boosted score for {metadata.get('cu_name')} by {context_score} based on context")
                
                # Re-sort after applying context boosts
                enhanced_matches.sort(key=lambda m: m._adjusted_score, reverse=True)
                logger.info("Matches re-sorted based on context relevance")
            
            # If minimum score is set, filter matches by that score
            if min_score > 0:
                filtered_matches = [m for m in enhanced_matches 
                                   if m._adjusted_score >= min_score]
                logger.info(f"Found {len(filtered_matches)}/{len(enhanced_matches)} matches after filtering by min_score {min_score}")
                enhanced_matches = filtered_matches
            
            # Always return at least the top match if none met the criteria
            if not enhanced_matches and response.matches:
                # If there are matches but none above the threshold, return the top match anyway
                top_raw = response.matches[0]
                
                # Add match info to make it compatible with other results
                top_raw._match_type = "FALLBACK"
                top_raw._field_matched = "vector_similarity"
                top_raw._match_reason = "Fallback match - below threshold but best available"
                top_raw._adjusted_score = top_raw.score
                
                logger.info(f"No matches met criteria, returning top raw match as fallback: {top_raw.metadata.get('cu_name', 'N/A')}")
                return [top_raw]
            
            logger.info(f"Returning {len(enhanced_matches)} enhanced matches")
            return enhanced_matches
            
        except Exception as e:
            logger.error(f"Error in comprehensive search: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return [] 