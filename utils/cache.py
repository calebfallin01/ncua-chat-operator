"""
Cache module for storing and retrieving frequently used data.

This module provides caching functionality to improve performance by storing
the results of expensive operations like database queries.
"""

import os
import json
import time
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps

logger = logging.getLogger(__name__)

class MetricsCache:
    """A simple in-memory and disk-based cache for financial metrics."""
    
    def __init__(self, cache_dir: str = ".cache", ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            cache_dir: Directory to store cached files
            ttl: Time to live in seconds (default: 1 hour)
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.memory_cache = {}
        
        # Create the cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Initialized metrics cache in {cache_dir}")

    def _get_cache_file_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key (e.g., cu_number)
            
        Returns:
            Cached value or None if not found or expired
        """
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                logger.info(f"Cache hit (memory) for {key}")
                return entry["data"]
            else:
                # Expired
                logger.info(f"Cache expired (memory) for {key}")
                del self.memory_cache[key]
        
        # Check file cache
        cache_file = self._get_cache_file_path(key)
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    entry = json.load(f)
                
                if time.time() - entry["timestamp"] < self.ttl:
                    # Update memory cache
                    self.memory_cache[key] = entry
                    logger.info(f"Cache hit (disk) for {key}")
                    return entry["data"]
                else:
                    # Expired
                    logger.info(f"Cache expired (disk) for {key}")
                    try:
                        os.unlink(cache_file)
                    except Exception as e:
                        logger.warning(f"Failed to delete expired cache file {cache_file}: {e}")
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Store a value in the cache.
        
        Args:
            key: Cache key (e.g., cu_number)
            value: Value to cache
        """
        entry = {
            "timestamp": time.time(),
            "data": value
        }
        
        # Update memory cache
        self.memory_cache[key] = entry
        
        # Update file cache
        cache_file = self._get_cache_file_path(key)
        try:
            with open(cache_file, 'w') as f:
                json.dump(entry, f)
            logger.info(f"Cached data for {key}")
        except Exception as e:
            logger.warning(f"Failed to write cache file {cache_file}: {e}")

    def invalidate(self, key: str) -> bool:
        """
        Remove a key from the cache.
        
        Args:
            key: Cache key to invalidate
            
        Returns:
            True if the key was removed, False otherwise
        """
        was_removed = False
        
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
            was_removed = True
        
        # Remove from file cache
        cache_file = self._get_cache_file_path(key)
        if os.path.exists(cache_file):
            try:
                os.unlink(cache_file)
                was_removed = True
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        if was_removed:
            logger.info(f"Invalidated cache for {key}")
        
        return was_removed

def cached(cache_instance: MetricsCache):
    """
    Decorator for caching function results.
    
    This decorator is intended for functions that take a cu_number as their first argument.
    
    Args:
        cache_instance: Instance of MetricsCache to use
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, cu_number: str, *args, **kwargs):
            # Use cu_number as the cache key
            # Normalize the cu_number to ensure consistent caching
            normalized_cu_number = cu_number
            if isinstance(cu_number, str) and '.' in cu_number:
                try:
                    normalized_cu_number = str(int(float(cu_number)))
                except ValueError:
                    pass
            
            # Check if we have a cached result
            cached_result = cache_instance.get(normalized_cu_number)
            if cached_result is not None:
                return cached_result
            
            # Call the original function
            result = func(self, cu_number, *args, **kwargs)
            
            # Cache the result if it's not None
            if result is not None:
                cache_instance.set(normalized_cu_number, result)
            
            return result
        
        @wraps(func)
        async def async_wrapper(self, cu_number: str, *args, **kwargs):
            # Use cu_number as the cache key
            # Normalize the cu_number to ensure consistent caching
            normalized_cu_number = cu_number
            if isinstance(cu_number, str) and '.' in cu_number:
                try:
                    normalized_cu_number = str(int(float(cu_number)))
                except ValueError:
                    pass
            
            # Check if we have a cached result
            cached_result = cache_instance.get(normalized_cu_number)
            if cached_result is not None:
                return cached_result
            
            # Call the original function
            result = await func(self, cu_number, *args, **kwargs)
            
            # Cache the result if it's not None
            if result is not None:
                cache_instance.set(normalized_cu_number, result)
            
            return result
            
        # Return the appropriate wrapper based on whether the function is async
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator 