"""Cache management for NG FX Predictor."""

import json
import pickle
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlparse

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..config import get_settings
from ..utils.logging import get_logger
from .exceptions import CacheError

logger = get_logger(__name__)


class InMemoryCache:
    """Simple in-memory cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize in-memory cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if key not in self._cache:
            return None
        
        item = self._cache[key]
        
        # Check if item has expired
        if item.get("expires_at") and datetime.now() > item["expires_at"]:
            self.delete(key)
            return None
        
        # Update access time
        self._access_times[key] = datetime.now()
        
        return item["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        # Evict old items if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        expires_at = None
        if ttl:
            expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self._cache[key] = {
            "value": value,
            "created_at": datetime.now(),
            "expires_at": expires_at,
        }
        self._access_times[key] = datetime.now()
    
    def delete(self, key: str) -> None:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
        """
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        # Find the least recently used key
        lru_key = min(self._access_times, key=self._access_times.get)
        self.delete(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        return {
            "type": "in_memory",
            "size": len(self._cache),
            "max_size": self.max_size,
            "keys": list(self._cache.keys()),
        }


class RedisCache:
    """Redis-based cache implementation."""
    
    def __init__(self, redis_url: str):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
        """
        if not REDIS_AVAILABLE:
            raise CacheError("Redis not available. Install redis package.")
        
        try:
            parsed_url = urlparse(redis_url)
            self.redis_client = redis.Redis(
                host=parsed_url.hostname,
                port=parsed_url.port or 6379,
                db=int(parsed_url.path.lstrip('/')) if parsed_url.path else 0,
                password=parsed_url.password,
                decode_responses=False,  # Handle binary data
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {e}")
            raise CacheError(f"Redis connection failed: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        try:
            data = self.redis_client.get(key)
            if data is None:
                return None
            
            # Try to deserialize as JSON first, then pickle
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return pickle.loads(data)
                
        except Exception as e:
            logger.warning(f"Failed to get cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        try:
            # Try to serialize as JSON first, then pickle
            try:
                data = json.dumps(value).encode('utf-8')
            except (TypeError, ValueError):
                data = pickle.dumps(value)
            
            self.redis_client.set(key, data, ex=ttl)
            
        except Exception as e:
            logger.warning(f"Failed to set cache key {key}: {e}")
            raise CacheError(f"Failed to set cache key {key}: {e}")
    
    def delete(self, key: str) -> None:
        """Delete key from Redis cache.
        
        Args:
            key: Cache key to delete
        """
        try:
            self.redis_client.delete(key)
        except Exception as e:
            logger.warning(f"Failed to delete cache key {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self.redis_client.flushdb()
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics.
        
        Returns:
            Cache statistics
        """
        try:
            info = self.redis_client.info()
            return {
                "type": "redis",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
            }
        except Exception as e:
            logger.warning(f"Failed to get Redis stats: {e}")
            return {"type": "redis", "error": str(e)}


class CacheManager:
    """Cache manager with automatic backend selection."""
    
    def __init__(self, cache_backend: Optional[str] = None):
        """Initialize cache manager.
        
        Args:
            cache_backend: Cache backend ('redis', 'memory', or None for auto)
        """
        self.settings = get_settings()
        
        # Determine cache backend
        if cache_backend == "redis" or (cache_backend is None and REDIS_AVAILABLE and self.settings.caching.enabled):
            try:
                self.cache = RedisCache(self.settings.caching.redis_url)
                self.backend = "redis"
            except CacheError:
                logger.warning("Redis cache failed, falling back to in-memory cache")
                self.cache = InMemoryCache(max_size=self.settings.caching.max_size)
                self.backend = "memory"
        else:
            self.cache = InMemoryCache(max_size=self.settings.caching.max_size)
            self.backend = "memory"
        
        logger.info(f"Cache manager initialized with {self.backend} backend")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.settings.caching.enabled:
            return None
        
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        if not self.settings.caching.enabled:
            return
        
        ttl = ttl or self.settings.caching.ttl_seconds
        self.cache.set(key, value, ttl)
    
    def delete(self, key: str) -> None:
        """Delete key from cache.
        
        Args:
            key: Cache key to delete
        """
        self.cache.delete(key)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Cache statistics
        """
        stats = self.cache.get_stats()
        stats["enabled"] = self.settings.caching.enabled
        stats["backend"] = self.backend
        return stats


def cached(ttl: Optional[int] = None, key_prefix: str = "") -> Callable:
    """Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache keys
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache_manager = CacheManager()
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            if args:
                key_parts.extend(str(arg) for arg in args)
            if kwargs:
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {cache_key}")
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    key_parts = []
    
    # Add positional arguments
    for arg in args:
        if hasattr(arg, '__dict__'):
            # For objects, use class name and id
            key_parts.append(f"{arg.__class__.__name__}:{id(arg)}")
        else:
            key_parts.append(str(arg))
    
    # Add keyword arguments
    for key, value in sorted(kwargs.items()):
        if hasattr(value, '__dict__'):
            key_parts.append(f"{key}={value.__class__.__name__}:{id(value)}")
        else:
            key_parts.append(f"{key}={value}")
    
    return ":".join(key_parts)


class CacheWarmer:
    """Cache warming utility."""
    
    def __init__(self, cache_manager: CacheManager):
        """Initialize cache warmer.
        
        Args:
            cache_manager: Cache manager instance
        """
        self.cache_manager = cache_manager
        self.logger = get_logger(__name__)
    
    def warm_cache(self, warm_functions: Dict[str, Callable]) -> Dict[str, bool]:
        """Warm cache with predefined functions.
        
        Args:
            warm_functions: Dictionary of cache keys and functions
            
        Returns:
            Dictionary of success status for each function
        """
        results = {}
        
        for cache_key, func in warm_functions.items():
            try:
                result = func()
                self.cache_manager.set(cache_key, result)
                results[cache_key] = True
                self.logger.info(f"Cache warmed for {cache_key}")
            except Exception as e:
                results[cache_key] = False
                self.logger.error(f"Failed to warm cache for {cache_key}: {e}")
        
        return results
    
    def schedule_warming(self, warm_functions: Dict[str, Callable], interval_seconds: int = 3600) -> None:
        """Schedule periodic cache warming.
        
        Args:
            warm_functions: Dictionary of cache keys and functions
            interval_seconds: Warming interval in seconds
        """
        import threading
        import time
        
        def warm_periodically():
            while True:
                self.warm_cache(warm_functions)
                time.sleep(interval_seconds)
        
        thread = threading.Thread(target=warm_periodically, daemon=True)
        thread.start()
        
        self.logger.info(f"Cache warming scheduled every {interval_seconds} seconds") 