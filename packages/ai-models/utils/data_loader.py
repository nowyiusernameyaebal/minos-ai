"""
Data Loader Module for Minos-AI DeFi Strategy Platform

This module provides utilities for loading, processing, and managing data from various sources
including Solana blockchain data, DEX/CEX market data, and external financial data providers.

The module implements a unified interface for data access with advanced capabilities:
- Multi-source data ingestion (blockchain, exchanges, APIs)
- Incremental data loading with checkpointing
- Parallel data processing for high throughput
- Consistent data format conversion
- Cache management for performance optimization
- Data quality validation and anomaly detection
- Rate limiting and backoff for external API calls

Integration Points:
- Connects to Solana RPC nodes for blockchain data
- Interfaces with DEX/CEX APIs for orderbook and trade data
- Accesses data warehouse for historical aggregated data
- Provides data to training pipelines and model inference

Author: Minos-AI Team
"""

import os
import json
import time
import logging
import asyncio
import hashlib
import warnings
import traceback
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Union, Tuple, Any, Callable, Generator, AsyncGenerator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from solana.rpc.async_api import AsyncClient as SolanaAsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm.auto import tqdm

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TIMEOUT = 60  # seconds
DEFAULT_RETRY_ATTEMPTS = 5
DEFAULT_CACHE_EXPIRY = 3600  # 1 hour
MAX_CONCURRENT_REQUESTS = 50


class DataSource(Enum):
    """Enumeration of supported data sources."""
    SOLANA_BLOCKCHAIN = "solana_blockchain"
    SERUM_DEX = "serum_dex"
    RAYDIUM_AMM = "raydium_amm"
    ORCA_AMM = "orca_amm"
    MANGO_MARKETS = "mango_markets"
    DRIFT_PROTOCOL = "drift_protocol"
    BINANCE = "binance"
    FTX = "ftx"
    COINBASE = "coinbase"
    CRYPTO_COMPARE = "crypto_compare"
    COINGECKO = "coingecko"
    SENTIMENT_API = "sentiment_api"
    FEAR_GREED_INDEX = "fear_greed_index"
    NEWS_API = "news_api"
    TWITTER_API = "twitter_api"
    DATA_WAREHOUSE = "data_warehouse"


class DataCategory(Enum):
    """Enumeration of data categories for organization."""
    PRICE = "price"
    VOLUME = "volume"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    LIQUIDITY = "liquidity"
    ON_CHAIN_METRICS = "on_chain_metrics"
    SENTIMENT = "sentiment"
    MARKET_INDICATORS = "market_indicators"
    FUNDAMENTAL = "fundamental"
    NEWS = "news"
    SOCIAL = "social"
    DERIVATIVES = "derivatives"
    MACRO_ECONOMIC = "macro_economic"


class DataFormat(Enum):
    """Enumeration of supported data formats."""
    DATAFRAME = "dataframe"
    NUMPY = "numpy"
    DICT = "dict"
    LIST = "list"
    TENSOR = "tensor"
    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"


class SamplingFrequency(Enum):
    """Enumeration of supported sampling frequencies."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE = "1min"
    FIVE_MINUTES = "5min"
    FIFTEEN_MINUTES = "15min"
    THIRTY_MINUTES = "30min"
    HOUR = "1h"
    FOUR_HOURS = "4h"
    DAY = "1d"
    WEEK = "1w"
    MONTH = "1mo"


class DataLoadingError(Exception):
    """Base exception for data loading errors."""
    pass


class ApiRateLimitError(DataLoadingError):
    """Exception raised when API rate limits are exceeded."""
    pass


class DataQualityError(DataLoadingError):
    """Exception raised when data fails quality checks."""
    pass


class DataCache:
    """
    Cache manager for dataset storage and retrieval.
    Implements LRU cache with expiration for efficient data access.
    """
    
    def __init__(self, max_size_mb: int = 1024, expiry_seconds: int = DEFAULT_CACHE_EXPIRY):
        """
        Initialize the data cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            expiry_seconds: Default expiration time in seconds
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._max_size_bytes = max_size_mb * 1024 * 1024
        self._current_size_bytes = 0
        self._default_expiry = expiry_seconds
        logger.info(f"Initialized data cache with {max_size_mb}MB capacity")
        
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve item from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached data or None if not found/expired
        """
        if key not in self._cache:
            return None
            
        cache_item = self._cache[key]
        
        # Check for expiration
        if datetime.now().timestamp() > cache_item["expiry"]:
            logger.debug(f"Cache item {key} has expired")
            self._remove(key)
            return None
            
        # Update access time for LRU tracking
        cache_item["last_access"] = datetime.now().timestamp()
        return cache_item["data"]
        
    def set(self, key: str, value: Any, expiry_seconds: Optional[int] = None) -> None:
        """
        Store item in cache with expiration.
        
        Args:
            key: Cache key
            value: Data to cache
            expiry_seconds: Custom expiration time in seconds (default: class default)
        """
        if key in self._cache:
            self._remove(key)
            
        # Calculate data size (approximate)
        data_size = self._estimate_size(value)
        
        # Ensure we have space available
        self._make_space(data_size)
        
        # Set expiry time
        expiry = datetime.now().timestamp() + (expiry_seconds or self._default_expiry)
        
        # Store in cache
        self._cache[key] = {
            "data": value,
            "size": data_size,
            "expiry": expiry,
            "last_access": datetime.now().timestamp()
        }
        
        self._current_size_bytes += data_size
        logger.debug(f"Added {key} to cache ({data_size / 1024 / 1024:.2f}MB)")
        
    def _remove(self, key: str) -> None:
        """
        Remove item from cache.
        
        Args:
            key: Cache key to remove
        """
        if key in self._cache:
            self._current_size_bytes -= self._cache[key]["size"]
            del self._cache[key]
            
    def _make_space(self, required_bytes: int) -> None:
        """
        Ensure cache has space for new data by removing LRU items.
        
        Args:
            required_bytes: Space needed for new item
        """
        # Check if the item is too large for cache
        if required_bytes > self._max_size_bytes:
            logger.warning(f"Item size ({required_bytes / 1024 / 1024:.2f}MB) exceeds cache capacity")
            return
            
        # Remove items until we have enough space
        while self._current_size_bytes + required_bytes > self._max_size_bytes and self._cache:
            # Find least recently used item
            lru_key = min(self._cache.keys(), key=lambda k: self._cache[k]["last_access"])
            logger.debug(f"Removing LRU item {lru_key} from cache")
            self._remove(lru_key)
            
    def _estimate_size(self, data: Any) -> int:
        """
        Estimate memory size of an object in bytes.
        
        Args:
            data: Object to measure
            
        Returns:
            Approximate size in bytes
        """
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
        elif isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, (dict, list)):
            # Rough estimation for complex objects
            return len(json.dumps(data)) * 2
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        else:
            # Fallback for other types
            return 1024  # Assume 1KB for unknown types
            
    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()
        self._current_size_bytes = 0
        logger.info("Cache cleared")
        
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "items_count": len(self._cache),
            "current_size_mb": self._current_size_bytes / 1024 / 1024,
            "max_size_mb": self._max_size_bytes / 1024 / 1024,
            "utilization_percent": (self._current_size_bytes / self._max_size_bytes) * 100 if self._max_size_bytes > 0 else 0
        }


class RateLimiter:
    """
    Rate limiter for API calls to prevent exceeding service limits.
    Implements token bucket algorithm for flexible rate limiting.
    """
    
    def __init__(self, calls_per_second: float, burst_limit: int = 1):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum calls allowed per second
            burst_limit: Maximum burst size allowed
        """
        self.rate = calls_per_second
        self.burst_limit = burst_limit
        self.tokens = burst_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()
        
    async def acquire(self) -> None:
        """
        Acquire permission to make an API call.
        Blocks until a token is available.
        """
        async with self.lock:
            while True:
                # Refill tokens based on time elapsed
                now = time.time()
                elapsed = now - self.last_update
                self.tokens = min(self.burst_limit, self.tokens + elapsed * self.rate)
                self.last_update = now
                
                if self.tokens >= 1.0:
                    # Token available
                    self.tokens -= 1.0
                    return
                    
                # Calculate wait time for next token
                wait_time = (1.0 - self.tokens) / self.rate
                
                # Release lock while waiting
                self.lock.release()
                await asyncio.sleep(wait_time)
                await self.lock.acquire()


class DataValidator:
    """
    Data validation utilities for ensuring data quality.
    Implements checks for common data issues like missing values,
    outliers, and inconsistent timestamps.
    """
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        max_null_percentage: float = 0.1,
        detect_outliers: bool = True,
        timestamp_col: Optional[str] = None,
        raises: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a DataFrame for data quality issues.
        
        Args:
            df: DataFrame to validate
            required_columns: List of columns that must be present
            min_rows: Minimum number of rows required
            max_null_percentage: Maximum percentage of null values allowed
            detect_outliers: Whether to detect and report outliers
            timestamp_col: Column name to verify timestamp continuity
            raises: Whether to raise exceptions for validation failures
            
        Returns:
            Dictionary with validation results
            
        Raises:
            DataQualityError: If validation fails and raises=True
        """
        results = {
            "passed": True,
            "issues": []
        }
        
        # Check DataFrame is not empty
        if len(df) < min_rows:
            msg = f"DataFrame has insufficient rows: {len(df)} < {min_rows}"
            results["passed"] = False
            results["issues"].append(msg)
            if raises:
                raise DataQualityError(msg)
                
        # Check required columns are present
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                msg = f"Missing required columns: {missing_columns}"
                results["passed"] = False
                results["issues"].append(msg)
                if raises:
                    raise DataQualityError(msg)
                    
        # Check for null values
        null_percentages = df.isnull().mean().to_dict()
        high_null_cols = {col: pct for col, pct in null_percentages.items() 
                          if pct > max_null_percentage}
        if high_null_cols:
            msg = f"Columns with high null percentages: {high_null_cols}"
            results["passed"] = False
            results["issues"].append(msg)
            if raises:
                raise DataQualityError(msg)
                
        # Check for outliers in numeric columns
        if detect_outliers:
            outliers = {}
            for col in df.select_dtypes(include=np.number).columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_percentage = outlier_count / len(df)
                if outlier_percentage > 0.01:  # More than 1% outliers
                    outliers[col] = {
                        "percentage": outlier_percentage,
                        "count": outlier_count
                    }
            
            if outliers:
                msg = f"Columns with outliers: {outliers}"
                results["issues"].append(msg)
                # Don't fail validation for outliers, just report them
                
        # Check timestamp continuity if specified
        if timestamp_col and timestamp_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                try:
                    # Try to convert to datetime
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                except:
                    msg = f"Column {timestamp_col} is not a valid timestamp"
                    results["passed"] = False
                    results["issues"].append(msg)
                    if raises:
                        raise DataQualityError(msg)
            
            # Sort by timestamp for continuity check
            sorted_df = df.sort_values(timestamp_col)
            timestamp_diff = sorted_df[timestamp_col].diff()
            
            # Check for gaps larger than expected
            expected_interval = timestamp_diff.median()
            large_gaps = (timestamp_diff > expected_interval * 3) & (timestamp_diff.notna())
            if large_gaps.any():
                gap_info = sorted_df.loc[large_gaps, timestamp_col].to_list()
                msg = f"Timestamp gaps detected at: {gap_info}"
                results["issues"].append(msg)
                # Don't fail validation for timestamp gaps, just report them
                
            # Check for duplicate timestamps
            duplicate_timestamps = sorted_df[timestamp_col].duplicated().sum()
            if duplicate_timestamps > 0:
                msg = f"Found {duplicate_timestamps} duplicate timestamps"
                results["issues"].append(msg)
                # Don't fail validation for duplicate timestamps, just report them
        
        return results


class DataLoader:
    """
    Main data loading interface for accessing financial data from
    various sources with unified API.
    
    This class provides:
    - Consistent interfaces for different data sources
    - Connection pooling and request optimization
    - Data caching and incremental loading
    - Retry logic and error handling
    - Data format conversion
    """
    
    def __init__(
        self,
        cache_size_mb: int = 2048,
        cache_expiry: int = DEFAULT_CACHE_EXPIRY,
        max_workers: int = 10,
        api_keys: Optional[Dict[str, str]] = None,
        solana_rpc_url: Optional[str] = None,
        data_warehouse_url: Optional[str] = None
    ):
        """
        Initialize data loader with configuration.
        
        Args:
            cache_size_mb: Cache size in megabytes
            cache_expiry: Cache expiry time in seconds
            max_workers: Maximum concurrent workers for parallel loading
            api_keys: Dictionary of API keys for various services
            solana_rpc_url: Custom Solana RPC URL (None for default)
            data_warehouse_url: URL for data warehouse API
        """
        # Initialize configuration
        self.cache = DataCache(max_size_mb=cache_size_mb, expiry_seconds=cache_expiry)
        self.max_workers = max_workers
        self.api_keys = api_keys or {}
        
        # Initialize rate limiters for various APIs
        self.rate_limiters = {
            DataSource.SOLANA_BLOCKCHAIN: RateLimiter(5.0, 10),     # 5 req/s, burst of 10
            DataSource.SERUM_DEX: RateLimiter(2.0, 5),              # 2 req/s, burst of 5
            DataSource.RAYDIUM_AMM: RateLimiter(2.0, 5),            # 2 req/s, burst of 5
            DataSource.BINANCE: RateLimiter(10.0, 20),              # 10 req/s, burst of 20
            DataSource.COINBASE: RateLimiter(3.0, 6),               # 3 req/s, burst of 6
            DataSource.CRYPTO_COMPARE: RateLimiter(5.0, 5),         # 5 req/s, no burst
            DataSource.COINGECKO: RateLimiter(1.0, 1),              # 1 req/s, no burst
            DataSource.FEAR_GREED_INDEX: RateLimiter(0.2, 1),       # 1 req/5s
            DataSource.SENTIMENT_API: RateLimiter(2.0, 2),          # 2 req/s
            DataSource.DATA_WAREHOUSE: RateLimiter(20.0, 50),       # 20 req/s, burst of 50
        }
        
        # Configure Solana client
        self.solana_rpc_url = solana_rpc_url or "https://api.mainnet-beta.solana.com"
        self.solana_client = None  # Lazy initialization
        
        # Configure data warehouse client
        self.data_warehouse_url = data_warehouse_url
        
        # HTTP session for API calls (initialized on first use)
        self._http_session = None
        
        logger.info(f"Initialized DataLoader with {cache_size_mb}MB cache")
    
    async def _get_solana_client(self) -> SolanaAsyncClient:
        """
        Get or create Solana client instance.
        
        Returns:
            Initialized Solana async client
        """
        if self.solana_client is None:
            self.solana_client = SolanaAsyncClient(self.solana_rpc_url)
        return self.solana_client
    
    async def _get_http_session(self) -> ClientSession:
        """
        Get or create HTTP session for API calls.
        
        Returns:
            Initialized aiohttp ClientSession
        """
        if self._http_session is None or self._http_session.closed:
            self._http_session = ClientSession(
                timeout=ClientTimeout(total=DEFAULT_TIMEOUT),
                connector=TCPConnector(limit=MAX_CONCURRENT_REQUESTS)
            )
        return self._http_session
    
    def _get_cache_key(self, 
                      source: DataSource,
                      asset: str,
                      start_time: Union[datetime, str],
                      end_time: Union[datetime, str],
                      interval: SamplingFrequency,
                      params: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a unique cache key for data query.
        
        Args:
            source: Data source
            asset: Asset identifier
            start_time: Start time for data range
            end_time: End time for data range
            interval: Sampling frequency
            params: Additional query parameters
            
        Returns:
            Unique cache key string
        """
        # Normalize datetime objects to ISO strings
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
            
        # Create unique key from parameters
        key_parts = [
            source.value,
            asset,
            start_time,
            end_time,
            interval.value
        ]
        
        # Add any additional parameters
        if params:
            for k, v in sorted(params.items()):
                key_parts.append(f"{k}={v}")
                
        # Join and hash the key
        key_string = "|".join(str(part) for part in key_parts)
        return f"data_{hashlib.md5(key_string.encode()).hexdigest()}"
    
    @retry(
        stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((
            ConnectionError, TimeoutError, ApiRateLimitError
        )),
        reraise=True
    )
    async def _fetch_api_data(self, 
                             url: str, 
                             source: DataSource,
                             params: Optional[Dict[str, Any]] = None,
                             headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Fetch data from REST API with rate limiting and retry logic.
        
        Args:
            url: API endpoint URL
            source: Data source for rate limiting
            params: Query parameters
            headers: HTTP headers
            
        Returns:
            JSON response as dictionary
            
        Raises:
            ApiRateLimitError: If rate limit is exceeded
            DataLoadingError: For other API errors
        """
        # Apply rate limiting if applicable
        if source in self.rate_limiters:
            await self.rate_limiters[source].acquire()
            
        # Add API key to headers if available
        if headers is None:
            headers = {}
            
        source_name = source.name.lower()
        if f"{source_name}_api_key" in self.api_keys:
            headers["Authorization"] = f"Bearer {self.api_keys[f'{source_name}_api_key']}"
            
        # Get HTTP session
        session = await self._get_http_session()
        
        try:
            # Make the API request
            async with session.get(url, params=params, headers=headers) as response:
                # Check for rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", "60"))
                    logger.warning(f"Rate limit exceeded for {source.value}. Retry after {retry_after}s")
                    raise ApiRateLimitError(f"Rate limit exceeded for {source.value}")
                    
                # Check for other errors
                if response.status >= 400:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    raise DataLoadingError(f"API error {response.status}: {error_text}")
                    
                # Parse JSON response
                return await response.json()
                
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Connection error for {url}: {str(e)}")
            raise
    
    async def get_price_data(self,
                           asset: str,
                           start_time: Union[datetime, str],
                           end_time: Union[datetime, str],
                           interval: SamplingFrequency = SamplingFrequency.HOUR,
                           source: DataSource = DataSource.DATA_WAREHOUSE,
                           include_indicators: bool = False,
                           output_format: DataFormat = DataFormat.DATAFRAME,
                           use_cache: bool = True) -> Union[pd.DataFrame, Dict[str, Any]]:
        """
        Get historical price data for an asset.
        
        Args:
            asset: Asset identifier (e.g., "SOL/USDC")
            start_time: Start time for data range
            end_time: End time for data range
            interval: Data sampling frequency
            source: Data source to use
            include_indicators: Whether to include technical indicators
            output_format: Format for returned data
            use_cache: Whether to use cached data
            
        Returns:
            Price data in the requested format
            
        Raises:
            DataLoadingError: If data loading fails
        """
        cache_key = None
        if use_cache:
            # Generate cache key
            cache_key = self._get_cache_key(
                source, asset, start_time, end_time, interval,
                {"indicators": include_indicators}
            )
            
            # Check cache
            cached_data = self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Using cached price data for {asset}")
                
                # Convert to requested format
                if output_format == DataFormat.DATAFRAME and not isinstance(cached_data, pd.DataFrame):
                    return pd.DataFrame(cached_data)
                elif output_format == DataFormat.DICT and isinstance(cached_data, pd.DataFrame):
                    return cached_data.to_dict(orient="records")
                else:
                    return cached_data
                    
        try:
            # Fetch from data warehouse if available (preferred source)
            if source == DataSource.DATA_WAREHOUSE and self.data_warehouse_url:
                data = await self._fetch_price_from_warehouse(
                    asset, start_time, end_time, interval, include_indicators
                )
            # Fallback to direct exchange API calls
            elif source in [DataSource.BINANCE, DataSource.COINBASE, DataSource.FTX]:
                data = await self._fetch_price_from_exchange(
                    asset, start_time, end_time, interval, source
                )
            # Get from crypto data aggregators
            elif source in [DataSource.CRYPTO_COMPARE, DataSource.COINGECKO]:
                data = await self._fetch_price_from_aggregator(
                    asset, start_time, end_time, interval, source
                )
            # Get on-chain DEX data
            elif source in [DataSource.SERUM_DEX, DataSource.RAYDIUM_AMM, DataSource.ORCA_AMM]:
                data = await self._fetch_price_from_dex(
                    asset, start_time, end_time, interval, source
                )
            else:
                raise DataLoadingError(f"Unsupported data source: {source.value}")
                
            # Add technical indicators if requested
            if include_indicators and isinstance(data, pd.DataFrame):
                data = self._add_technical_indicators(data)
                
            # Validate data quality
            validation_result = DataValidator.validate_dataframe(
                data,
                required_columns=["timestamp", "open", "high", "low", "close"],
                min_rows=1,
                timestamp_col="timestamp",
                raises=False
            )
            
            if not validation_result["passed"]:
                logger.warning(f"Data quality issues: {validation_result['issues']}")
                
            # Cache the result (always cache the DataFrame format)
            if use_cache and cache_key:
                if isinstance(data, pd.DataFrame):
                    self.cache.set(cache_key, data)
                else:
                    self.cache.set(cache_key, pd.DataFrame(data))
                    
            # Convert to requested output format
            if output_format == DataFormat.DICT and isinstance(data, pd.DataFrame):
                return data.to_dict(orient="records")
            elif output_format == DataFormat.DATAFRAME and not isinstance(data, pd.DataFrame):
                return pd.DataFrame(data)
            else:
                return data
                
        except Exception as e:
            logger.error(f"Error fetching price data for {asset}: {str(e)}")
            logger.error(traceback.format_exc())
            raise DataLoadingError(f"Failed to load price data: {str(e)}")
    
    async def _fetch_price_from_warehouse(self,
                                        asset: str,
                                        start_time: Union[datetime, str],
                                        end_time: Union[datetime, str],
                                        interval: SamplingFrequency,
                                        include_indicators: bool) -> pd.DataFrame:
        """
        Fetch price data from internal data warehouse.
        
        Args:
            asset: Asset identifier
            start_time: Start time
            end_time: End time
            interval: Sampling frequency
            include_indicators: Whether to include technical indicators
            
        Returns:
            DataFrame with price data
        """
        if not self.data_warehouse_url:
            raise DataLoadingError("Data warehouse URL not configured")
            
        # Convert datetime objects to ISO format strings
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
            
        # Prepare API request
        url = f"{self.data_warehouse_url}/api/v1/price"
        params = {
            "asset": asset,
            "start_time": start_time,
            "end_time": end_time,
            "interval": interval.value,
            "indicators": "true" if include_indicators else "false"
        }
        
        # Make request
        response_data = await self._fetch_api_data(
            url, DataSource.DATA_WAREHOUSE, params
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(response_data["data"])
        
        # Ensure timestamp column is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        return df
    
    async def _fetch_price_from_exchange(self,
                                       asset: str,
                                       start_time: Union[datetime, str],
                                       end_time: Union[datetime, str],
                                       interval: SamplingFrequency,
                                       source: DataSource) -> pd.DataFrame:
        """
        Fetch price data directly from cryptocurrency exchanges.
        
        Args:
            asset: Asset identifier
            start_time: Start time
            end_time: End time
            interval: Sampling frequency
            source: Exchange data source
            
        Returns:
            DataFrame with price data
        """
        # Normalize asset format for specific exchanges
        normalized_asset = asset.replace("/", "")  # Remove slash for most exchanges
        
        # Convert datetime objects to timestamps or ISO format strings
        start_ts = int(start_time.timestamp() * 1000) if isinstance(start_time, datetime) else None
        end_ts = int(end_time.timestamp() * 1000) if isinstance(end_time, datetime) else None
        
        if start_ts is None and isinstance(start_time, str):
            start_ts = int(pd.to_datetime(start_time).timestamp() * 1000)
            
        if end_ts is None and isinstance(end_time, str):
            end_ts = int(pd.to_datetime(end_time).timestamp() * 1000)
        
        # Map interval to exchange-specific format
        interval_map = {
            SamplingFrequency.MINUTE: "1m",
            SamplingFrequency.FIVE_MINUTES: "5m",
            SamplingFrequency.FIFTEEN_MINUTES: "15m",
            SamplingFrequency.THIRTY_MINUTES: "30m",
            SamplingFrequency.HOUR: "1h",
            SamplingFrequency.FOUR_HOURS: "4h",
            SamplingFrequency.DAY: "1d",
            SamplingFrequency.WEEK: "1w",
        }
        exchange_interval = interval_map.get(interval, "1h")
        
        # Exchange-specific API handling
        if source == DataSource.BINANCE:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": normalized_asset,
                "interval": exchange_interval,
                "startTime": start_ts,
                "endTime": end_ts,
                "limit": 1000  # Maximum allowed
            }
            
            # Handle pagination for large date ranges
            all_candles = []
            while True:
                response_data = await self._fetch_api_data(url, source, params)
                
                if not response_data:
                    break
                    
                all_candles.extend(response_data)
                
                if len(response_data) < 1000:
                    break
                    
                # Update start time for next batch
                params["startTime"] = int(response_data[-1][0]) + 1
                
                if params["startTime"] >= end_ts:
                    break
            
            # Convert to DataFrame
            columns = ["timestamp", "open", "high", "low", "close", "volume", 
                      "close_time", "quote_asset_volume", "number_of_trades",
                      "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"]
            df = pd.DataFrame(all_candles, columns=columns)
            
            # Convert types
            numeric_columns = ["open", "high", "low", "close", "volume", 
                              "quote_asset_volume", "taker_buy_base_asset_volume", 
                              "taker_buy_quote_asset_volume"]
            df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            
            return df
            
        elif source == DataSource.COINBASE:
            # Extract base and quote from asset
            asset_parts = asset.split("/")
            if len(asset_parts) != 2:
                raise DataLoadingError(f"Invalid asset format for Coinbase: {asset}")
                
            base, quote = asset_parts
            product_id = f"{base}-{quote}"
            
            # Convert interval to Coinbase format
            coinbase_interval_map = {
                SamplingFrequency.MINUTE: 60,
                SamplingFrequency.FIVE_MINUTES: 300,
                SamplingFrequency.FIFTEEN_MINUTES: 900,
                SamplingFrequency.THIRTY_MINUTES: 1800,
                SamplingFrequency.HOUR: 3600,
                SamplingFrequency.SIX_HOURS: 21600,
                SamplingFrequency.DAY: 86400,
            }
            granularity = coinbase_interval_map.get(interval, 3600)
            
            url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
            params = {
                "granularity": granularity,
                "start": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
                "end": end_time.isoformat() if isinstance(end_time, datetime) else end_time
            }
            
            response_data = await self._fetch_api_data(url, source, params)
            
            # Convert to DataFrame
            columns = ["timestamp", "low", "high", "open", "close", "volume"]
            df = pd.DataFrame(response_data, columns=columns)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            
            # Sort by timestamp (Coinbase returns data in descending order)
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            return df
            
        elif source == DataSource.FTX:
            # Note: As of knowledge cutoff, FTX was no longer operating
            # This implementation is kept for historical reference
            url = "https://ftx.com/api/markets/{market_name}/candles"
            url = url.format(market_name=asset.replace("/", ""))
            
            params = {
                "resolution": int(exchange_interval[:-1]) * 60,  # Convert to seconds
                "start_time": int(start_ts / 1000),
                "end_time": int(end_ts / 1000)
            }
            
            response_data = await self._fetch_api_data(url, source, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(response_data["result"])
            df["timestamp"] = pd.to_datetime(df["startTime"])
            
            return df
            
        else:
            raise DataLoadingError(f"Unsupported exchange: {source.value}")
    
    async def _fetch_price_from_aggregator(self,
                                         asset: str,
                                         start_time: Union[datetime, str],
                                         end_time: Union[datetime, str],
                                         interval: SamplingFrequency,
                                         source: DataSource) -> pd.DataFrame:
        """
        Fetch price data from cryptocurrency data aggregators.
        
        Args:
            asset: Asset identifier
            start_time: Start time
            end_time: End time
            interval: Sampling frequency
            source: Aggregator data source
            
        Returns:
            DataFrame with price data
        """
        # Convert datetime objects to timestamps
        start_ts = int(start_time.timestamp()) if isinstance(start_time, datetime) else None
        end_ts = int(end_time.timestamp()) if isinstance(end_time, datetime) else None
        
        if start_ts is None and isinstance(start_time, str):
            start_ts = int(pd.to_datetime(start_time).timestamp())
            
        if end_ts is None and isinstance(end_time, str):
            end_ts = int(pd.to_datetime(end_time).timestamp())
            
        # Handle different aggregator sources
        if source == DataSource.CRYPTO_COMPARE:
            # Parse asset for CryptoCompare format
            asset_parts = asset.split("/")
            if len(asset_parts) != 2:
                raise DataLoadingError(f"Invalid asset format for CryptoCompare: {asset}")
                
            from_symbol, to_symbol = asset_parts
            
            # Map interval to CryptoCompare format
            interval_map = {
                SamplingFrequency.MINUTE: "histominute",
                SamplingFrequency.HOUR: "histohour",
                SamplingFrequency.DAY: "histoday",
            }
            
            # Default to closest available if exact interval not available
            if interval == SamplingFrequency.FIVE_MINUTES or interval == SamplingFrequency.FIFTEEN_MINUTES or interval == SamplingFrequency.THIRTY_MINUTES:
                endpoint = "histominute"
                # Determine aggregation factor based on interval
                if interval == SamplingFrequency.FIVE_MINUTES:
                    aggregation = 5
                elif interval == SamplingFrequency.FIFTEEN_MINUTES:
                    aggregation = 15
                else:  # 30 minutes
                    aggregation = 30
            elif interval == SamplingFrequency.FOUR_HOURS:
                endpoint = "histohour"
                aggregation = 4
            elif interval == SamplingFrequency.WEEK:
                endpoint = "histoday"
                aggregation = 7
            else:
                endpoint = interval_map.get(interval, "histohour")
                aggregation = 1
                
            url = f"https://min-api.cryptocompare.com/data/v2/{endpoint}"
            params = {
                "fsym": from_symbol,
                "tsym": to_symbol,
                "limit": 2000,  # Maximum allowed in one request
                "toTs": end_ts,
                "aggregate": aggregation,
                "e": "CCCAGG"  # Use all exchanges
            }
            
            if "cryptocompare_api_key" in self.api_keys:
                params["api_key"] = self.api_keys["cryptocompare_api_key"]
                
            # Handle pagination for large date ranges
            all_data = []
            current_end_ts = end_ts
            
            while current_end_ts >= start_ts and len(all_data) < 100000:  # Safety limit
                params["toTs"] = current_end_ts
                response_data = await self._fetch_api_data(url, source, params)
                
                if "Data" not in response_data or "Data" not in response_data["Data"]:
                    break
                    
                data_points = response_data["Data"]["Data"]
                
                if not data_points:
                    break
                    
                all_data = data_points + all_data
                
                # Update end timestamp for next batch
                current_end_ts = data_points[0]["time"] - 1
                
                # Check if we've reached the start time
                if data_points[0]["time"] <= start_ts:
                    break
                    
            # Filter by actual start time
            all_data = [d for d in all_data if d["time"] >= start_ts]
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            df["timestamp"] = pd.to_datetime(df["time"], unit="s")
            
            # Rename columns to standard format
            df = df.rename(columns={
                "volumefrom": "volume",
                "volumeto": "quote_volume"
            })
            
            # Sort by timestamp
            df = df.sort_values("timestamp").reset_index(drop=True)
            
            return df
            
        elif source == DataSource.COINGECKO:
            # Extract necessary parameters
            asset_parts = asset.split("/")
            if len(asset_parts) != 2:
                raise DataLoadingError(f"Invalid asset format for CoinGecko: {asset}")
                
            base_currency, quote_currency = asset_parts
            
            # CoinGecko requires cryptocurrency IDs
            # This mapping should be expanded or retrieved dynamically
            currency_map = {
                "BTC": "bitcoin",
                "ETH": "ethereum",
                "SOL": "solana",
                "USDC": "usd-coin",
                "USDT": "tether",
                "DOT": "polkadot",
                "ADA": "cardano",
                "AVAX": "avalanche-2",
                "MATIC": "polygon",
                "BNB": "binancecoin",
                "XRP": "ripple",
                "DOGE": "dogecoin",
                "SHIB": "shiba-inu",
                "LTC": "litecoin",
                "LINK": "chainlink",
                "UNI": "uniswap",
                "DAI": "dai",
                "ATOM": "cosmos",
                "TRX": "tron",
                "FTM": "fantom",
                "ALGO": "algorand",
                "NEAR": "near",
                "FIL": "filecoin",
                "ICP": "internet-computer",
                "APE": "apecoin",
                "AXS": "axie-infinity",
                "SAND": "the-sandbox",
                "MANA": "decentraland",
                "AAVE": "aave",
                "CRO": "crypto-com-chain",
                "EGLD": "elrond-erd-2",
                "XTZ": "tezos",
                "FLOW": "flow",
                "GRT": "the-graph",
                "XMR": "monero",
            }
            
            if base_currency.upper() not in currency_map:
                raise DataLoadingError(f"Unsupported base currency for CoinGecko: {base_currency}")
                
            coin_id = currency_map[base_currency.upper()]
            vs_currency = quote_currency.lower()
            
            # Map interval to CoinGecko format (CoinGecko only supports limited intervals)
            days_map = {
                SamplingFrequency.MINUTE: 1,
                SamplingFrequency.FIVE_MINUTES: 1,
                SamplingFrequency.FIFTEEN_MINUTES: 1,
                SamplingFrequency.THIRTY_MINUTES: 1,
                SamplingFrequency.HOUR: 7,
                SamplingFrequency.FOUR_HOURS: 30,
                SamplingFrequency.DAY: 90,
                SamplingFrequency.WEEK: 365,
                SamplingFrequency.MONTH: 365,
            }
            
            interval_map = {
                SamplingFrequency.MINUTE: "minutely",
                SamplingFrequency.FIVE_MINUTES: "minutely",
                SamplingFrequency.FIFTEEN_MINUTES: "minutely",
                SamplingFrequency.THIRTY_MINUTES: "minutely",
                SamplingFrequency.HOUR: "hourly",
                SamplingFrequency.FOUR_HOURS: "hourly",
                SamplingFrequency.DAY: "daily",
                SamplingFrequency.WEEK: "daily",
                SamplingFrequency.MONTH: "daily",
            }
            
            days = days_map.get(interval, 30)
            interval_param = interval_map.get(interval, "hourly")
            
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": vs_currency,
                "days": days,
                "interval": interval_param
            }
            
            response_data = await self._fetch_api_data(url, source, params)
            
            # Extract price data
            price_data = response_data.get("prices", [])
            volume_data = response_data.get("total_volumes", [])
            
            # Create DataFrame
            df_price = pd.DataFrame(price_data, columns=["timestamp", "price"])
            df_price["timestamp"] = pd.to_datetime(df_price["timestamp"], unit="ms")
            
            df_volume = pd.DataFrame(volume_data, columns=["timestamp", "volume"])
            df_volume["timestamp"] = pd.to_datetime(df_volume["timestamp"], unit="ms")
            
            # Merge price and volume data
            df = pd.merge(df_price, df_volume, on="timestamp", how="outer")
            
            # Filter by date range
            start_dt = pd.to_datetime(start_time) if isinstance(start_time, str) else start_time
            end_dt = pd.to_datetime(end_time) if isinstance(end_time, str) else end_time
            
            df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
            
            # CoinGecko only provides close prices, so set them for OHLC
            df["close"] = df["price"]
            df["open"] = df["price"]
            df["high"] = df["price"]
            df["low"] = df["price"]
            
            # If we need exact interval resampling
            if interval in [SamplingFrequency.FIVE_MINUTES, SamplingFrequency.FIFTEEN_MINUTES, 
                           SamplingFrequency.THIRTY_MINUTES, SamplingFrequency.FOUR_HOURS]:
                # Resample to desired frequency
                interval_str = interval.value.replace("min", "T")  # Convert to pandas format
                df = df.set_index("timestamp")
                
                # Resample OHLCV
                df_resampled = pd.DataFrame()
                df_resampled["open"] = df["open"].resample(interval_str).first()
                df_resampled["high"] = df["high"].resample(interval_str).max()
                df_resampled["low"] = df["low"].resample(interval_str).min()
                df_resampled["close"] = df["close"].resample(interval_str).last()
                df_resampled["volume"] = df["volume"].resample(interval_str).sum()
                
                # Reset index
                df_resampled = df_resampled.reset_index()
                df = df_resampled
            
            return df
            
        else:
            raise DataLoadingError(f"Unsupported aggregator: {source.value}")
    
    async def _fetch_price_from_dex(self,
                                  asset: str,
                                  start_time: Union[datetime, str],
                                  end_time: Union[datetime, str],
                                  interval: SamplingFrequency,
                                  source: DataSource) -> pd.DataFrame:
        """
        Fetch price data from Solana DEXes.
        
        Args:
            asset: Asset identifier
            start_time: Start time
            end_time: End time
            interval: Sampling frequency
            source: DEX data source
            
        Returns:
            DataFrame with price data
        """
        # Parse trading pair
        asset_parts = asset.split("/")
        if len(asset_parts) != 2:
            raise DataLoadingError(f"Invalid asset format for DEX: {asset}")
            
        base_token, quote_token = asset_parts
        
        # Get Solana client
        solana_client = await self._get_solana_client()
        
        if source == DataSource.SERUM_DEX:
            # For Serum, we need the market address
            # This should be expanded with a proper market registry
            market_registry = {
                "SOL/USDC": "9wFFyRfZBsuAha4YcuxcXLKwMxJR43S7fPfQLusDBzvT",
                "BTC/USDC": "A8YFbxQYFVqKZaoYJLLUVcQiWP7G2MeEgW5wsAQgMvFw",
                "ETH/USDC": "4tSvZvnbyzHXLMTiFonMyxZoHmFqau1XArcRCVHLZ5gX",
                # Add more market addresses as needed
            }
            
            if asset not in market_registry:
                raise DataLoadingError(f"Unknown Serum market for {asset}")
                
            market_address = market_registry[asset]
            
            # This implementation would be complex, requiring interaction with the Serum DEX program
            # For brevity, we'll use a simplified approach via our data warehouse
            # In a real implementation, this would involve parsing on-chain data for trades, etc.
            
            url = f"{self.data_warehouse_url}/api/v1/dex/serum/market/{market_address}/candles"
            params = {
                "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
                "end_time": end_time.isoformat() if isinstance(end_time, datetime) else end_time,
                "interval": interval.value
            }
            
            response_data = await self._fetch_api_data(url, source, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(response_data["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            return df
            
        elif source == DataSource.RAYDIUM_AMM:
            # Similar to Serum, we'll use data warehouse for Raydium data
            # Real implementation would involve interaction with Raydium contracts
            
            pool_registry = {
                "SOL/USDC": "58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2",
                "BTC/USDC": "6kbC5epG18DF2DwPEW34tBy5pGFS7pEGALR3v5MGxgc5",
                "ETH/USDC": "5j6BozKdXBpj4R7L9KfGsGsDgMZGHRPGvLyS83v5d6j",
                # Add more pool addresses as needed
            }
            
            if asset not in pool_registry:
                raise DataLoadingError(f"Unknown Raydium pool for {asset}")
                
            pool_address = pool_registry[asset]
            
            url = f"{self.data_warehouse_url}/api/v1/dex/raydium/pool/{pool_address}/candles"
            params = {
                "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
                "end_time": end_time.isoformat() if isinstance(end_time, datetime) else end_time,
                "interval": interval.value
            }
            
            response_data = await self._fetch_api_data(url, source, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(response_data["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            return df
            
        elif source == DataSource.ORCA_AMM:
            # Similar approach for Orca
            
            pool_registry = {
                "SOL/USDC": "EGZ7tiLeH62TPV1gL8WwbXGzEPa9zmcpVnnkPKKnrE2U",
                "BTC/USDC": "D9ivbEKAS9K1QCMz6PN1UVbwUPLvYgXBu3YQ3nsworSM",
                "ETH/USDC": "FgZut2qVQEyPBibaTJbbX2PxaMZvT1vjDebiVaDp5BWP",
                # Add more pool addresses as needed
            }
            
            if asset not in pool_registry:
                raise DataLoadingError(f"Unknown Orca pool for {asset}")
                
            pool_address = pool_registry[asset]
            
            url = f"{self.data_warehouse_url}/api/v1/dex/orca/pool/{pool_address}/candles"
            params = {
                "start_time": start_time.isoformat() if isinstance(start_time, datetime) else start_time,
                "end_time": end_time.isoformat() if isinstance(end_time, datetime) else end_time,
                "interval": interval.value
            }
            
            response_data = await self._fetch_api_data(url, source, params)
            
            # Convert to DataFrame
            df = pd.DataFrame(response_data["data"])
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            return df
            
        else:
            raise DataLoadingError(f"Unsupported DEX: {source.value}")
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to price data.
        
        Args:
            df: DataFrame with OHLCV price data
            
        Returns:
            DataFrame with added technical indicators
        """
        try:
            # Ensure required columns exist
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required_columns):
                logger.warning("Cannot add indicators, missing required columns")
                return df
                
            # Import here to avoid dependency if not needed
            import ta
            
            # Make a copy to avoid modifying the original
            result_df = df.copy()
            
            # Add common indicators
            
            # Moving Averages
            result_df["sma_7"] = ta.trend.sma_indicator(result_df["close"], window=7)
            result_df["sma_25"] = ta.trend.sma_indicator(result_df["close"], window=25)
            result_df["sma_99"] = ta.trend.sma_indicator(result_df["close"], window=99)
            result_df["ema_12"] = ta.trend.ema_indicator(result_df["close"], window=12)
            result_df["ema_26"] = ta.trend.ema_indicator(result_df["close"], window=26)
            
            # MACD
            macd = ta.trend.MACD(result_df["close"])
            result_df["macd"] = macd.macd()
            result_df["macd_signal"] = macd.macd_signal()
            result_df["macd_diff"] = macd.macd_diff()
            
            # RSI
            result_df["rsi_14"] = ta.momentum.rsi(result_df["close"], window=14)
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(result_df["close"])
            result_df["bb_upper"] = bollinger.bollinger_hband()
            result_df["bb_middle"] = bollinger.bollinger_mavg()
            result_df["bb_lower"] = bollinger.bollinger_lband()
            result_df["bb_width"] = (result_df["bb_upper"] - result_df["bb_lower"]) / result_df["bb_middle"]
            
            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(result_df["high"], result_df["low"], result_df["close"])
            result_df["stoch_k"] = stoch.stoch()
            result_df["stoch_d"] = stoch.stoch_signal()
            
            # ATR (Average True Range)
            result_df["atr"] = ta.volatility.average_true_range(result_df["high"], result_df["low"], result_df["close"])
            
            # OBV (On-Balance Volume)
            result_df["obv"] = ta.volume.on_balance_volume(result_df["close"], result_df["volume"])
            
            # Calculate returns
            result_df["returns"] = result_df["close"].pct_change()
            result_df["log_returns"] = np.log(result_df["close"]).diff()
            
            # Volatility (rolling standard deviation of returns)
            result_df["volatility_7"] = result_df["log_returns"].rolling(7).std()
            result_df["volatility_21"] = result_df["log_returns"].rolling(21).std()
            
            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(result_df["high"], result_df["low"])
            result_df["ichimoku_a"] = ichimoku.ichimoku_a()
            result_df["ichimoku_b"] = ichimoku.ichimoku_b()
            result_df["ichimoku_conversion"] = ichimoku.ichimoku_conversion_line()
            result_df["ichimoku_base"] = ichimoku.ichimoku_base_line()
            
            # VWAP (Volume Weighted Average Price) - for intraday data
            if interval.value in ["1min", "5min", "15min", "30min", "1h", "4h"]:
                result_df["vwap"] = (result_df["volume"] * (result_df["high"] + result_df["low"] + result_df["close"]) / 3).cumsum() / result_df["volume"].cumsum()
            
            return result_df
            
        except Exception as e:
            logger.warning(f"Error adding technical indicators: {str(e)}")
            return df
    
    async def get_market_depth(self,
                             asset: str,
                             source: DataSource,
                             limit: int = 100,
                             use_cache: bool = True) -> Dict[str, Any]:
        """
        Get current market depth (orderbook) for an asset.
        
        Args:
            asset: Asset identifier
            source: Data source
            limit: Maximum number of price levels to retrieve
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary with bids and asks
        """
        # Simplified implementation to focus on key functionality
        # Real implementation would handle various sources and deeper orderbook parsing
        
        if source == DataSource.SERUM_DEX:
            # For Serum DEX, we would need to parse the on-chain orderbook
            # This is complex and would involve Solana RPC calls
            # For brevity, we'll use our data warehouse as a proxy
            
            url = f"{self.data_warehouse_url}/api/v1/dex/serum/orderbook"
            params = {
                "asset": asset,
                "limit": limit
            }
            
            response_data = await self._fetch_api_data(url, source, params)
            return response_data
            
        elif source in [DataSource.BINANCE, DataSource.COINBASE]:
            # Exchange-specific implementation
            normalized_asset = asset.replace("/", "")
            
            if source == DataSource.BINANCE:
                url = "https://api.binance.com/api/v3/depth"
                params = {
                    "symbol": normalized_asset,
                    "limit": limit
                }
                
                response_data = await self._fetch_api_data(url, source, params)
                
                return {
                    "bids": [[float(price), float(amount)] for price, amount in response_data["bids"]],
                    "asks": [[float(price), float(amount)] for price, amount in response_data["asks"]],
                    "timestamp": datetime.now().isoformat()
                }
                
            elif source == DataSource.COINBASE:
                asset_parts = asset.split("/")
                if len(asset_parts) != 2:
                    raise DataLoadingError(f"Invalid asset format for Coinbase: {asset}")
                    
                base, quote = asset_parts
                product_id = f"{base}-{quote}"
                
                url = f"https://api.exchange.coinbase.com/products/{product_id}/book"
                params = {
                    "level": 2
                }
                
                response_data = await self._fetch_api_data(url, source, params)
                
                return {
                    "bids": [[float(price), float(amount)] for price, amount in response_data["bids"]],
                    "asks": [[float(price), float(amount)] for price, amount in response_data["asks"]],
                    "timestamp": response_data["time"]
                }
            
            else:
                raise DataLoadingError(f"Unsupported source for order book: {source.value}")
        
    async def get_on_chain_metrics(self,
                                 asset: str,
                                 start_time: Union[datetime, str],
                                 end_time: Union[datetime, str],
                                 interval: SamplingFrequency = SamplingFrequency.DAY,
                                 metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get on-chain metrics for a Solana token.
        
        Args:
            asset: Asset identifier (token address or symbol)
            start_time: Start time for data range
            end_time: End time for data range
            interval: Data sampling frequency
            metrics: List of metrics to retrieve (None for all available)
            
        Returns:
            DataFrame with on-chain metrics
        """
        # Map common token symbols to addresses
        token_map = {
            "SOL": "So11111111111111111111111111111111111111111",
            "BTC": "9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E",  # Wrapped BTC
            "ETH": "2FPyTwcZLUg1MDrwsyoP4D6s1tM7hAkHYRjkNb5w6Pxk",  # Wrapped ETH
            "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
            "SRM": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt",
            # Add more tokens as needed
        }
        
        # Normalize asset identifier
        token_address = asset
        if asset in token_map:
            token_address = token_map[asset]
            
        # Available metrics
        all_metrics = [
            "active_addresses",
            "new_addresses",
            "transaction_count",
            "transaction_volume",
            "average_transaction_size",
            "large_transactions",
            "token_velocity",
            "token_age_consumed",
            "staking_rate",
            "concentration_metrics"
        ]
        
        # Use all metrics if none specified
        if metrics is None:
            metrics = all_metrics
            
        # Validate metrics
        invalid_metrics = [m for m in metrics if m not in all_metrics]
        if invalid_metrics:
            logger.warning(f"Invalid metrics requested: {invalid_metrics}")
            
        valid_metrics = [m for m in metrics if m in all_metrics]
        
        # Convert datetime objects to ISO format strings
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
            
        # Fetch data from data warehouse
        url = f"{self.data_warehouse_url}/api/v1/onchain/metrics"
        params = {
            "token": token_address,
            "start_time": start_time,
            "end_time": end_time,
            "interval": interval.value,
            "metrics": ",".join(valid_metrics)
        }
        
        response_data = await self._fetch_api_data(
            url, DataSource.DATA_WAREHOUSE, params
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(response_data["data"])
        
        # Ensure timestamp column is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        return df
    
    async def get_sentiment_data(self,
                               asset: str,
                               start_time: Union[datetime, str],
                               end_time: Union[datetime, str],
                               interval: SamplingFrequency = SamplingFrequency.DAY,
                               sources: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get market sentiment data for an asset.
        
        Args:
            asset: Asset identifier
            start_time: Start time for data range
            end_time: End time for data range
            interval: Data sampling frequency
            sources: List of sentiment sources to include (None for all)
            
        Returns:
            DataFrame with sentiment data
        """
        # Available sentiment sources
        all_sources = ["twitter", "reddit", "news", "fear_greed_index", "trading_view"]
        
        # Use all sources if none specified
        if sources is None:
            sources = all_sources
            
        # Validate sources
        valid_sources = [s for s in sources if s in all_sources]
        
        # Convert datetime objects to ISO format strings
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
            
        # Fetch data from sentiment API
        url = f"{self.data_warehouse_url}/api/v1/sentiment"
        params = {
            "asset": asset,
            "start_time": start_time,
            "end_time": end_time,
            "interval": interval.value,
            "sources": ",".join(valid_sources)
        }
        
        response_data = await self._fetch_api_data(
            url, DataSource.SENTIMENT_API, params
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(response_data["data"])
        
        # Ensure timestamp column is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        return df
    
    async def get_multiple_assets_data(self,
                                     assets: List[str],
                                     metric: str,
                                     start_time: Union[datetime, str],
                                     end_time: Union[datetime, str],
                                     interval: SamplingFrequency = SamplingFrequency.DAY) -> pd.DataFrame:
        """
        Get a specific metric for multiple assets, useful for correlation analysis.
        
        Args:
            assets: List of asset identifiers
            metric: Metric to retrieve (e.g., "price", "volume", "market_cap")
            start_time: Start time for data range
            end_time: End time for data range
            interval: Data sampling frequency
            
        Returns:
            DataFrame with the metric for all assets
        """
        # Convert datetime objects to ISO format strings
        if isinstance(start_time, datetime):
            start_time = start_time.isoformat()
        if isinstance(end_time, datetime):
            end_time = end_time.isoformat()
            
        # Fetch data from data warehouse
        url = f"{self.data_warehouse_url}/api/v1/assets/comparison"
        params = {
            "assets": ",".join(assets),
            "metric": metric,
            "start_time": start_time,
            "end_time": end_time,
            "interval": interval.value
        }
        
        response_data = await self._fetch_api_data(
            url, DataSource.DATA_WAREHOUSE, params
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(response_data["data"])
        
        # Ensure timestamp column is datetime
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        return df
    
    async def save_dataset(self,
                         df: pd.DataFrame,
                         name: str,
                         description: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         format: str = "parquet") -> str:
        """
        Save a dataset to the data warehouse for future use.
        
        Args:
            df: DataFrame to save
            name: Dataset name
            description: Dataset description
            tags: Tags for dataset
            format: File format ("parquet" or "csv")
            
        Returns:
            Dataset ID in the data warehouse
        """
        if format not in ["parquet", "csv"]:
            raise ValueError(f"Unsupported format: {format}")
            
        # Generate dataset ID
        dataset_id = f"{name.lower().replace(' ', '_')}_{int(time.time())}"
        
        # Serialize data
        if format == "parquet":
            buffer = pa.BufferOutputStream()
            pq.write_table(pa.Table.from_pandas(df), buffer)
            data_bytes = buffer.getvalue().to_pybytes()
        else:  # csv
            data_bytes = df.to_csv(index=False).encode("utf-8")
            
        # Upload to data warehouse
        url = f"{self.data_warehouse_url}/api/v1/datasets"
        
        # This would typically be a multipart/form-data POST request
        # For simplicity, using a placeholder implementation
        session = await self._get_http_session()
        
        data = {
            "id": dataset_id,
            "name": name,
            "description": description or "",
            "tags": tags or [],
            "format": format,
            "rows": len(df),
            "columns": list(df.columns),
            "created_at": datetime.now().isoformat()
        }
        
        files = {
            "metadata": (None, json.dumps(data), "application/json"),
            "data": (f"{dataset_id}.{format}", data_bytes, f"application/{format}")
        }
        
        async with session.post(url, data=files) as response:
            if response.status >= 400:
                error_text = await response.text()
                logger.error(f"Dataset upload error {response.status}: {error_text}")
                raise DataLoadingError(f"Dataset upload failed: {error_text}")
                
            result = await response.json()
            return result["dataset_id"]
    
    async def load_dataset(self, dataset_id: str) -> pd.DataFrame:
        """
        Load a previously saved dataset from the data warehouse.
        
        Args:
            dataset_id: Dataset ID or name
            
        Returns:
            DataFrame with the dataset
        """
        url = f"{self.data_warehouse_url}/api/v1/datasets/{dataset_id}"
        
        response_data = await self._fetch_api_data(
            url, DataSource.DATA_WAREHOUSE, {}
        )
        
        # Check format and load accordingly
        format = response_data["metadata"]["format"]
        data_url = response_data["data_url"]
        
        # Download dataset
        session = await self._get_http_session()
        async with session.get(data_url) as response:
            if response.status >= 400:
                error_text = await response.text()
                logger.error(f"Dataset download error {response.status}: {error_text}")
                raise DataLoadingError(f"Dataset download failed: {error_text}")
                
            data_bytes = await response.read()
            
        # Parse dataset
        if format == "parquet":
            table = pq.read_table(pa.py_buffer(data_bytes))
            return table.to_pandas()
        else:  # csv
            return pd.read_csv(io.BytesIO(data_bytes))
    
    async def close(self):
        """Close all connections and resources."""
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            
        if self.solana_client:
            await self.solana_client.close()


# Convenience function to create a data loader instance
async def create_data_loader(
    cache_size_mb: int = 2048,
    api_keys_file: Optional[str] = None,
    solana_rpc_url: Optional[str] = None,
    data_warehouse_url: Optional[str] = None
) -> DataLoader:
    """
    Create and initialize a DataLoader instance.
    
    Args:
        cache_size_mb: Cache size in megabytes
        api_keys_file: Path to API keys JSON file
        solana_rpc_url: Custom Solana RPC URL
        data_warehouse_url: Data warehouse URL
        
    Returns:
        Initialized DataLoader instance
    """
    # Load API keys if provided
    api_keys = {}
    if api_keys_file and os.path.exists(api_keys_file):
        try:
            with open(api_keys_file, 'r') as f:
                api_keys = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load API keys from {api_keys_file}: {str(e)}")
    
    # Create data loader
    loader = DataLoader(
        cache_size_mb=cache_size_mb,
        api_keys=api_keys,
        solana_rpc_url=solana_rpc_url,
        data_warehouse_url=data_warehouse_url
    )
    
    return loader


# Usage example
async def example_usage():
    """Example usage of the DataLoader."""
    # Create data loader
    loader = await create_data_loader(
        api_keys_file="api_keys.json",
        data_warehouse_url="https://data.minos-ai.com"
    )
    
    try:
        # Get price data for SOL/USDC
        price_data = await loader.get_price_data(
            asset="SOL/USDC",
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now(),
            interval=SamplingFrequency.HOUR,
            include_indicators=True
        )
        
        print(f"Loaded {len(price_data)} price points for SOL/USDC")
        
        # Get on-chain metrics
        onchain_data = await loader.get_on_chain_metrics(
            asset="SOL",
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now(),
            metrics=["active_addresses", "transaction_count"]
        )
        
        print(f"Loaded {len(onchain_data)} on-chain data points for SOL")
        
        # Get sentiment data
        sentiment_data = await loader.get_sentiment_data(
            asset="SOL",
            start_time=datetime.now() - timedelta(days=30),
            end_time=datetime.now(),
            sources=["twitter", "reddit"]
        )
        
        print(f"Loaded {len(sentiment_data)} sentiment data points for SOL")
        
        # Combine datasets for analysis
        price_data.set_index("timestamp", inplace=True)
        onchain_data.set_index("timestamp", inplace=True)
        sentiment_data.set_index("timestamp", inplace=True)
        
        # Resample to daily frequency for comparison
        price_daily = price_data["close"].resample("1D").last()
        onchain_daily = onchain_data.resample("1D").mean()
        sentiment_daily = sentiment_data.resample("1D").mean()
        
        # Merge datasets
        combined = pd.concat([price_daily, onchain_daily, sentiment_daily], axis=1)
        print(f"Combined dataset shape: {combined.shape}")
        
        # Save dataset
        dataset_id = await loader.save_dataset(
            df=combined,
            name="SOL_30day_analysis",
            description="Combined price, on-chain, and sentiment data for SOL",
            tags=["SOL", "analysis", "combined"]
        )
        
        print(f"Saved dataset with ID: {dataset_id}")
        
    finally:
        # Clean up resources
        await loader.close()


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run example
    asyncio.run(example_usage())