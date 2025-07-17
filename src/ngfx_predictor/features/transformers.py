"""Feature transformers for NG FX Predictor."""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
import polars as pl

from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseTransformer(ABC):
    """Base class for feature transformers."""
    
    @abstractmethod
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform the input DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        pass


class LagFeatureTransformer(BaseTransformer):
    """Create lag features for time series data."""
    
    def __init__(self, lag_days: Optional[List[int]] = None):
        """Initialize lag feature transformer.
        
        Args:
            lag_days: List of lag days to create
        """
        self.lag_days = lag_days or [1, 2, 3, 5, 7, 14, 30]
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create lag features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with lag features
        """
        # Select numeric columns for lag features
        numeric_cols = [
            col for col in df.columns 
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            and col not in ['year', 'month', 'quarter', 'weekday', 'day_of_year']
        ]
        
        for col in numeric_cols:
            for lag in self.lag_days:
                lag_col_name = f"{col}_lag_{lag}"
                df = df.with_columns([
                    pl.col(col).shift(lag).alias(lag_col_name)
                ])
        
        logger.info(f"Created {len(numeric_cols) * len(self.lag_days)} lag features")
        
        return df


class RollingStatsTransformer(BaseTransformer):
    """Create rolling statistics features."""
    
    def __init__(self, windows: Optional[List[int]] = None):
        """Initialize rolling stats transformer.
        
        Args:
            windows: List of window sizes for rolling stats
        """
        self.windows = windows or [7, 14, 30]
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create rolling statistics features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with rolling statistics
        """
        # Select numeric columns
        numeric_cols = [
            col for col in df.columns 
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            and col not in ['year', 'month', 'quarter', 'weekday', 'day_of_year']
            and not col.endswith('_lag_')  # Skip lag features
        ]
        
        for col in numeric_cols:
            for window in self.windows:
                # Rolling mean
                df = df.with_columns([
                    pl.col(col).rolling_mean(window_size=window)
                    .alias(f"{col}_rolling_mean_{window}")
                ])
                
                # Rolling std
                df = df.with_columns([
                    pl.col(col).rolling_std(window_size=window)
                    .alias(f"{col}_rolling_std_{window}")
                ])
                
                # Rolling min/max
                df = df.with_columns([
                    pl.col(col).rolling_min(window_size=window)
                    .alias(f"{col}_rolling_min_{window}"),
                    
                    pl.col(col).rolling_max(window_size=window)
                    .alias(f"{col}_rolling_max_{window}")
                ])
                
                # Price range (for rate columns)
                if 'rate' in col or 'price' in col:
                    df = df.with_columns([
                        (pl.col(f"{col}_rolling_max_{window}") - 
                         pl.col(f"{col}_rolling_min_{window}"))
                        .alias(f"{col}_rolling_range_{window}")
                    ])
        
        logger.info(f"Created rolling statistics for {len(numeric_cols)} columns")
        
        return df


class TechnicalIndicatorTransformer(BaseTransformer):
    """Create technical indicators for financial time series."""
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create technical indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        # Calculate RSI for rate columns
        rate_cols = [col for col in df.columns if 'rate' in col and 'official' in col]
        
        for col in rate_cols:
            df = self._add_rsi(df, col, period=14)
        
        # Calculate MACD for rate columns
        for col in rate_cols:
            df = self._add_macd(df, col)
        
        # Bollinger Bands for rate columns
        for col in rate_cols:
            df = self._add_bollinger_bands(df, col, window=20)
        
        # Calculate volatility indicators
        df = self._add_volatility_indicators(df)
        
        logger.info("Added technical indicators")
        
        return df
    
    def _add_rsi(self, df: pl.DataFrame, col: str, period: int = 14) -> pl.DataFrame:
        """Add RSI (Relative Strength Index).
        
        Args:
            df: Input DataFrame
            col: Column name
            period: RSI period
            
        Returns:
            DataFrame with RSI
        """
        # Calculate price changes
        df = df.with_columns([
            (pl.col(col) - pl.col(col).shift(1)).alias(f"{col}_change")
        ])
        
        # Separate gains and losses
        df = df.with_columns([
            pl.when(pl.col(f"{col}_change") > 0)
            .then(pl.col(f"{col}_change"))
            .otherwise(0)
            .alias(f"{col}_gain"),
            
            pl.when(pl.col(f"{col}_change") < 0)
            .then(-pl.col(f"{col}_change"))
            .otherwise(0)
            .alias(f"{col}_loss")
        ])
        
        # Calculate average gains and losses
        df = df.with_columns([
            pl.col(f"{col}_gain").rolling_mean(window_size=period).alias(f"{col}_avg_gain"),
            pl.col(f"{col}_loss").rolling_mean(window_size=period).alias(f"{col}_avg_loss")
        ])
        
        # Calculate RSI
        df = df.with_columns([
            (100 - (100 / (1 + pl.col(f"{col}_avg_gain") / pl.col(f"{col}_avg_loss"))))
            .alias(f"{col}_rsi_{period}")
        ])
        
        # Clean up intermediate columns
        df = df.drop([f"{col}_change", f"{col}_gain", f"{col}_loss", 
                      f"{col}_avg_gain", f"{col}_avg_loss"])
        
        return df
    
    def _add_macd(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Add MACD (Moving Average Convergence Divergence).
        
        Args:
            df: Input DataFrame
            col: Column name
            
        Returns:
            DataFrame with MACD
        """
        # Calculate EMAs
        df = df.with_columns([
            pl.col(col).ewm_mean(span=12).alias(f"{col}_ema_12"),
            pl.col(col).ewm_mean(span=26).alias(f"{col}_ema_26")
        ])
        
        # Calculate MACD line
        df = df.with_columns([
            (pl.col(f"{col}_ema_12") - pl.col(f"{col}_ema_26")).alias(f"{col}_macd")
        ])
        
        # Calculate signal line
        df = df.with_columns([
            pl.col(f"{col}_macd").ewm_mean(span=9).alias(f"{col}_macd_signal")
        ])
        
        # Calculate histogram
        df = df.with_columns([
            (pl.col(f"{col}_macd") - pl.col(f"{col}_macd_signal")).alias(f"{col}_macd_hist")
        ])
        
        # Clean up intermediate columns
        df = df.drop([f"{col}_ema_12", f"{col}_ema_26"])
        
        return df
    
    def _add_bollinger_bands(self, df: pl.DataFrame, col: str, window: int = 20) -> pl.DataFrame:
        """Add Bollinger Bands.
        
        Args:
            df: Input DataFrame
            col: Column name
            window: Window size
            
        Returns:
            DataFrame with Bollinger Bands
        """
        # Calculate moving average and standard deviation
        df = df.with_columns([
            pl.col(col).rolling_mean(window_size=window).alias(f"{col}_bb_middle"),
            pl.col(col).rolling_std(window_size=window).alias(f"{col}_bb_std")
        ])
        
        # Calculate bands
        df = df.with_columns([
            (pl.col(f"{col}_bb_middle") + 2 * pl.col(f"{col}_bb_std")).alias(f"{col}_bb_upper"),
            (pl.col(f"{col}_bb_middle") - 2 * pl.col(f"{col}_bb_std")).alias(f"{col}_bb_lower")
        ])
        
        # Calculate band width and position
        df = df.with_columns([
            (pl.col(f"{col}_bb_upper") - pl.col(f"{col}_bb_lower")).alias(f"{col}_bb_width"),
            ((pl.col(col) - pl.col(f"{col}_bb_lower")) / 
             (pl.col(f"{col}_bb_upper") - pl.col(f"{col}_bb_lower"))).alias(f"{col}_bb_position")
        ])
        
        # Clean up intermediate columns
        df = df.drop([f"{col}_bb_std"])
        
        return df
    
    def _add_volatility_indicators(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add volatility indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volatility indicators
        """
        # Calculate daily returns for rate columns
        rate_cols = [col for col in df.columns if 'rate' in col and 'official' in col]
        
        for col in rate_cols:
            # Daily returns
            df = df.with_columns([
                ((pl.col(col) - pl.col(col).shift(1)) / pl.col(col).shift(1) * 100)
                .alias(f"{col}_returns")
            ])
            
            # Historical volatility (20-day)
            df = df.with_columns([
                pl.col(f"{col}_returns").rolling_std(window_size=20)
                .alias(f"{col}_volatility_20")
            ])
            
            # GARCH-like conditional volatility
            df = df.with_columns([
                (pl.col(f"{col}_returns").pow(2).rolling_mean(window_size=5))
                .sqrt()
                .alias(f"{col}_conditional_vol")
            ])
        
        return df


class SentimentTransformer(BaseTransformer):
    """Transform sentiment data into features."""
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform sentiment features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with sentiment features
        """
        sentiment_cols = [col for col in df.columns if 'sentiment' in col]
        
        if not sentiment_cols:
            logger.warning("No sentiment columns found")
            return df
        
        for col in sentiment_cols:
            if 'score' in col:
                # Sentiment momentum
                df = df.with_columns([
                    (pl.col(col) - pl.col(col).shift(1)).alias(f"{col}_momentum"),
                    (pl.col(col) - pl.col(col).shift(7)).alias(f"{col}_weekly_change")
                ])
                
                # Sentiment volatility
                df = df.with_columns([
                    pl.col(col).rolling_std(window_size=7).alias(f"{col}_volatility")
                ])
                
                # Sentiment categories
                df = df.with_columns([
                    pl.when(pl.col(col) > 0.5).then(1)
                    .when(pl.col(col) < -0.5).then(-1)
                    .otherwise(0)
                    .alias(f"{col}_category")
                ])
                
                # Cumulative sentiment
                df = df.with_columns([
                    pl.col(col).cumsum().alias(f"{col}_cumulative")
                ])
        
        logger.info("Added sentiment features")
        
        return df 