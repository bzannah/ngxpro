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
        if df.is_empty():
            return df
            
        # Select numeric columns for lag features
        numeric_cols = [
            col for col in df.columns 
            if col != 'date' and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        # Create lag features
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
            windows: List of rolling window sizes
        """
        self.windows = windows or [3, 7, 14, 30]
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create rolling statistics features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with rolling statistics
        """
        if df.is_empty():
            return df
            
        # Select numeric columns for rolling stats
        numeric_cols = [
            col for col in df.columns 
            if col != 'date' and df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            and not col.startswith('lag_')  # Skip lag features
        ]
        
        # Create rolling statistics
        for col in numeric_cols:
            for window in self.windows:
                # Rolling mean
                df = df.with_columns([
                    pl.col(col).rolling_mean(window).alias(f"{col}_rolling_mean_{window}")
                ])
                
                # Rolling std
                df = df.with_columns([
                    pl.col(col).rolling_std(window).alias(f"{col}_rolling_std_{window}")
                ])
                
                # Rolling min
                df = df.with_columns([
                    pl.col(col).rolling_min(window).alias(f"{col}_rolling_min_{window}")
                ])
                
                # Rolling max
                df = df.with_columns([
                    pl.col(col).rolling_max(window).alias(f"{col}_rolling_max_{window}")
                ])
        
        logger.info(f"Created {len(numeric_cols) * len(self.windows) * 4} rolling statistics features")
        return df


class TechnicalIndicatorTransformer(BaseTransformer):
    """Create technical indicator features."""
    
    def __init__(self, rsi_window: int = 14, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9):
        """Initialize technical indicator transformer.
        
        Args:
            rsi_window: RSI calculation window
            macd_fast: MACD fast period
            macd_slow: MACD slow period
            macd_signal: MACD signal period
        """
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create technical indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        if df.is_empty():
            return df
            
        # Select price columns for technical indicators
        price_cols = [
            col for col in df.columns 
            if any(keyword in col.lower() for keyword in ['rate', 'price', 'exchange'])
            and df[col].dtype in [pl.Float64, pl.Float32]
        ]
        
        for col in price_cols:
            # RSI
            df = self._calculate_rsi(df, col)
            
            # MACD
            df = self._calculate_macd(df, col)
            
            # Bollinger Bands
            df = self._calculate_bollinger_bands(df, col)
            
            # Price momentum
            df = self._calculate_momentum(df, col)
        
        logger.info(f"Created technical indicators for {len(price_cols)} price columns")
        return df
    
    def _calculate_rsi(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Calculate RSI (Relative Strength Index).
        
        Args:
            df: Input DataFrame
            col: Column name to calculate RSI for
            
        Returns:
            DataFrame with RSI column
        """
        # Calculate daily changes
        df = df.with_columns([
            (pl.col(col) - pl.col(col).shift(1)).alias(f"{col}_change")
        ])
        
        # Calculate gains and losses
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
        
        # Calculate rolling averages
        df = df.with_columns([
            pl.col(f"{col}_gain").rolling_mean(self.rsi_window).alias(f"{col}_avg_gain"),
            pl.col(f"{col}_loss").rolling_mean(self.rsi_window).alias(f"{col}_avg_loss")
        ])
        
        # Calculate RSI
        df = df.with_columns([
            (100 - (100 / (1 + pl.col(f"{col}_avg_gain") / pl.col(f"{col}_avg_loss"))))
            .alias(f"{col}_rsi")
        ])
        
        # Clean up temporary columns
        df = df.drop([f"{col}_change", f"{col}_gain", f"{col}_loss", f"{col}_avg_gain", f"{col}_avg_loss"])
        
        return df
    
    def _calculate_macd(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: Input DataFrame
            col: Column name to calculate MACD for
            
        Returns:
            DataFrame with MACD columns
        """
        # Calculate EMAs
        df = df.with_columns([
            pl.col(col).ewm_mean(span=self.macd_fast).alias(f"{col}_ema_fast"),
            pl.col(col).ewm_mean(span=self.macd_slow).alias(f"{col}_ema_slow")
        ])
        
        # Calculate MACD line
        df = df.with_columns([
            (pl.col(f"{col}_ema_fast") - pl.col(f"{col}_ema_slow")).alias(f"{col}_macd")
        ])
        
        # Calculate signal line
        df = df.with_columns([
            pl.col(f"{col}_macd").ewm_mean(span=self.macd_signal).alias(f"{col}_macd_signal")
        ])
        
        # Calculate histogram
        df = df.with_columns([
            (pl.col(f"{col}_macd") - pl.col(f"{col}_macd_signal")).alias(f"{col}_macd_histogram")
        ])
        
        # Clean up temporary columns
        df = df.drop([f"{col}_ema_fast", f"{col}_ema_slow"])
        
        return df
    
    def _calculate_bollinger_bands(self, df: pl.DataFrame, col: str, window: int = 20, num_std: float = 2.0) -> pl.DataFrame:
        """Calculate Bollinger Bands.
        
        Args:
            df: Input DataFrame
            col: Column name to calculate Bollinger Bands for
            window: Rolling window size
            num_std: Number of standard deviations
            
        Returns:
            DataFrame with Bollinger Bands columns
        """
        # Calculate rolling mean and std
        df = df.with_columns([
            pl.col(col).rolling_mean(window).alias(f"{col}_bb_middle"),
            pl.col(col).rolling_std(window).alias(f"{col}_bb_std")
        ])
        
        # Calculate upper and lower bands
        df = df.with_columns([
            (pl.col(f"{col}_bb_middle") + num_std * pl.col(f"{col}_bb_std")).alias(f"{col}_bb_upper"),
            (pl.col(f"{col}_bb_middle") - num_std * pl.col(f"{col}_bb_std")).alias(f"{col}_bb_lower")
        ])
        
        # Calculate bandwidth and %B
        df = df.with_columns([
            (pl.col(f"{col}_bb_upper") - pl.col(f"{col}_bb_lower")).alias(f"{col}_bb_bandwidth"),
            ((pl.col(col) - pl.col(f"{col}_bb_lower")) / 
             (pl.col(f"{col}_bb_upper") - pl.col(f"{col}_bb_lower"))).alias(f"{col}_bb_percent")
        ])
        
        # Clean up temporary columns
        df = df.drop([f"{col}_bb_std"])
        
        return df
    
    def _calculate_momentum(self, df: pl.DataFrame, col: str) -> pl.DataFrame:
        """Calculate momentum indicators.
        
        Args:
            df: Input DataFrame
            col: Column name to calculate momentum for
            
        Returns:
            DataFrame with momentum columns
        """
        # Price momentum (% change over different periods)
        for period in [5, 10, 20]:
            df = df.with_columns([
                ((pl.col(col) - pl.col(col).shift(period)) / pl.col(col).shift(period) * 100)
                .alias(f"{col}_momentum_{period}")
            ])
        
        # Rate of change
        df = df.with_columns([
            (pl.col(col) / pl.col(col).shift(10) - 1).alias(f"{col}_roc_10")
        ])
        
        return df


class SentimentTransformer(BaseTransformer):
    """Create sentiment-based features."""
    
    def __init__(self, sentiment_window: int = 7):
        """Initialize sentiment transformer.
        
        Args:
            sentiment_window: Window for sentiment aggregation
        """
        self.sentiment_window = sentiment_window
    
    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create sentiment features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with sentiment features
        """
        if df.is_empty():
            return df
            
        # Find sentiment columns
        sentiment_cols = [
            col for col in df.columns 
            if 'sentiment' in col.lower()
        ]
        
        if not sentiment_cols:
            logger.info("No sentiment columns found, skipping sentiment transformation")
            return df
        
        for col in sentiment_cols:
            # Rolling sentiment statistics
            df = df.with_columns([
                pl.col(col).rolling_mean(self.sentiment_window).alias(f"{col}_rolling_mean"),
                pl.col(col).rolling_std(self.sentiment_window).alias(f"{col}_rolling_std")
            ])
            
            # Sentiment momentum
            df = df.with_columns([
                (pl.col(col) - pl.col(col).shift(1)).alias(f"{col}_change"),
                (pl.col(col) - pl.col(col).shift(self.sentiment_window)).alias(f"{col}_momentum")
            ])
            
            # Sentiment extremes
            df = df.with_columns([
                (pl.col(col) > 0.7).alias(f"{col}_very_positive"),
                (pl.col(col) < -0.7).alias(f"{col}_very_negative"),
                (pl.col(col).abs() < 0.1).alias(f"{col}_neutral")
            ])
        
        logger.info(f"Created sentiment features for {len(sentiment_cols)} sentiment columns")
        return df 