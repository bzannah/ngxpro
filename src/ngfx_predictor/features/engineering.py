"""Feature engineering for NG FX Predictor."""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import polars as pl
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..data.models import RawDataModel, FeatureModel
from ..data.database import DatabaseManager
from ..utils.logging import get_logger
from ..utils.exceptions import FeatureEngineeringError
from ..utils.metrics import MetricsManager
from .transformers import (
    LagFeatureTransformer,
    RollingStatsTransformer,
    TechnicalIndicatorTransformer,
    SentimentTransformer
)

logger = get_logger(__name__)


class FeatureEngineer:
    """Main feature engineering class."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.metrics = MetricsManager()
        
        # Initialize transformers
        self.transformers = {
            'lag': LagFeatureTransformer(),
            'rolling': RollingStatsTransformer(),
            'technical': TechnicalIndicatorTransformer(),
            'sentiment': SentimentTransformer()
        }
        
        logger.info("Feature engineer initialized")
    
    async def create_features(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> pl.DataFrame:
        """Create features from raw data.
        
        Args:
            start_date: Start date for feature creation
            end_date: End date for feature creation
            
        Returns:
            DataFrame with engineered features
        """
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now().date()
        if start_date is None:
            start_date = end_date - timedelta(days=60)  # Need extra data for lags
        
        logger.info(f"Creating features from {start_date} to {end_date}")
        
        with self.metrics.timer("features.creation"):
            # Load raw data
            raw_data = await self._load_raw_data(start_date, end_date)
            
            if raw_data.is_empty():
                logger.warning("No raw data available for feature creation")
                return pl.DataFrame()
            
            # Create base features
            features = self._create_base_features(raw_data)
            
            # Apply transformers
            for name, transformer in self.transformers.items():
                logger.info(f"Applying {name} transformer")
                features = transformer.transform(features)
            
            # Add derived features
            features = self._create_derived_features(features)
            
            # Add time features
            features = self._add_time_features(features)
            
            # Validate features
            features = self._validate_features(features)
            
            logger.info(f"Created {len(features.columns)} features for {len(features)} records")
            
        return features
    
    async def _load_raw_data(self, start_date: date, end_date: date) -> pl.DataFrame:
        """Load raw data from database.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with raw data
        """
        async with self.db_manager.get_session() as session:
            # Query raw data with proper text() wrapper
            result = session.execute(
                text("""
                SELECT source, date, data
                FROM raw_data
                WHERE date BETWEEN :start_date AND :end_date
                ORDER BY date, source
                """),
                {'start_date': start_date, 'end_date': end_date}
            )
            
            records = result.fetchall()
        
        if not records:
            return pl.DataFrame()
        
        # Convert to structured format
        data_by_source = {}
        
        for record in records:
            source = record.source
            if source not in data_by_source:
                data_by_source[source] = []
            
            # Extract data based on source
            raw = record.data
            if isinstance(raw, str):
                import json
                raw = json.loads(raw)
            
            data = {
                'date': record.date,
                **self._extract_source_data(source, raw)
            }
            data_by_source[source].append(data)
        
        # Create DataFrames for each source
        dfs = []
        for source, data in data_by_source.items():
            if data:
                df = pl.DataFrame(data)
                # Add source prefix to columns
                df = df.rename({
                    col: f"{source}_{col}" if col != 'date' else col
                    for col in df.columns
                })
                dfs.append(df)
        
        # Merge all sources on date
        if dfs:
            result = dfs[0]
            for df in dfs[1:]:
                result = result.join(df, on='date', how='outer')
            
            # Sort by date
            result = result.sort('date')
            return result
        
        return pl.DataFrame()
    
    def _extract_source_data(self, source: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant data from raw source data.
        
        Args:
            source: Data source name
            raw_data: Raw data dictionary
            
        Returns:
            Extracted data dictionary
        """
        if source == 'cbn_rates':
            return {
                'buying_rate': raw_data.get('buying_rate'),
                'selling_rate': raw_data.get('selling_rate'),
                'central_rate': raw_data.get('central_rate'),
                'official_rate': raw_data.get('official_rate'),
                'parallel_rate': raw_data.get('parallel_rate')
            }
        elif source == 'worldbank_reserves':
            return {
                'reserves_usd': raw_data.get('reserves_usd')
            }
        elif source == 'dmo_debt':
            return {
                'total_debt': raw_data.get('total_debt'),
                'external_debt': raw_data.get('external_debt'),
                'domestic_debt': raw_data.get('domestic_debt'),
                'debt_servicing': raw_data.get('debt_servicing')
            }
        elif source == 'eia_brent':
            return {
                'price_usd': raw_data.get('price_usd')
            }
        elif source == 'news_sentiment':
            return {
                'sentiment_score': raw_data.get('sentiment_score'),
                'sentiment_label': raw_data.get('sentiment_label'),
                'relevance_score': raw_data.get('relevance_score')
            }
        else:
            return {}
    
    def _create_base_features(self, raw_data: pl.DataFrame) -> pl.DataFrame:
        """Create base features from raw data.
        
        Args:
            raw_data: Raw data DataFrame
            
        Returns:
            DataFrame with base features
        """
        if raw_data.is_empty():
            return pl.DataFrame()
        
        # Start with the raw data and add derived features
        features = raw_data.clone()
        
        # Add exchange rate features
        if 'cbn_rates_central_rate' in raw_data.columns:
            features = features.with_columns([
                pl.col('cbn_rates_central_rate').alias('exchange_rate'),
                pl.col('cbn_rates_parallel_rate').alias('parallel_rate'),
                (pl.col('cbn_rates_parallel_rate') - pl.col('cbn_rates_central_rate')).alias('rate_spread'),
                (pl.col('cbn_rates_parallel_rate') / pl.col('cbn_rates_central_rate') - 1).alias('rate_premium')
            ])
        
        # Add oil price features
        if 'eia_brent_price_usd' in raw_data.columns:
            features = features.with_columns([
                pl.col('eia_brent_price_usd').alias('oil_price')
            ])
        
        # Add debt features
        if 'dmo_debt_total_debt' in raw_data.columns:
            features = features.with_columns([
                pl.col('dmo_debt_total_debt').alias('total_debt'),
                pl.col('dmo_debt_external_debt').alias('external_debt'),
                pl.col('dmo_debt_debt_servicing').alias('debt_servicing'),
                (pl.col('dmo_debt_external_debt') / pl.col('dmo_debt_total_debt')).alias('external_debt_ratio')
            ])
        
        # Add reserves features
        if 'worldbank_reserves_reserves_usd' in raw_data.columns:
            features = features.with_columns([
                pl.col('worldbank_reserves_reserves_usd').alias('fx_reserves')
            ])
        
        # Add sentiment features
        if 'news_sentiment_sentiment_score' in raw_data.columns:
            features = features.with_columns([
                pl.col('news_sentiment_sentiment_score').alias('sentiment_score'),
                pl.col('news_sentiment_relevance_score').alias('relevance_score')
            ])
        
        return features
    
    def _create_derived_features(self, features: pl.DataFrame) -> pl.DataFrame:
        """Create derived features.
        
        Args:
            features: Input features DataFrame
            
        Returns:
            DataFrame with derived features
        """
        if features.is_empty():
            return features
        
        # Calculate percentage changes
        numeric_cols = [col for col in features.columns if features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
        
        for col in numeric_cols:
            if col != 'date':
                # Daily change
                features = features.with_columns([
                    ((pl.col(col) - pl.col(col).shift(1)) / pl.col(col).shift(1) * 100).alias(f'{col}_pct_change')
                ])
        
        # Economic indicators
        if 'exchange_rate' in features.columns and 'oil_price' in features.columns:
            features = features.with_columns([
                (pl.col('exchange_rate') / pl.col('oil_price')).alias('exchange_rate_per_oil'),
                (pl.col('oil_price') * pl.col('exchange_rate')).alias('oil_price_ngn')
            ])
        
        if 'fx_reserves' in features.columns and 'total_debt' in features.columns:
            features = features.with_columns([
                (pl.col('fx_reserves') / pl.col('total_debt')).alias('reserves_to_debt_ratio')
            ])
        
        return features
    
    def _add_time_features(self, features: pl.DataFrame) -> pl.DataFrame:
        """Add time-based features.
        
        Args:
            features: Input features DataFrame
            
        Returns:
            DataFrame with time features
        """
        if features.is_empty():
            return features
        
        # Convert date to datetime for time features
        features = features.with_columns([
            pl.col('date').cast(pl.Datetime).alias('datetime')
        ])
        
        # Add time features
        features = features.with_columns([
            pl.col('datetime').dt.year().alias('year'),
            pl.col('datetime').dt.month().alias('month'),
            pl.col('datetime').dt.day().alias('day'),
            pl.col('datetime').dt.weekday().alias('weekday'),
            pl.col('datetime').dt.ordinal_day().alias('day_of_year'),
            ((pl.col('datetime').dt.month() - 1) // 3 + 1).alias('quarter')
        ])
        
        # Add cyclical features
        features = features.with_columns([
            (2 * np.pi * pl.col('month') / 12).sin().alias('month_sin'),
            (2 * np.pi * pl.col('month') / 12).cos().alias('month_cos'),
            (2 * np.pi * pl.col('day_of_year') / 365).sin().alias('day_of_year_sin'),
            (2 * np.pi * pl.col('day_of_year') / 365).cos().alias('day_of_year_cos')
        ])
        
        # Drop intermediate datetime column
        features = features.drop('datetime')
        
        return features
    
    def _validate_features(self, features: pl.DataFrame) -> pl.DataFrame:
        """Validate engineered features.
        
        Args:
            features: Input features DataFrame
            
        Returns:
            Validated features DataFrame
        """
        if features.is_empty():
            return features
        
        # Remove rows with all NaN values (except date)
        non_date_cols = [col for col in features.columns if col != 'date']
        features = features.filter(
            pl.any_horizontal([pl.col(col).is_not_null() for col in non_date_cols])
        )
        
        # Fill forward missing values for time series
        features = features.sort('date')
        for col in non_date_cols:
            if features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
                features = features.with_columns([
                    pl.col(col).fill_null(strategy='forward').alias(col)
                ])
        
        return features
    
    async def store_features(self, features: pl.DataFrame) -> int:
        """Store engineered features in database.
        
        Args:
            features: Features DataFrame
            
        Returns:
            Number of records stored
        """
        if features.is_empty():
            return 0
        
        records_stored = 0
        
        async with self.db_manager.get_session() as session:
            for row in features.iter_rows(named=True):
                # Convert to feature vector
                feature_vector = {
                    col: val for col, val in row.items() 
                    if col != 'date' and val is not None
                }
                
                # Create feature record
                feature_record = FeatureModel(
                    date=row['date'],
                    feature_vector=feature_vector,
                    feature_count=len(feature_vector),
                    is_complete=True,
                    quality_score=1.0  # Will be updated by validation
                )
                
                session.add(feature_record)
                records_stored += 1
            
            session.commit()
        
        logger.info(f"Stored {records_stored} feature records")
        return records_stored
    
    async def get_latest_features(self, n_days: int = 30) -> pl.DataFrame:
        """Get latest engineered features.
        
        Args:
            n_days: Number of days to retrieve
            
        Returns:
            DataFrame with latest features
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=n_days)
        
        async with self.db_manager.get_session() as session:
            result = session.execute(
                text("""
                SELECT date, feature_vector
                FROM features
                WHERE date BETWEEN :start_date AND :end_date
                ORDER BY date DESC
                """),
                {'start_date': start_date, 'end_date': end_date}
            )
            
            records = result.fetchall()
        
        if not records:
            return pl.DataFrame()
        
        # Convert to DataFrame
        data = []
        for record in records:
            row = {'date': record.date}
            row.update(record.feature_vector)
            data.append(row)
        
        return pl.DataFrame(data) 