"""Feature engineering for NG FX Predictor."""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import polars as pl
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
            # Query raw data
            result = await session.execute(
                """
                SELECT source, date, data
                FROM raw_data
                WHERE date BETWEEN :start_date AND :end_date
                ORDER BY date, source
                """,
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
        
        # Join all DataFrames on date
        if dfs:
            result_df = dfs[0]
            for df in dfs[1:]:
                result_df = result_df.join(df, on='date', how='outer')
            
            # Sort by date
            result_df = result_df.sort('date')
            
            return result_df
        
        return pl.DataFrame()
    
    def _extract_source_data(self, source: str, raw_data: Dict) -> Dict:
        """Extract relevant data from raw source data.
        
        Args:
            source: Source name
            raw_data: Raw data dictionary
            
        Returns:
            Extracted data dictionary
        """
        if source == 'cbn_rates':
            return {
                'official_rate': raw_data.get('official_rate'),
                'parallel_rate': raw_data.get('parallel_rate'),
                'buying_rate': raw_data.get('buying_rate'),
                'selling_rate': raw_data.get('selling_rate')
            }
        
        elif source == 'worldbank_reserves':
            return {
                'reserves_usd': raw_data.get('reserves_usd')
            }
        
        elif source == 'dmo_debt':
            return {
                'total_debt': raw_data.get('total_debt'),
                'external_debt': raw_data.get('external_debt'),
                'domestic_debt': raw_data.get('domestic_debt')
            }
        
        elif source == 'eia_brent':
            return {
                'oil_price': raw_data.get('price_usd')
            }
        
        elif source == 'news_sentiment':
            return {
                'sentiment_score': raw_data.get('sentiment_score'),
                'relevance_score': raw_data.get('relevance_score')
            }
        
        return {}
    
    def _create_base_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create base features from raw data.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            DataFrame with base features
        """
        # Calculate rate spreads
        if 'cbn_rates_official_rate' in df.columns and 'cbn_rates_parallel_rate' in df.columns:
            df = df.with_columns([
                (pl.col('cbn_rates_parallel_rate') - pl.col('cbn_rates_official_rate'))
                .alias('rate_spread'),
                
                ((pl.col('cbn_rates_parallel_rate') - pl.col('cbn_rates_official_rate')) / 
                 pl.col('cbn_rates_official_rate') * 100)
                .alias('rate_spread_pct')
            ])
        
        # Calculate debt ratios
        if 'dmo_debt_external_debt' in df.columns and 'dmo_debt_total_debt' in df.columns:
            df = df.with_columns([
                (pl.col('dmo_debt_external_debt') / pl.col('dmo_debt_total_debt') * 100)
                .alias('external_debt_ratio')
            ])
        
        # Forward fill missing values for non-daily data
        df = df.fill_null(strategy='forward')
        
        return df
    
    def _create_derived_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create derived features.
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with derived features
        """
        # Oil price to exchange rate correlation feature
        if 'eia_brent_oil_price' in df.columns and 'cbn_rates_official_rate' in df.columns:
            df = df.with_columns([
                (pl.col('eia_brent_oil_price') / pl.col('cbn_rates_official_rate'))
                .alias('oil_to_fx_ratio')
            ])
        
        # Reserves to debt ratio
        if 'worldbank_reserves_reserves_usd' in df.columns and 'dmo_debt_total_debt' in df.columns:
            df = df.with_columns([
                (pl.col('worldbank_reserves_reserves_usd') / pl.col('dmo_debt_total_debt'))
                .alias('reserves_to_debt_ratio')
            ])
        
        # Sentiment impact on rate spread
        if 'news_sentiment_sentiment_score' in df.columns and 'rate_spread_pct' in df.columns:
            df = df.with_columns([
                (pl.col('news_sentiment_sentiment_score') * pl.col('rate_spread_pct'))
                .alias('sentiment_spread_interaction')
            ])
        
        return df
    
    def _add_time_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        df = df.with_columns([
            pl.col('date').dt.year().alias('year'),
            pl.col('date').dt.month().alias('month'),
            pl.col('date').dt.quarter().alias('quarter'),
            pl.col('date').dt.weekday().alias('weekday'),
            pl.col('date').dt.ordinal_day().alias('day_of_year'),
            
            # Binary features
            (pl.col('date').dt.weekday() >= 5).alias('is_weekend'),
            (pl.col('date').dt.month() == 12).alias('is_december'),  # Year-end effects
            (pl.col('date').dt.month() <= 3).alias('is_q1'),  # Budget season
        ])
        
        # Cyclical encoding for month
        df = df.with_columns([
            (2 * np.pi * pl.col('month') / 12).sin().alias('month_sin'),
            (2 * np.pi * pl.col('month') / 12).cos().alias('month_cos'),
        ])
        
        # Cyclical encoding for day of year
        df = df.with_columns([
            (2 * np.pi * pl.col('day_of_year') / 365).sin().alias('day_sin'),
            (2 * np.pi * pl.col('day_of_year') / 365).cos().alias('day_cos'),
        ])
        
        return df
    
    def _validate_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Validate and clean features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validated DataFrame
        """
        # Remove features with too many missing values
        null_threshold = 0.3  # 30% threshold
        total_rows = len(df)
        
        cols_to_keep = []
        for col in df.columns:
            null_count = df[col].null_count()
            null_ratio = null_count / total_rows
            
            if null_ratio <= null_threshold:
                cols_to_keep.append(col)
            else:
                logger.warning(f"Dropping column {col} with {null_ratio:.2%} missing values")
        
        df = df.select(cols_to_keep)
        
        # Remove rows with any missing values in critical features
        critical_features = ['date', 'cbn_rates_official_rate']
        critical_exists = [col for col in critical_features if col in df.columns]
        
        if critical_exists:
            df = df.drop_nulls(subset=critical_exists)
        
        # Check for infinite values
        numeric_cols = [col for col in df.columns if df[col].dtype in [pl.Float32, pl.Float64]]
        
        for col in numeric_cols:
            # Replace infinities with nulls
            df = df.with_columns([
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col))
                .alias(col)
            ])
        
        # Remove duplicate dates
        df = df.unique(subset=['date'])
        
        logger.info(f"Validated features: {len(df)} records, {len(df.columns)} features")
        
        return df
    
    async def store_features(self, features: pl.DataFrame) -> int:
        """Store engineered features in database.
        
        Args:
            features: DataFrame with features
            
        Returns:
            Number of records stored
        """
        if features.is_empty():
            logger.warning("No features to store")
            return 0
        
        records_stored = 0
        
        async with self.db_manager.get_session() as session:
            # Convert to records
            records = features.to_dicts()
            
            for record in records:
                try:
                    # Extract date and remove from features
                    feature_date = record.pop('date')
                    
                    # Create feature model
                    feature_model = FeatureModel(
                        date=feature_date,
                        feature_vector=record,
                        feature_version=self.settings.features.version,
                        feature_count=len(record)
                    )
                    
                    session.add(feature_model)
                    records_stored += 1
                    
                except Exception as e:
                    logger.error(f"Error storing feature record: {e}")
                    continue
            
            # Commit all records
            await session.commit()
        
        logger.info(f"Stored {records_stored} feature records")
        return records_stored
    
    async def get_latest_features(
        self,
        days: int = 30
    ) -> pl.DataFrame:
        """Get latest engineered features.
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            DataFrame with features
        """
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                """
                SELECT date, feature_vector
                FROM features
                WHERE date BETWEEN :start_date AND :end_date
                AND feature_version = :version
                ORDER BY date DESC
                """,
                {
                    'start_date': start_date,
                    'end_date': end_date,
                    'version': self.settings.features.version
                }
            )
            
            records = result.fetchall()
        
        if not records:
            logger.warning("No features found in database")
            return pl.DataFrame()
        
        # Convert to DataFrame
        data = []
        for record in records:
            features = record.feature_vector
            features['date'] = record.date
            data.append(features)
        
        df = pl.DataFrame(data)
        
        # Reorder columns with date first
        cols = ['date'] + [col for col in df.columns if col != 'date']
        df = df.select(cols)
        
        return df.sort('date') 