"""Data ingestion service for NG FX Predictor."""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List

import polars as pl
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import get_settings
from ..data.sources import (
    CBNRatesSource,
    WorldBankReservesSource,
    DMODebtSource,
    EIABrentSource,
    NewsSentimentSource
)
from ..data.models import RawDataModel
from ..data.database import DatabaseManager
from ..utils.logging import get_logger
from ..utils.exceptions import DataIngestionError
from ..utils.metrics import MetricsManager

logger = get_logger(__name__)


class DataIngestionService:
    """Service for coordinating data ingestion from all sources."""
    
    def __init__(self):
        """Initialize data ingestion service."""
        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.metrics = MetricsManager()
        
        # Initialize data sources
        self.sources = {
            'cbn_rates': CBNRatesSource(),
            'worldbank_reserves': WorldBankReservesSource(),
            'dmo_debt': DMODebtSource(),
            'eia_brent': EIABrentSource(),
            'news_sentiment': NewsSentimentSource()
        }
        
        logger.info("Data ingestion service initialized")
    
    async def ingest_all(
        self, 
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Ingest data from all sources.
        
        Args:
            start_date: Start date for data ingestion
            end_date: End date for data ingestion
            
        Returns:
            Dictionary with ingestion results
        """
        # Set default date range if not provided
        if end_date is None:
            end_date = datetime.now().date()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        logger.info(f"Starting data ingestion from {start_date} to {end_date}")
        
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'sources': {},
            'total_records': 0,
            'errors': []
        }
        
        # Ingest from each source
        for source_name, source in self.sources.items():
            try:
                with self.metrics.timer(f"ingestion.{source_name}"):
                    result = await self._ingest_source(
                        source_name, source, start_date, end_date
                    )
                    results['sources'][source_name] = result
                    results['total_records'] += result['records_ingested']
                    
                    self.metrics.increment(
                        f"ingestion.{source_name}.success",
                        result['records_ingested']
                    )
                    
            except Exception as e:
                logger.error(f"Error ingesting from {source_name}: {e}")
                results['errors'].append({
                    'source': source_name,
                    'error': str(e)
                })
                self.metrics.increment(f"ingestion.{source_name}.error")
        
        # Log summary
        logger.info(
            f"Data ingestion completed. "
            f"Total records: {results['total_records']}, "
            f"Errors: {len(results['errors'])}"
        )
        
        return results
    
    async def _ingest_source(
        self,
        source_name: str,
        source: Any,
        start_date: date,
        end_date: date
    ) -> Dict[str, Any]:
        """Ingest data from a single source.
        
        Args:
            source_name: Name of the data source
            source: Data source instance
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting from {source_name}...")
        
        # Fetch data from source
        df = source.fetch(start_date, end_date)
        
        if df.is_empty():
            logger.warning(f"No data returned from {source_name}")
            return {
                'records_fetched': 0,
                'records_ingested': 0,
                'records_updated': 0,
                'errors': []
            }
        
        # Store in database
        records_ingested = await self._store_raw_data(source_name, df)
        
        return {
            'records_fetched': len(df),
            'records_ingested': records_ingested,
            'records_updated': 0,  # TODO: Implement update logic
            'errors': []
        }
    
    async def _store_raw_data(
        self,
        source_name: str,
        df: pl.DataFrame
    ) -> int:
        """Store raw data in database.
        
        Args:
            source_name: Name of the data source
            df: DataFrame with data to store
            
        Returns:
            Number of records stored
        """
        records_stored = 0
        
        async with self.db_manager.get_session() as session:
            # Convert DataFrame to records
            records = df.to_dicts()
            
            for record in records:
                try:
                    # Create raw data model
                    raw_data = RawDataModel(
                        source=source_name,
                        date=record.get('date'),
                        data=record,
                        ingested_at=datetime.utcnow()
                    )
                    
                    # Add to session
                    session.add(raw_data)
                    records_stored += 1
                    
                except Exception as e:
                    logger.error(f"Error storing record: {e}")
                    continue
            
            # Commit all records
            await session.commit()
        
        logger.info(f"Stored {records_stored} records from {source_name}")
        return records_stored
    
    async def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current ingestion status.
        
        Returns:
            Dictionary with ingestion status for each source
        """
        status = {
            'sources': {},
            'last_update': None,
            'health': 'ok'
        }
        
        for source_name, source in self.sources.items():
            try:
                # Get source health
                source_status = source.get_health_status()
                
                # Get last ingestion time from database
                last_ingestion = await self._get_last_ingestion_time(source_name)
                
                status['sources'][source_name] = {
                    'health': source_status.get('status', 'unknown'),
                    'last_ingestion': last_ingestion,
                    'is_available': source_status.get('is_available', False)
                }
                
                # Update overall health
                if source_status.get('status') == 'error':
                    status['health'] = 'degraded'
                
                # Update last update time
                if last_ingestion and (
                    status['last_update'] is None or 
                    last_ingestion > status['last_update']
                ):
                    status['last_update'] = last_ingestion
                    
            except Exception as e:
                logger.error(f"Error getting status for {source_name}: {e}")
                status['sources'][source_name] = {
                    'health': 'error',
                    'error': str(e)
                }
                status['health'] = 'degraded'
        
        return status
    
    async def _get_last_ingestion_time(self, source_name: str) -> Optional[datetime]:
        """Get last ingestion time for a source.
        
        Args:
            source_name: Name of the data source
            
        Returns:
            Last ingestion timestamp or None
        """
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                f"""
                SELECT MAX(ingested_at) as last_ingestion
                FROM raw_data
                WHERE source = :source
                """,
                {'source': source_name}
            )
            
            row = result.first()
            return row.last_ingestion if row else None
    
    async def validate_data_quality(self, source_name: str) -> Dict[str, Any]:
        """Validate data quality for a source.
        
        Args:
            source_name: Name of the data source
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating data quality for {source_name}")
        
        validation_results = {
            'source': source_name,
            'timestamp': datetime.utcnow(),
            'checks': {},
            'overall_status': 'pass'
        }
        
        # Get recent data
        async with self.db_manager.get_session() as session:
            result = await session.execute(
                f"""
                SELECT data
                FROM raw_data
                WHERE source = :source
                AND ingested_at >= :cutoff
                ORDER BY ingested_at DESC
                LIMIT 100
                """,
                {
                    'source': source_name,
                    'cutoff': datetime.utcnow() - timedelta(days=7)
                }
            )
            
            records = [row.data for row in result]
        
        if not records:
            validation_results['checks']['data_availability'] = {
                'status': 'fail',
                'message': 'No recent data available'
            }
            validation_results['overall_status'] = 'fail'
            return validation_results
        
        # Run quality checks
        validation_results['checks']['data_availability'] = {
            'status': 'pass',
            'records_found': len(records)
        }
        
        # Check for completeness
        missing_fields = self._check_completeness(records, source_name)
        if missing_fields:
            validation_results['checks']['completeness'] = {
                'status': 'fail',
                'missing_fields': missing_fields
            }
            validation_results['overall_status'] = 'fail'
        else:
            validation_results['checks']['completeness'] = {'status': 'pass'}
        
        # Check for consistency
        consistency_issues = self._check_consistency(records, source_name)
        if consistency_issues:
            validation_results['checks']['consistency'] = {
                'status': 'warning',
                'issues': consistency_issues
            }
            if validation_results['overall_status'] == 'pass':
                validation_results['overall_status'] = 'warning'
        else:
            validation_results['checks']['consistency'] = {'status': 'pass'}
        
        return validation_results
    
    def _check_completeness(self, records: List[Dict], source_name: str) -> List[str]:
        """Check data completeness.
        
        Args:
            records: List of data records
            source_name: Name of the data source
            
        Returns:
            List of missing fields
        """
        required_fields = {
            'cbn_rates': ['date', 'currency', 'official_rate'],
            'worldbank_reserves': ['date', 'reserves_usd'],
            'dmo_debt': ['date', 'total_debt', 'external_debt', 'domestic_debt'],
            'eia_brent': ['date', 'price_usd'],
            'news_sentiment': ['date', 'title', 'sentiment_score']
        }
        
        fields = required_fields.get(source_name, [])
        missing_fields = []
        
        for field in fields:
            if any(record.get(field) is None for record in records):
                missing_fields.append(field)
        
        return missing_fields
    
    def _check_consistency(self, records: List[Dict], source_name: str) -> List[str]:
        """Check data consistency.
        
        Args:
            records: List of data records
            source_name: Name of the data source
            
        Returns:
            List of consistency issues
        """
        issues = []
        
        # Source-specific consistency checks
        if source_name == 'cbn_rates':
            # Check for rate spikes
            rates = [r.get('official_rate', 0) for r in records if r.get('official_rate')]
            if rates:
                avg_rate = sum(rates) / len(rates)
                for rate in rates:
                    if abs(rate - avg_rate) / avg_rate > 0.1:  # 10% deviation
                        issues.append(f"Large rate deviation detected: {rate} vs avg {avg_rate}")
                        break
        
        elif source_name == 'eia_brent':
            # Check for negative prices
            if any(r.get('price_usd', 0) < 0 for r in records):
                issues.append("Negative oil prices detected")
        
        return issues 