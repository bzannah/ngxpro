"""Scheduler for automated data ingestion."""

import asyncio
from datetime import datetime, time, timedelta
from typing import Optional, Dict, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from ..config import get_settings
from ..utils.logging import get_logger
from .ingestion import DataIngestionService

logger = get_logger(__name__)


class IngestionScheduler:
    """Scheduler for automated data ingestion."""
    
    def __init__(self):
        """Initialize ingestion scheduler."""
        self.settings = get_settings()
        self.scheduler = AsyncIOScheduler()
        self.ingestion_service = DataIngestionService()
        self.is_running = False
        
        logger.info("Ingestion scheduler initialized")
    
    def start(self):
        """Start the scheduler."""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        # Schedule daily ingestion
        self.scheduler.add_job(
            self._run_daily_ingestion,
            CronTrigger(
                hour=self.settings.scheduler.daily_ingestion_hour,
                minute=self.settings.scheduler.daily_ingestion_minute
            ),
            id='daily_ingestion',
            name='Daily Data Ingestion',
            misfire_grace_time=3600  # 1 hour grace time
        )
        
        # Schedule hourly validation
        self.scheduler.add_job(
            self._run_data_validation,
            CronTrigger(minute=0),  # Every hour at minute 0
            id='hourly_validation',
            name='Hourly Data Validation'
        )
        
        # Schedule immediate ingestion for testing
        if self.settings.debug:
            self.scheduler.add_job(
                self._run_initial_ingestion,
                'date',
                run_date=datetime.now() + timedelta(seconds=10),
                id='initial_ingestion',
                name='Initial Data Ingestion'
            )
        
        self.scheduler.start()
        self.is_running = True
        
        logger.info("Ingestion scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        self.scheduler.shutdown(wait=True)
        self.is_running = False
        
        logger.info("Ingestion scheduler stopped")
    
    async def _run_daily_ingestion(self):
        """Run daily data ingestion."""
        logger.info("Starting scheduled daily ingestion")
        
        try:
            # Ingest last 7 days of data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=7)
            
            results = await self.ingestion_service.ingest_all(
                start_date=start_date,
                end_date=end_date
            )
            
            # Log results
            logger.info(
                f"Daily ingestion completed. "
                f"Total records: {results['total_records']}, "
                f"Errors: {len(results['errors'])}"
            )
            
            # Send notification if there were errors
            if results['errors']:
                await self._send_error_notification(results['errors'])
                
        except Exception as e:
            logger.error(f"Error in daily ingestion: {e}")
            await self._send_error_notification([{
                'source': 'scheduler',
                'error': str(e)
            }])
    
    async def _run_data_validation(self):
        """Run hourly data validation."""
        logger.info("Starting scheduled data validation")
        
        try:
            # Validate data quality for each source
            validation_results = {}
            
            for source_name in self.ingestion_service.sources:
                result = await self.ingestion_service.validate_data_quality(source_name)
                validation_results[source_name] = result
                
                # Log warnings or failures
                if result['overall_status'] != 'pass':
                    logger.warning(
                        f"Data validation {result['overall_status']} for {source_name}: "
                        f"{result['checks']}"
                    )
            
            # Send notification if there are failures
            failures = [
                source for source, result in validation_results.items()
                if result['overall_status'] == 'fail'
            ]
            
            if failures:
                await self._send_validation_notification(failures)
                
        except Exception as e:
            logger.error(f"Error in data validation: {e}")
    
    async def _run_initial_ingestion(self):
        """Run initial ingestion for testing."""
        logger.info("Running initial ingestion for testing")
        
        try:
            # Ingest last 30 days of data
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)
            
            results = await self.ingestion_service.ingest_all(
                start_date=start_date,
                end_date=end_date
            )
            
            logger.info(
                f"Initial ingestion completed. "
                f"Total records: {results['total_records']}"
            )
            
        except Exception as e:
            logger.error(f"Error in initial ingestion: {e}")
    
    async def _send_error_notification(self, errors: list):
        """Send error notification.
        
        Args:
            errors: List of error dictionaries
        """
        # TODO: Implement email/webhook notification
        logger.error(f"Ingestion errors: {errors}")
    
    async def _send_validation_notification(self, failures: list):
        """Send validation failure notification.
        
        Args:
            failures: List of failed sources
        """
        # TODO: Implement email/webhook notification
        logger.error(f"Validation failures: {failures}")
    
    def get_jobs(self) -> list:
        """Get list of scheduled jobs.
        
        Returns:
            List of job dictionaries
        """
        jobs = []
        
        for job in self.scheduler.get_jobs():
            jobs.append({
                'id': job.id,
                'name': job.name,
                'next_run': job.next_run_time,
                'trigger': str(job.trigger)
            })
        
        return jobs
    
    async def trigger_ingestion(
        self,
        source_name: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Manually trigger data ingestion.
        
        Args:
            source_name: Specific source to ingest (None for all)
            start_date: Start date for ingestion
            end_date: End date for ingestion
            
        Returns:
            Ingestion results
        """
        logger.info(f"Manually triggered ingestion for {source_name or 'all sources'}")
        
        if source_name:
            # Ingest single source
            source = self.ingestion_service.sources.get(source_name)
            if not source:
                raise ValueError(f"Unknown source: {source_name}")
            
            result = await self.ingestion_service._ingest_source(
                source_name, source, start_date, end_date
            )
            
            return {source_name: result}
        else:
            # Ingest all sources
            return await self.ingestion_service.ingest_all(start_date, end_date) 