#!/usr/bin/env python3
"""Test data ingestion with mock data."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ngfx_predictor.ingestion import DataIngestionService
from src.ngfx_predictor.features import FeatureEngineer
from src.ngfx_predictor.utils.logging import get_logger

logger = get_logger(__name__)


async def test_ingestion():
    """Test data ingestion from mock sources."""
    logger.info("Starting data ingestion test...")
    
    # Initialize ingestion service
    ingestion_service = DataIngestionService()
    
    # Test getting ingestion status
    logger.info("Getting ingestion status...")
    status = await ingestion_service.get_ingestion_status()
    logger.info(f"Ingestion status: {status}")
    
    # Test ingesting data
    logger.info("Ingesting data from all sources...")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    results = await ingestion_service.ingest_all(
        start_date=start_date,
        end_date=end_date
    )
    
    logger.info(f"Ingestion results: {results}")
    
    # Test data validation
    logger.info("Validating data quality...")
    for source_name in ingestion_service.sources:
        validation_result = await ingestion_service.validate_data_quality(source_name)
        logger.info(f"Validation for {source_name}: {validation_result}")
    
    return results


async def test_feature_engineering():
    """Test feature engineering."""
    logger.info("Starting feature engineering test...")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create features
    logger.info("Creating features...")
    features = await feature_engineer.create_features()
    
    if not features.is_empty():
        logger.info(f"Created features: {len(features)} records, {len(features.columns)} columns")
        logger.info(f"Feature columns: {features.columns[:10]}...")  # Show first 10 columns
        
        # Store features
        logger.info("Storing features in database...")
        records_stored = await feature_engineer.store_features(features)
        logger.info(f"Stored {records_stored} feature records")
        
        # Retrieve features
        logger.info("Retrieving latest features...")
        latest_features = await feature_engineer.get_latest_features(days=7)
        logger.info(f"Retrieved {len(latest_features)} feature records")
    else:
        logger.warning("No features created - check if data ingestion completed successfully")
    
    return features


async def main():
    """Main test function."""
    try:
        # Test data ingestion
        ingestion_results = await test_ingestion()
        
        # Only test feature engineering if ingestion was successful
        if ingestion_results['total_records'] > 0:
            await test_feature_engineering()
        else:
            logger.warning("No data ingested, skipping feature engineering test")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main()) 