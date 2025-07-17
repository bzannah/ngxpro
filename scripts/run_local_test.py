#!/usr/bin/env python3
"""Run comprehensive local test of data ingestion, feature engineering, and validation."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.ngfx_predictor.ingestion import DataIngestionService
from src.ngfx_predictor.features import FeatureEngineer
from src.ngfx_predictor.validation import DataValidator
from src.ngfx_predictor.utils.logging import get_logger

logger = get_logger(__name__)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


async def test_data_sources():
    """Test individual data sources."""
    print_section("Testing Data Sources")
    
    from src.ngfx_predictor.data.sources import (
        CBNRatesSource,
        WorldBankReservesSource,
        DMODebtSource,
        EIABrentSource,
        NewsSentimentSource
    )
    
    sources = {
        'CBN Rates': CBNRatesSource(),
        'World Bank Reserves': WorldBankReservesSource(),
        'DMO Debt': DMODebtSource(),
        'EIA Brent': EIABrentSource(),
        'News Sentiment': NewsSentimentSource()
    }
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)
    
    for name, source in sources.items():
        try:
            print(f"\n📊 Testing {name}...")
            
            # Test health check
            health = source.get_health_status()
            print(f"   Health: {health['status']}")
            
            # Test data fetch
            df = source.fetch(start_date, end_date)
            print(f"   Fetched: {len(df)} records")
            
            if not df.is_empty():
                print(f"   Columns: {df.columns}")
                print(f"   Sample: {df.head(2)}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True


async def test_ingestion():
    """Test data ingestion service."""
    print_section("Testing Data Ingestion")
    
    ingestion_service = DataIngestionService()
    
    # Test status check
    print("📊 Checking ingestion status...")
    status = await ingestion_service.get_ingestion_status()
    print(f"   Overall health: {status['health']}")
    for source, source_status in status['sources'].items():
        print(f"   {source}: {source_status['health']}")
    
    # Test data ingestion
    print("\n📊 Ingesting data...")
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    results = await ingestion_service.ingest_all(start_date, end_date)
    
    print(f"\n✅ Ingestion Results:")
    print(f"   Total records: {results['total_records']}")
    print(f"   Errors: {len(results['errors'])}")
    
    for source, result in results['sources'].items():
        print(f"\n   {source}:")
        print(f"     - Fetched: {result['records_fetched']}")
        print(f"     - Ingested: {result['records_ingested']}")
    
    if results['errors']:
        print(f"\n❌ Errors:")
        for error in results['errors']:
            print(f"   - {error['source']}: {error['error']}")
    
    # Test data validation
    print("\n📊 Validating data quality...")
    for source in ['cbn_rates', 'worldbank_reserves', 'dmo_debt', 'eia_brent', 'news_sentiment']:
        validation = await ingestion_service.validate_data_quality(source)
        print(f"\n   {source}:")
        print(f"     Status: {validation['overall_status']}")
        for check, result in validation['checks'].items():
            print(f"     - {check}: {result['status']}")
    
    return results


async def test_feature_engineering(ingestion_results):
    """Test feature engineering."""
    print_section("Testing Feature Engineering")
    
    if ingestion_results['total_records'] == 0:
        print("⚠️  No data ingested, skipping feature engineering")
        return None
    
    feature_engineer = FeatureEngineer()
    
    # Create features
    print("📊 Creating features...")
    features = await feature_engineer.create_features()
    
    if features.is_empty():
        print("❌ No features created")
        return None
    
    print(f"\n✅ Feature Engineering Results:")
    print(f"   Records: {len(features)}")
    print(f"   Features: {len(features.columns)}")
    print(f"\n   Feature types:")
    
    # Categorize features
    base_features = [col for col in features.columns if not any(x in col for x in ['_lag_', '_rolling_', '_bb_', '_rsi_', '_macd_'])]
    lag_features = [col for col in features.columns if '_lag_' in col]
    rolling_features = [col for col in features.columns if '_rolling_' in col]
    technical_features = [col for col in features.columns if any(x in col for x in ['_bb_', '_rsi_', '_macd_'])]
    
    print(f"     - Base: {len(base_features)}")
    print(f"     - Lag: {len(lag_features)}")
    print(f"     - Rolling: {len(rolling_features)}")
    print(f"     - Technical: {len(technical_features)}")
    
    # Store features
    print("\n📊 Storing features...")
    records_stored = await feature_engineer.store_features(features)
    print(f"   Stored: {records_stored} records")
    
    # Retrieve features
    print("\n📊 Retrieving features...")
    latest_features = await feature_engineer.get_latest_features(days=7)
    print(f"   Retrieved: {len(latest_features)} records")
    
    return features


async def test_validation(features):
    """Test data validation."""
    print_section("Testing Data Validation")
    
    if features is None or features.is_empty():
        print("⚠️  No features available for validation")
        return
    
    validator = DataValidator()
    
    # Test different validation types
    test_cases = [
        ("rates", features.select([col for col in features.columns if 'rate' in col.lower()])),
        ("time_series", features),
        ("features", features)
    ]
    
    for data_type, test_data in test_cases:
        if test_data.is_empty() or len(test_data.columns) == 0:
            continue
            
        print(f"\n📊 Validating {data_type}...")
        is_valid, errors = validator.validate(test_data, data_type)
        
        print(f"   Valid: {is_valid}")
        if errors:
            print(f"   Errors ({len(errors)}):")
            for error in errors[:5]:  # Show first 5 errors
                print(f"     - {error}")
            if len(errors) > 5:
                print(f"     ... and {len(errors) - 5} more")


async def test_end_to_end():
    """Run end-to-end test."""
    print_section("End-to-End Test Summary")
    
    try:
        # Test data sources
        await test_data_sources()
        
        # Test ingestion
        ingestion_results = await test_ingestion()
        
        # Test feature engineering
        features = await test_feature_engineering(ingestion_results)
        
        # Test validation
        await test_validation(features)
        
        print_section("Test Complete")
        print("✅ All components tested successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.error("Test failed", exc_info=True)
        sys.exit(1)


async def main():
    """Main test function."""
    print("\n🚀 NG FX Predictor - Local Test Suite")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    await test_end_to_end()
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main()) 