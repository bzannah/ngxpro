#!/usr/bin/env python3
"""Simple test of data ingestion and feature engineering."""

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


async def test_ingestion_service():
    """Test the data ingestion service."""
    print("🚀 Testing Data Ingestion Service")
    print("=" * 50)
    
    try:
        # Initialize service
        ingestion_service = DataIngestionService()
        print("✅ Ingestion service initialized")
        
        # Test status
        print("\n📊 Checking ingestion status...")
        status = await ingestion_service.get_ingestion_status()
        print(f"   Overall health: {status['health']}")
        
        # Test data ingestion
        print("\n📊 Running data ingestion...")
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        
        results = await ingestion_service.ingest_all(start_date, end_date)
        
        print(f"\n✅ Ingestion Results:")
        print(f"   Total records: {results['total_records']}")
        print(f"   Errors: {len(results['errors'])}")
        
        if results['errors']:
            print("\n❌ Errors found:")
            for error in results['errors']:
                print(f"   - {error['source']}: {error['error']}")
        
        return results['total_records'] > 0
        
    except Exception as e:
        print(f"❌ Ingestion test failed: {e}")
        logger.error("Ingestion test failed", exc_info=True)
        return False


async def test_feature_engineering():
    """Test feature engineering."""
    print("\n🚀 Testing Feature Engineering")
    print("=" * 50)
    
    try:
        # Initialize feature engineer
        feature_engineer = FeatureEngineer()
        print("✅ Feature engineer initialized")
        
        # Create features
        print("\n📊 Creating features...")
        features = await feature_engineer.create_features()
        
        if features.is_empty():
            print("⚠️  No features created (likely due to insufficient data)")
            return False
        
        print(f"\n✅ Feature Engineering Results:")
        print(f"   Records: {len(features)}")
        print(f"   Features: {len(features.columns)}")
        
        # Store features
        print("\n📊 Storing features...")
        records_stored = await feature_engineer.store_features(features)
        print(f"   Stored: {records_stored} records")
        
        return records_stored > 0
        
    except Exception as e:
        print(f"❌ Feature engineering test failed: {e}")
        logger.error("Feature engineering test failed", exc_info=True)
        return False


async def main():
    """Main test function."""
    print("🚀 NG FX Predictor - Simple Test")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test ingestion
        ingestion_success = await test_ingestion_service()
        
        # Test feature engineering if ingestion was successful
        if ingestion_success:
            feature_success = await test_feature_engineering()
            
            if feature_success:
                print("\n🎉 All tests passed!")
            else:
                print("\n⚠️  Feature engineering test failed")
        else:
            print("\n⚠️  Ingestion test failed, skipping feature engineering")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(main()) 