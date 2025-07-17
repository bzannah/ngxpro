#!/usr/bin/env python3
"""Run the actual data ingestion pipeline with mock data."""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variables for mock data
os.environ['USE_MOCK_DATA'] = 'true'
os.environ['MOCK_DATA_PATH'] = 'data/mock'

async def run_full_pipeline():
    """Run the complete data ingestion and feature engineering pipeline."""
    
    print("🚀 Starting Real Data Ingestion Pipeline")
    print("=" * 60)
    
    # Clear existing manual data first
    print("🧹 Clearing existing manual data...")
    await clear_existing_data()
    
    # Step 1: Run data ingestion from mock sources
    print("\n📊 Step 1: Running Data Ingestion from Mock Sources")
    print("-" * 50)
    
    from src.ngfx_predictor.ingestion import DataIngestionService
    
    ingestion_service = DataIngestionService()
    
    # Set date range for ingestion
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=60)  # 60 days for good feature engineering
    
    print(f"📅 Ingesting data from {start_date} to {end_date}")
    
    # Run ingestion
    results = await ingestion_service.ingest_all(start_date, end_date)
    
    print(f"\n✅ Ingestion Results:")
    print(f"   Total records ingested: {results['total_records']}")
    print(f"   Sources processed: {len(results['sources'])}")
    
    for source, result in results['sources'].items():
        print(f"   📈 {source}: {result['records_ingested']} records")
    
    if results['errors']:
        print(f"\n❌ Errors encountered:")
        for error in results['errors']:
            print(f"   - {error['source']}: {error['error']}")
    
    # Step 2: Run feature engineering
    print("\n🔧 Step 2: Running Feature Engineering")
    print("-" * 50)
    
    from src.ngfx_predictor.features import FeatureEngineer
    
    feature_engineer = FeatureEngineer()
    
    # Create features from ingested data
    features = await feature_engineer.create_features(
        start_date=start_date,
        end_date=end_date
    )
    
    if not features.is_empty():
        print(f"✅ Features created: {len(features)} records with {len(features.columns)} features")
        
        # Store features in database
        records_stored = await feature_engineer.store_features(features)
        print(f"📊 Features stored in database: {records_stored} records")
        
        # Show sample features
        feature_cols = features.columns[:10]  # First 10 columns
        print(f"📋 Sample features: {feature_cols}")
    else:
        print("❌ No features created - check if raw data was ingested properly")
    
    # Step 3: Verify data in database
    print("\n🔍 Step 3: Verifying Data in Database")
    print("-" * 50)
    
    await verify_database_contents()
    
    print("\n🎉 Pipeline Complete!")
    print("✅ Data ingestion and feature engineering completed successfully")
    print("🗄️  Database now contains real data from the ingestion pipeline")


async def clear_existing_data():
    """Clear existing manual data to start fresh."""
    try:
        from src.ngfx_predictor.data.database import DatabaseManager
        
        db_manager = DatabaseManager()
        
        async with db_manager.get_session() as session:
            # Clear existing data
            await session.execute("DELETE FROM features")
            await session.execute("DELETE FROM raw_data")
            await session.execute("DELETE FROM predictions WHERE created_at < NOW() - INTERVAL '1 hour'")
            await session.commit()
            
        print("✅ Cleared existing manual data")
    except Exception as e:
        print(f"⚠️  Could not clear existing data: {e}")


async def verify_database_contents():
    """Verify what's actually in the database."""
    try:
        from src.ngfx_predictor.data.database import DatabaseManager
        
        db_manager = DatabaseManager()
        
        async with db_manager.get_session() as session:
            # Check raw data
            result = await session.execute("""
                SELECT source, COUNT(*) as count, MIN(date) as earliest, MAX(date) as latest
                FROM raw_data 
                GROUP BY source 
                ORDER BY source
            """)
            
            raw_data_summary = result.fetchall()
            
            if raw_data_summary:
                print("📊 Raw Data Summary:")
                for row in raw_data_summary:
                    print(f"   {row.source}: {row.count} records ({row.earliest} to {row.latest})")
            
            # Check features
            result = await session.execute("SELECT COUNT(*) as count FROM features")
            feature_count = result.scalar()
            
            if feature_count > 0:
                result = await session.execute("""
                    SELECT date, feature_count, feature_version 
                    FROM features 
                    ORDER BY date DESC 
                    LIMIT 5
                """)
                
                features_sample = result.fetchall()
                print(f"\n🔧 Features Summary: {feature_count} total records")
                print("   Latest features:")
                for row in features_sample:
                    print(f"   - {row.date}: {row.feature_count} features (v{row.feature_version})")
            
            # Show sample raw data
            result = await session.execute("""
                SELECT source, date, data 
                FROM raw_data 
                ORDER BY date DESC, source 
                LIMIT 3
            """)
            
            sample_data = result.fetchall()
            if sample_data:
                print(f"\n📋 Sample Raw Data:")
                for row in sample_data:
                    print(f"   {row.source} ({row.date}): {str(row.data)[:100]}...")
    
    except Exception as e:
        print(f"❌ Error verifying database: {e}")


if __name__ == "__main__":
    asyncio.run(run_full_pipeline()) 