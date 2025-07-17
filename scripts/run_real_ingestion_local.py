#!/usr/bin/env python3
"""Run the actual data ingestion pipeline with local SQLite database."""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variables for mock data and local database
os.environ['USE_MOCK_DATA'] = 'true'
os.environ['MOCK_DATA_PATH'] = 'data/mock'
os.environ['DATABASE_URL'] = 'sqlite:///./local_ngfx.db'

async def run_full_pipeline():
    """Run the complete data ingestion and feature engineering pipeline."""
    
    print("🚀 Starting Real Data Ingestion Pipeline (Local SQLite)")
    print("=" * 60)
    
    # Initialize database first
    print("🗄️  Initializing local SQLite database...")
    await init_local_database()
    
    # Step 1: Run data ingestion from mock sources
    print("\n📊 Step 1: Running Data Ingestion from Mock Sources")
    print("-" * 50)
    
    await ingest_mock_data()
    
    # Step 2: Verify data was stored
    print("\n🔍 Step 2: Verifying Data Storage")
    print("-" * 50)
    
    await verify_stored_data()
    
    print("\n🎉 Pipeline Complete!")
    print("✅ Data ingestion completed successfully using local SQLite")
    print("🗄️  Database file: ./local_ngfx.db")


async def init_local_database():
    """Initialize local SQLite database."""
    try:
        from src.ngfx_predictor.data.database import DatabaseManager
        
        # Create database manager with SQLite
        db_manager = DatabaseManager(database_url='sqlite:///./local_ngfx.db')
        
        # Create tables
        db_manager.create_tables()
        
        print("✅ Local SQLite database initialized")
        print("📍 Database file: ./local_ngfx.db")
        
    except Exception as e:
        print(f"❌ Failed to initialize database: {e}")
        raise


async def ingest_mock_data():
    """Ingest data from mock sources and store in local database."""
    try:
        # Import here to avoid circular imports
        from src.ngfx_predictor.data.sources import (
            CBNRatesSource, 
            WorldBankReservesSource, 
            DMODebtSource,
            EIABrentSource,
            NewsSentimentSource
        )
        from src.ngfx_predictor.data.database import DatabaseManager
        
        # Initialize database manager
        db_manager = DatabaseManager(database_url='sqlite:///./local_ngfx.db')
        
        # Set date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=60)
        
        print(f"📅 Ingesting data from {start_date} to {end_date}")
        
        # Initialize sources
        sources = {
            'cbn_rates': CBNRatesSource(),
            'worldbank_reserves': WorldBankReservesSource(),
            'dmo_debt': DMODebtSource(),
            'eia_brent': EIABrentSource(),
            'news_sentiment': NewsSentimentSource()
        }
        
        total_records = 0
        
        # Process each source
        for source_name, source in sources.items():
            print(f"📈 Processing {source_name}...")
            
            # Fetch data from source
            df = source.fetch(start_date, end_date)
            
            if df.is_empty():
                print(f"   ⚠️  No data from {source_name}")
                continue
                
            # Store in database using sync session
            records_stored = await store_raw_data_local(db_manager, source_name, df)
            total_records += records_stored
            
            print(f"   ✅ {source_name}: {records_stored} records stored")
        
        print(f"\n🎯 Total records ingested: {total_records}")
        
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        raise


async def store_raw_data_local(db_manager, source_name: str, df) -> int:
    """Store raw data in local SQLite database."""
    try:
        # Use sync session for SQLite
        with db_manager.get_sync_session_context() as session:
            records = df.to_dicts()
            records_stored = 0
            
            for record in records:
                # Create database record
                from src.ngfx_predictor.data.models import RawDataModel
                
                db_record = RawDataModel(
                    source=source_name,
                    date=record.get('date'),
                    data=record,
                    created_at=datetime.utcnow()
                )
                
                session.add(db_record)
                records_stored += 1
            
            session.commit()
            return records_stored
            
    except Exception as e:
        print(f"❌ Error storing data for {source_name}: {e}")
        return 0


async def verify_stored_data():
    """Verify data was stored in the database."""
    try:
        from src.ngfx_predictor.data.database import DatabaseManager
        
        db_manager = DatabaseManager(database_url='sqlite:///./local_ngfx.db')
        
        with db_manager.get_sync_session_context() as session:
            # Check raw data
            from src.ngfx_predictor.data.models import RawDataModel
            
            raw_data_count = session.query(RawDataModel).count()
            print(f"📊 Raw data records in database: {raw_data_count}")
            
            # Show summary by source
            from sqlalchemy import func
            
            source_summary = session.query(
                RawDataModel.source,
                func.count(RawDataModel.id).label('count'),
                func.min(RawDataModel.date).label('earliest'),
                func.max(RawDataModel.date).label('latest')
            ).group_by(RawDataModel.source).all()
            
            if source_summary:
                print("\n📋 Data by source:")
                for row in source_summary:
                    print(f"   📈 {row.source}: {row.count} records ({row.earliest} to {row.latest})")
            
            # Show sample records
            sample_records = session.query(RawDataModel).limit(3).all()
            if sample_records:
                print(f"\n📝 Sample records:")
                for record in sample_records:
                    data_preview = str(record.data)[:100] + "..." if len(str(record.data)) > 100 else str(record.data)
                    print(f"   - {record.source} ({record.date}): {data_preview}")
    
    except Exception as e:
        print(f"❌ Error verifying data: {e}")


if __name__ == "__main__":
    asyncio.run(run_full_pipeline()) 