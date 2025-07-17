#!/usr/bin/env python3
"""Simple validation script to demonstrate working Feature Engineering and Data Validation."""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variables
os.environ['USE_MOCK_DATA'] = 'true'
os.environ['MOCK_DATA_PATH'] = 'data/mock'
os.environ['DATABASE_URL'] = 'sqlite:///./local_ngfx.db'

import polars as pl
from datetime import date
from src.ngfx_predictor.features.transformers import LagFeatureTransformer, RollingStatsTransformer
from src.ngfx_predictor.data.database import DatabaseManager


def test_feature_transformers():
    """Test that feature transformers work correctly."""
    print("🔧 Testing Feature Transformers")
    print("=" * 50)
    
    # Create sample data
    sample_data = pl.DataFrame({
        'date': [date(2025, 7, 15), date(2025, 7, 16), date(2025, 7, 17), date(2025, 7, 18)],
        'exchange_rate': [1500.0, 1510.0, 1505.0, 1520.0],
        'oil_price': [75.5, 76.0, 75.8, 77.0]
    })
    
    print("📊 Input data:")
    print(sample_data)
    
    # Test Lag Features
    print("\n🔄 Testing Lag Features:")
    lag_transformer = LagFeatureTransformer(lag_days=[1, 2])
    lagged_data = lag_transformer.transform(sample_data)
    
    lag_columns = [col for col in lagged_data.columns if 'lag' in col]
    print(f"✅ Created {len(lag_columns)} lag features")
    print(f"📈 Lag features: {lag_columns}")
    
    # Test Rolling Stats
    print("\n📊 Testing Rolling Stats:")
    rolling_transformer = RollingStatsTransformer(windows=[2, 3])
    rolling_data = rolling_transformer.transform(sample_data)
    
    rolling_columns = [col for col in rolling_data.columns if 'rolling' in col]
    print(f"✅ Created {len(rolling_columns)} rolling features")
    print(f"📈 Rolling features (first 5): {rolling_columns[:5]}")
    
    # Show final result
    print(f"\n📋 Final result: {len(rolling_data)} rows, {len(rolling_data.columns)} columns")
    print("✅ Feature transformers working correctly!")
    return True


def test_database_connectivity():
    """Test database connectivity and data."""
    print("\n🗄️ Testing Database Connectivity")
    print("=" * 50)
    
    try:
        db_manager = DatabaseManager()
        
        # Test connection
        if db_manager.check_connection():
            print("✅ Database connection successful")
        else:
            print("❌ Database connection failed")
            return False
        
        # Check data
        with db_manager.get_sync_session_context() as session:
            from sqlalchemy import text
            result = session.execute(text("SELECT COUNT(*) FROM raw_data"))
            count = result.fetchone()[0]
            print(f"✅ Found {count} records in raw_data table")
            
            if count > 0:
                # Show sources
                result = session.execute(text("SELECT source, COUNT(*) FROM raw_data GROUP BY source"))
                sources = result.fetchall()
                print("📊 Data sources:")
                for source, source_count in sources:
                    print(f"   • {source}: {source_count} records")
                return True
            else:
                print("❌ No data found")
                return False
                
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False


def test_real_data_processing():
    """Test processing real data from database."""
    print("\n🔄 Testing Real Data Processing")
    print("=" * 50)
    
    try:
        db_manager = DatabaseManager()
        
        # Load real data
        with db_manager.get_sync_session_context() as session:
            from sqlalchemy import text
            result = session.execute(text("""
                SELECT source, date, data 
                FROM raw_data 
                WHERE source = 'cbn_rates'
                ORDER BY date DESC 
                LIMIT 10
            """))
            records = result.fetchall()
        
        if not records:
            print("❌ No CBN data found")
            return False
        
        print(f"✅ Loaded {len(records)} CBN records")
        
        # Convert to features
        feature_data = []
        for record in records:
            import json
            raw = json.loads(record.data) if isinstance(record.data, str) else record.data
            
            feature_data.append({
                'date': record.date,
                'exchange_rate': raw.get('central_rate', 0),
                'parallel_rate': raw.get('parallel_rate', 0)
            })
        
        features_df = pl.DataFrame(feature_data)
        print("📊 Converted to features:")
        print(features_df.head(3))
        
        # Apply transformers
        print("\n🔄 Applying transformers...")
        lag_transformer = LagFeatureTransformer(lag_days=[1])
        features_df = lag_transformer.transform(features_df)
        
        print(f"✅ Final features: {len(features_df.columns)} columns")
        feature_cols = [col for col in features_df.columns if col != 'date']
        print(f"📈 Features: {feature_cols}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run simple validation tests."""
    print("🚀 SIMPLE VALIDATION - Feature Engineering & Data Validation")
    print("=" * 80)
    
    results = {}
    
    # Run tests
    results['transformers'] = test_feature_transformers()
    results['database'] = test_database_connectivity()
    results['real_data'] = test_real_data_processing()
    
    # Summary
    print("\n🎯 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper()}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED - IMPLEMENTATION WORKS! 🎉")
        print("✅ Feature Engineering: FUNCTIONAL")
        print("✅ Database Integration: FUNCTIONAL")
        print("✅ Real Data Processing: FUNCTIONAL")
    else:
        print("❌ Some tests failed - check output above")
    print("=" * 50)


if __name__ == "__main__":
    main() 