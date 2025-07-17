#!/usr/bin/env python3
"""Complete end-to-end data processing pipeline for NG FX Predictor."""

import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime, date, timedelta

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variables for local development
os.environ['USE_MOCK_DATA'] = 'true'
os.environ['MOCK_DATA_PATH'] = 'data/mock'
os.environ['DATABASE_URL'] = 'sqlite:///./local_ngfx.db'

from src.ngfx_predictor.data.database import DatabaseManager
from src.ngfx_predictor.ingestion.ingestion import DataIngestionService
from src.ngfx_predictor.features.engineering import FeatureEngineer
from src.ngfx_predictor.validation.data_validator import DataValidator
from src.ngfx_predictor.utils.logging import get_logger

logger = get_logger(__name__)


async def run_full_pipeline():
    """Run the complete data processing pipeline."""
    
    print("🚀 Starting Full Data Processing Pipeline")
    print("=" * 80)
    
    # Initialize components
    db_manager = DatabaseManager()
    ingestion_service = DataIngestionService()
    feature_engineer = FeatureEngineer()
    validator = DataValidator()
    
    try:
        # Step 1: Initialize database
        print("📊 Step 1: Initializing Database")
        print("-" * 40)
        db_manager.create_tables()
        print("✅ Database initialized successfully\n")
        
        # Step 2: Data Ingestion
        print("📥 Step 2: Data Ingestion")
        print("-" * 40)
        ingestion_results = await ingestion_service.ingest_all()
        
        total_ingested = sum(result.records_count for result in ingestion_results)
        print(f"✅ Ingested {total_ingested} records from {len(ingestion_results)} sources")
        
        # Display ingestion summary
        for result in ingestion_results:
            status = "✅" if result.success else "❌"
            print(f"   {status} {result.source}: {result.records_count} records")
        print()
        
        # Step 3: Raw Data Validation
        print("🔍 Step 3: Raw Data Validation")
        print("-" * 40)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        raw_validation_results = await validator.validate_raw_data(start_date, end_date)
        raw_validation_report = validator.generate_validation_report(raw_validation_results)
        
        print(f"✅ Raw Data Validation: {raw_validation_report['overall_status']}")
        print(f"   Quality Score: {raw_validation_report['quality_score']:.2%}")
        print(f"   Checks: {raw_validation_report['passed_checks']}/{raw_validation_report['total_checks']} passed")
        
        # Display validation summary by severity
        severity_counts = raw_validation_report['severity_counts']
        if severity_counts['critical'] > 0:
            print(f"   ⚠️  Critical: {severity_counts['critical']}")
        if severity_counts['error'] > 0:
            print(f"   ❌ Errors: {severity_counts['error']}")
        if severity_counts['warning'] > 0:
            print(f"   ⚠️  Warnings: {severity_counts['warning']}")
        print()
        
        # Step 4: Feature Engineering
        print("🔧 Step 4: Feature Engineering")
        print("-" * 40)
        
        # Create features from raw data
        features = await feature_engineer.create_features(start_date, end_date)
        
        if not features.is_empty():
            print(f"✅ Created {len(features.columns)} features for {len(features)} records")
            
            # Display feature summary
            feature_types = {}
            for col in features.columns:
                if col != 'date':
                    if 'lag' in col:
                        feature_types.setdefault('Lag Features', 0)
                        feature_types['Lag Features'] += 1
                    elif 'rolling' in col:
                        feature_types.setdefault('Rolling Stats', 0)
                        feature_types['Rolling Stats'] += 1
                    elif any(x in col for x in ['rsi', 'macd', 'bb_']):
                        feature_types.setdefault('Technical Indicators', 0)
                        feature_types['Technical Indicators'] += 1
                    elif 'sentiment' in col:
                        feature_types.setdefault('Sentiment Features', 0)
                        feature_types['Sentiment Features'] += 1
                    else:
                        feature_types.setdefault('Base Features', 0)
                        feature_types['Base Features'] += 1
            
            for feature_type, count in feature_types.items():
                print(f"   📊 {feature_type}: {count}")
            print()
            
            # Step 5: Feature Validation
            print("🧪 Step 5: Feature Validation")
            print("-" * 40)
            
            feature_validation_results = await validator.validate_features(features)
            feature_validation_report = validator.generate_validation_report(feature_validation_results)
            
            print(f"✅ Feature Validation: {feature_validation_report['overall_status']}")
            print(f"   Quality Score: {feature_validation_report['quality_score']:.2%}")
            print(f"   Checks: {feature_validation_report['passed_checks']}/{feature_validation_report['total_checks']} passed")
            
            # Display feature validation summary
            severity_counts = feature_validation_report['severity_counts']
            if severity_counts['critical'] > 0:
                print(f"   ⚠️  Critical: {severity_counts['critical']}")
            if severity_counts['error'] > 0:
                print(f"   ❌ Errors: {severity_counts['error']}")
            if severity_counts['warning'] > 0:
                print(f"   ⚠️  Warnings: {severity_counts['warning']}")
            print()
            
            # Step 6: Feature Storage
            print("💾 Step 6: Feature Storage")
            print("-" * 40)
            
            stored_records = await feature_engineer.store_features(features)
            print(f"✅ Stored {stored_records} feature records in database\n")
            
            # Step 7: Generate Final Report
            print("📋 Step 7: Pipeline Summary")
            print("-" * 40)
            
            # Create comprehensive report
            pipeline_report = {
                "timestamp": datetime.now().isoformat(),
                "pipeline_status": "SUCCESS",
                "data_ingestion": {
                    "total_records": total_ingested,
                    "sources": len(ingestion_results),
                    "successful_sources": sum(1 for r in ingestion_results if r.success),
                    "failed_sources": sum(1 for r in ingestion_results if not r.success)
                },
                "raw_data_validation": raw_validation_report,
                "feature_engineering": {
                    "total_features": len(features.columns) - 1,  # Exclude date column
                    "feature_types": feature_types,
                    "records_processed": len(features),
                    "records_stored": stored_records
                },
                "feature_validation": feature_validation_report,
                "overall_quality_score": (
                    raw_validation_report['quality_score'] + 
                    feature_validation_report['quality_score']
                ) / 2
            }
            
            # Save report to file
            report_path = "pipeline_report.json"
            with open(report_path, 'w') as f:
                json.dump(pipeline_report, f, indent=2)
            
            print(f"✅ Pipeline completed successfully!")
            print(f"   📊 Total Records: {total_ingested} ingested → {stored_records} features stored")
            print(f"   🎯 Overall Quality Score: {pipeline_report['overall_quality_score']:.2%}")
            print(f"   📋 Full report saved to: {report_path}")
            
        else:
            print("❌ No features created - check raw data availability")
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        raise
    
    print("\n🎉 Full Pipeline Complete!")
    print("=" * 80)


async def demo_feature_retrieval():
    """Demonstrate retrieving latest features."""
    
    print("\n🔍 Demo: Feature Retrieval")
    print("-" * 40)
    
    feature_engineer = FeatureEngineer()
    
    # Get latest features
    latest_features = await feature_engineer.get_latest_features(n_days=7)
    
    if not latest_features.is_empty():
        print(f"✅ Retrieved {len(latest_features)} records with {len(latest_features.columns)} features")
        
        # Display sample features
        print("\nSample Feature Data:")
        print(latest_features.head(3))
        
        # Show feature statistics
        numeric_features = [
            col for col in latest_features.columns 
            if col != 'date' and latest_features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        if numeric_features:
            print(f"\nFeature Statistics (first 5 numeric features):")
            for col in numeric_features[:5]:
                values = latest_features[col].drop_nulls()
                if len(values) > 0:
                    print(f"  📊 {col}: mean={values.mean():.4f}, std={values.std():.4f}")
    else:
        print("❌ No features found in database")


async def validate_pipeline_health():
    """Validate the health of the entire pipeline."""
    
    print("\n🏥 Pipeline Health Check")
    print("-" * 40)
    
    # Check database connectivity
    db_manager = DatabaseManager()
    try:
        db_manager.create_tables()
        print("✅ Database: Connected")
    except Exception as e:
        print(f"❌ Database: Failed - {e}")
        return False
    
    # Check data ingestion service
    ingestion_service = DataIngestionService()
    try:
        status = await ingestion_service.get_ingestion_status()
        print(f"✅ Ingestion Service: {len(status)} sources configured")
    except Exception as e:
        print(f"❌ Ingestion Service: Failed - {e}")
        return False
    
    # Check feature engineering
    feature_engineer = FeatureEngineer()
    try:
        # Test with small date range
        test_date = datetime.now().date()
        features = await feature_engineer.create_features(test_date, test_date)
        print(f"✅ Feature Engineering: Ready (test created {len(features.columns)} features)")
    except Exception as e:
        print(f"❌ Feature Engineering: Failed - {e}")
        return False
    
    # Check validation system
    validator = DataValidator()
    try:
        # Test validation with empty data
        test_results = await validator.validate_raw_data(test_date, test_date)
        print(f"✅ Validation System: Ready (ran {len(test_results)} checks)")
    except Exception as e:
        print(f"❌ Validation System: Failed - {e}")
        return False
    
    print("✅ All pipeline components healthy!")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="NG FX Predictor Pipeline")
    parser.add_argument("--action", choices=["run", "demo", "health"], default="run",
                       help="Action to perform: run full pipeline, demo features, or health check")
    
    args = parser.parse_args()
    
    if args.action == "run":
        asyncio.run(run_full_pipeline())
    elif args.action == "demo":
        asyncio.run(demo_feature_retrieval())
    elif args.action == "health":
        asyncio.run(validate_pipeline_health()) 