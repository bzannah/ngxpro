#!/usr/bin/env python3
"""Test script to demonstrate feature engineering and validation with existing data."""

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

from src.ngfx_predictor.features.engineering import FeatureEngineer
from src.ngfx_predictor.validation.data_validator import DataValidator
from src.ngfx_predictor.utils.logging import get_logger

logger = get_logger(__name__)


async def test_feature_engineering():
    """Test feature engineering with existing data."""
    
    print("🔧 Testing Feature Engineering")
    print("=" * 50)
    
    feature_engineer = FeatureEngineer()
    
    # Test feature creation
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    print(f"Creating features from {start_date} to {end_date}")
    
    try:
        features = await feature_engineer.create_features(start_date, end_date)
        
        if not features.is_empty():
            print(f"✅ Successfully created {len(features.columns)} features for {len(features)} records")
            
            # Show sample features
            print("\n📊 Sample Feature Data:")
            print("Columns:", features.columns[:10])  # First 10 columns
            
            # Show feature types
            feature_types = {
                'Base Features': 0,
                'Lag Features': 0,
                'Rolling Stats': 0,
                'Technical Indicators': 0,
                'Time Features': 0
            }
            
            for col in features.columns:
                if col != 'date':
                    if 'lag' in col:
                        feature_types['Lag Features'] += 1
                    elif 'rolling' in col:
                        feature_types['Rolling Stats'] += 1
                    elif any(x in col for x in ['rsi', 'macd', 'bb_']):
                        feature_types['Technical Indicators'] += 1
                    elif any(x in col for x in ['month', 'quarter', 'year']):
                        feature_types['Time Features'] += 1
                    else:
                        feature_types['Base Features'] += 1
            
            print("\n📈 Feature Types:")
            for feature_type, count in feature_types.items():
                print(f"  {feature_type}: {count}")
            
            # Test feature storage
            print("\n💾 Testing Feature Storage:")
            stored_records = await feature_engineer.store_features(features)
            print(f"✅ Stored {stored_records} feature records")
            
            return features
            
        else:
            print("❌ No features created - check raw data")
            return None
            
    except Exception as e:
        print(f"❌ Feature engineering failed: {e}")
        return None


async def test_data_validation():
    """Test data validation with existing data."""
    
    print("\n🔍 Testing Data Validation")
    print("=" * 50)
    
    validator = DataValidator()
    
    # Test raw data validation
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    print(f"Validating raw data from {start_date} to {end_date}")
    
    try:
        validation_results = await validator.validate_raw_data(start_date, end_date)
        
        if validation_results:
            print(f"✅ Completed {len(validation_results)} validation checks")
            
            # Generate report
            report = validator.generate_validation_report(validation_results)
            
            print(f"\n📋 Validation Report:")
            print(f"  Overall Status: {report['overall_status']}")
            print(f"  Quality Score: {report['quality_score']:.2%}")
            print(f"  Passed Checks: {report['passed_checks']}/{report['total_checks']}")
            
            # Show severity breakdown
            severity_counts = report['severity_counts']
            print(f"\n📊 Validation Results:")
            print(f"  ✅ Info: {severity_counts['info']}")
            print(f"  ⚠️  Warning: {severity_counts['warning']}")
            print(f"  ❌ Error: {severity_counts['error']}")
            print(f"  🚨 Critical: {severity_counts['critical']}")
            
            # Show sample validation details
            print(f"\n🔍 Sample Validation Results:")
            for i, result in enumerate(validation_results[:3]):  # Show first 3
                status = "✅" if result.passed else "❌"
                print(f"  {status} {result.check_name}: {result.message}")
            
            return validation_results
            
        else:
            print("❌ No validation results")
            return None
            
    except Exception as e:
        print(f"❌ Data validation failed: {e}")
        return None


async def test_feature_validation(features):
    """Test feature validation."""
    
    if features is None:
        print("❌ No features to validate")
        return None
    
    print("\n🧪 Testing Feature Validation")
    print("=" * 50)
    
    validator = DataValidator()
    
    try:
        feature_results = await validator.validate_features(features)
        
        if feature_results:
            print(f"✅ Completed {len(feature_results)} feature validation checks")
            
            # Generate report
            report = validator.generate_validation_report(feature_results)
            
            print(f"\n📋 Feature Validation Report:")
            print(f"  Overall Status: {report['overall_status']}")
            print(f"  Quality Score: {report['quality_score']:.2%}")
            print(f"  Passed Checks: {report['passed_checks']}/{report['total_checks']}")
            
            # Show severity breakdown
            severity_counts = report['severity_counts']
            print(f"\n📊 Feature Validation Results:")
            print(f"  ✅ Info: {severity_counts['info']}")
            print(f"  ⚠️  Warning: {severity_counts['warning']}")
            print(f"  ❌ Error: {severity_counts['error']}")
            print(f"  🚨 Critical: {severity_counts['critical']}")
            
            return feature_results
            
        else:
            print("❌ No feature validation results")
            return None
            
    except Exception as e:
        print(f"❌ Feature validation failed: {e}")
        return None


async def main():
    """Run all tests."""
    
    print("🚀 Testing Feature Engineering & Validation Pipeline")
    print("=" * 80)
    
    # Test feature engineering
    features = await test_feature_engineering()
    
    # Test data validation
    data_validation_results = await test_data_validation()
    
    # Test feature validation
    feature_validation_results = await test_feature_validation(features)
    
    # Summary
    print("\n🎉 Test Summary")
    print("=" * 50)
    
    if features is not None:
        print("✅ Feature Engineering: PASSED")
    else:
        print("❌ Feature Engineering: FAILED")
    
    if data_validation_results is not None:
        print("✅ Data Validation: PASSED")
    else:
        print("❌ Data Validation: FAILED")
    
    if feature_validation_results is not None:
        print("✅ Feature Validation: PASSED")
    else:
        print("❌ Feature Validation: FAILED")
    
    # Check if we have working pipeline
    if all([features is not None, data_validation_results is not None, feature_validation_results is not None]):
        print("\n🎯 Full Processing Layer: FUNCTIONAL! ✅")
        
        # Create summary report
        summary = {
            "timestamp": datetime.now().isoformat(),
            "feature_engineering": {
                "status": "SUCCESS",
                "features_created": len(features.columns) if features is not None else 0,
                "records_processed": len(features) if features is not None else 0
            },
            "data_validation": {
                "status": "SUCCESS",
                "checks_performed": len(data_validation_results) if data_validation_results else 0
            },
            "feature_validation": {
                "status": "SUCCESS",
                "checks_performed": len(feature_validation_results) if feature_validation_results else 0
            }
        }
        
        with open("test_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("📋 Test results saved to test_results.json")
        
    else:
        print("\n❌ Pipeline has issues - check error messages above")


if __name__ == "__main__":
    asyncio.run(main()) 