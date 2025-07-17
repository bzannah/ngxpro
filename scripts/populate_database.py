#!/usr/bin/env python3
"""Populate the database with mock data."""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

async def populate_database():
    """Populate database with mock data."""
    print("🚀 Populating database with mock data...")
    
    # Import here to avoid import issues
    from src.ngfx_predictor.data.database import DatabaseManager
    from src.ngfx_predictor.data.models import RawDataModel, FeatureModel, PredictionModel, ModelMetadataModel
    from src.ngfx_predictor.config import get_settings
    
    settings = get_settings()
    db_manager = DatabaseManager()
    
    # Create database tables
    await db_manager.create_tables()
    
    async with db_manager.get_session() as session:
        # 1. Add raw data
        print("📊 Adding raw data...")
        today = datetime.now().date()
        
        for i in range(30):  # 30 days of data
            date = today - timedelta(days=i)
            
            # CBN rates
            cbn_data = {
                'date': date.isoformat(),
                'official_rate': 1650.0 + (i * 2),  # Simulate rate changes
                'parallel_rate': 1700.0 + (i * 2.5),
                'buying_rate': 1645.0 + (i * 2),
                'selling_rate': 1655.0 + (i * 2)
            }
            
            raw_record = RawDataModel(
                source='cbn_rates',
                date=date,
                data=cbn_data,
                ingested_at=datetime.utcnow()
            )
            session.add(raw_record)
            
            # World Bank reserves (weekly data)
            if i % 7 == 0:
                wb_data = {
                    'date': date.isoformat(),
                    'reserves_usd': 35000000000 - (i * 100000000)  # Simulate declining reserves
                }
                
                raw_record = RawDataModel(
                    source='worldbank_reserves',
                    date=date,
                    data=wb_data,
                    ingested_at=datetime.utcnow()
                )
                session.add(raw_record)
            
            # Oil prices
            oil_data = {
                'date': date.isoformat(),
                'price_usd': 75.0 + (i * 0.5)  # Simulate price changes
            }
            
            raw_record = RawDataModel(
                source='eia_brent',
                date=date,
                data=oil_data,
                ingested_at=datetime.utcnow()
            )
            session.add(raw_record)
        
        # 2. Add engineered features
        print("📊 Adding engineered features...")
        for i in range(10):  # 10 days of features
            date = today - timedelta(days=i)
            
            features = {
                'cbn_rates_official_rate': 1650.0 + (i * 2),
                'cbn_rates_parallel_rate': 1700.0 + (i * 2.5),
                'rate_spread': 50.0 + (i * 0.5),
                'rate_spread_pct': 3.0 + (i * 0.1),
                'cbn_rates_official_rate_lag_1': 1648.0 + (i * 2),
                'cbn_rates_official_rate_lag_7': 1640.0 + (i * 2),
                'eia_brent_oil_price': 75.0 + (i * 0.5),
                'oil_to_fx_ratio': 0.045 + (i * 0.0001),
                'month': date.month,
                'weekday': date.weekday(),
                'is_weekend': date.weekday() >= 5
            }
            
            feature_record = FeatureModel(
                date=date,
                feature_vector=features,
                feature_version='1.0.0',
                feature_count=len(features)
            )
            session.add(feature_record)
        
        # 3. Add predictions
        print("📊 Adding predictions...")
        for i in range(5):  # 5 recent predictions
            pred_date = today - timedelta(days=i)
            
            for horizon in [1, 3, 5]:
                prediction = PredictionModel(
                    prediction_date=pred_date,
                    horizon=horizon,
                    point_forecast=1650.0 + (i * 2) + (horizon * 5),
                    lower_bound=1600.0 + (i * 2) + (horizon * 4),
                    upper_bound=1700.0 + (i * 2) + (horizon * 6),
                    confidence=0.8,
                    model_name=f'lgb_ngn_usd_{horizon}d',
                    features_used=['rate_lag_1', 'oil_price', 'sentiment'],
                    model_version='1.0.0'
                )
                session.add(prediction)
        
        # 4. Add model metadata
        print("📊 Adding model metadata...")
        for horizon in [1, 3, 5]:
            model_meta = ModelMetadataModel(
                model_name=f'lgb_ngn_usd_{horizon}d',
                model_version='1.0.0',
                horizon=horizon,
                model_type='lightgbm',
                accuracy_score=0.85,
                training_date=datetime.utcnow(),
                is_active=True,
                hyperparameters={'n_estimators': 100, 'max_depth': 6}
            )
            session.add(model_meta)
        
        # Commit all changes
        await session.commit()
        print("✅ Database populated successfully!")
        
        # Show summary
        result = await session.execute("SELECT COUNT(*) FROM raw_data")
        raw_count = result.scalar()
        
        result = await session.execute("SELECT COUNT(*) FROM features")
        feature_count = result.scalar()
        
        result = await session.execute("SELECT COUNT(*) FROM predictions")
        pred_count = result.scalar()
        
        result = await session.execute("SELECT COUNT(*) FROM model_metadata")
        model_count = result.scalar()
        
        print(f"\n📊 Database Summary:")
        print(f"   Raw data records: {raw_count}")
        print(f"   Feature records: {feature_count}")
        print(f"   Predictions: {pred_count}")
        print(f"   Models: {model_count}")


if __name__ == "__main__":
    asyncio.run(populate_database()) 