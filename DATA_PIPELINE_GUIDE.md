# NG FX Predictor - Data Pipeline Guide

This guide covers the complete data pipeline including data ingestion, feature engineering, and validation components.

## Overview

The data pipeline consists of three main components:

1. **Data Ingestion**: Fetches data from multiple sources (CBN, World Bank, DMO, EIA, News)
2. **Feature Engineering**: Transforms raw data into ML-ready features
3. **Data Validation**: Ensures data quality and integrity

## Data Ingestion

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Data Sources   │────▶│ Ingestion Service│────▶│   PostgreSQL   │
└─────────────────┘     └──────────────────┘     └────────────────┘
       │                         │                        │
       ├─ CBN API               ├─ Fetch                ├─ raw_data
       ├─ World Bank            ├─ Transform            └─ features
       ├─ DMO                   ├─ Validate
       ├─ EIA                   └─ Store
       └─ News API
```

### Running Data Ingestion

#### Manual Ingestion

```python
from src.ngfx_predictor.ingestion import DataIngestionService
from datetime import datetime, timedelta

# Initialize service
ingestion_service = DataIngestionService()

# Ingest last 30 days
end_date = datetime.now().date()
start_date = end_date - timedelta(days=30)

results = await ingestion_service.ingest_all(start_date, end_date)
```

#### Scheduled Ingestion

The system automatically runs ingestion at 2 AM daily. To customize:

```env
SCHEDULER_DAILY_INGESTION_HOUR=2
SCHEDULER_DAILY_INGESTION_MINUTE=0
SCHEDULER_ENABLE_AUTO_INGESTION=true
```

### Data Sources

1. **CBN Exchange Rates**
   - Official, parallel, buying, selling rates
   - Daily updates
   - USD/NGN pairs

2. **World Bank Reserves**
   - Foreign reserves data
   - Monthly updates
   - Additional economic indicators

3. **DMO Public Debt**
   - Total, external, domestic debt
   - Monthly updates
   - Debt composition details

4. **EIA Oil Prices**
   - Brent crude prices
   - Daily updates
   - Price forecasts

5. **News Sentiment**
   - Financial news articles
   - Sentiment scores (-1 to 1)
   - Relevance scores

## Feature Engineering

### Feature Types

The system creates multiple feature types:

1. **Base Features**
   - Rate spreads (parallel - official)
   - Debt ratios (external/total)
   - Oil-to-FX ratios

2. **Lag Features**
   - Previous values (1, 2, 3, 5, 7, 14, 30 days)
   - For all numeric columns

3. **Rolling Statistics**
   - Mean, std, min, max over windows (7, 14, 30 days)
   - Price ranges for rate columns

4. **Technical Indicators**
   - RSI (14-day)
   - MACD (12, 26, 9)
   - Bollinger Bands (20-day)
   - Volatility measures

5. **Time Features**
   - Year, month, quarter, weekday
   - Cyclical encodings
   - Binary indicators (weekend, Q1, December)

6. **Sentiment Features**
   - Momentum and weekly changes
   - Volatility
   - Categories and cumulative scores

### Creating Features

```python
from src.ngfx_predictor.features import FeatureEngineer

# Initialize
feature_engineer = FeatureEngineer()

# Create features
features = await feature_engineer.create_features()

# Store in database
records_stored = await feature_engineer.store_features(features)

# Retrieve latest
latest_features = await feature_engineer.get_latest_features(days=30)
```

### Feature Configuration

Customize feature engineering in settings:

```python
# src/ngfx_predictor/config/settings.py
FeatureSettings:
    lag_days = [1, 2, 3, 5, 7, 14, 30]
    rolling_windows = [7, 14, 30]
    enable_technical_indicators = True
    enable_sentiment_features = True
    version = "1.0.0"
```

## Data Validation

### Validation Types

1. **Rate Validation**
   - Checks for negative values
   - Validates reasonable ranges (100-2000 NGN/USD)
   - Monitors daily changes (<20%)
   - Validates spreads

2. **Time Series Validation**
   - Checks for duplicate dates
   - Ensures chronological order
   - Identifies missing dates
   - Flags stale data (>7 days old)

3. **Feature Validation**
   - Minimum feature count
   - High correlation detection (>0.95)
   - Outlier detection (>3 std)
   - Data leakage checks

### Running Validation

```python
from src.ngfx_predictor.validation import DataValidator

validator = DataValidator()

# Validate different data types
is_valid, errors = validator.validate(data, data_type="rates")
is_valid, errors = validator.validate(data, data_type="time_series")
is_valid, errors = validator.validate(data, data_type="features")
```

### Validation Rules

| Check | Threshold | Action |
|-------|-----------|--------|
| Missing values | >30% | Drop column |
| Outliers | >5% of data | Warning |
| Rate changes | >20% daily | Error |
| Correlation | >0.95 | Warning |
| Stale data | >7 days | Error |

## Testing

### Running Tests

1. **Test Individual Components**:
   ```bash
   python scripts/test_ingestion.py
   ```

2. **Run End-to-End Test**:
   ```bash
   python scripts/run_local_test.py
   ```

3. **Update Mock Data Dates**:
   ```bash
   python scripts/update_mock_dates.py
   ```

### Test Coverage

The test suite covers:
- Data source connectivity
- Mock data loading
- Ingestion pipeline
- Feature creation
- Data validation
- Database operations

## Monitoring

### Health Checks

```python
# Check ingestion status
status = await ingestion_service.get_ingestion_status()
# Returns: {
#   'sources': {...},
#   'last_update': datetime,
#   'health': 'ok|degraded|error'
# }

# Validate data quality
validation = await ingestion_service.validate_data_quality('cbn_rates')
# Returns: {
#   'source': 'cbn_rates',
#   'checks': {...},
#   'overall_status': 'pass|warning|fail'
# }
```

### Metrics

The system tracks:
- Ingestion success/failure rates
- Records processed per source
- Feature engineering time
- Validation pass rates

Access metrics at `/metrics` endpoint.

## Troubleshooting

### Common Issues

1. **No data returned from source**
   - Check mock data dates are current
   - Verify source health status
   - Check API credentials

2. **Feature engineering fails**
   - Ensure sufficient historical data (60+ days)
   - Check for missing required columns
   - Verify data types

3. **Validation errors**
   - Review error messages for specific issues
   - Check data quality thresholds
   - Adjust validation rules if needed

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=DEBUG
DEBUG=true
```

## Best Practices

1. **Data Ingestion**
   - Run daily at low-traffic hours
   - Keep 90 days of historical data
   - Monitor source availability

2. **Feature Engineering**
   - Version features for reproducibility
   - Document feature definitions
   - Monitor feature importance

3. **Validation**
   - Set appropriate thresholds
   - Log validation failures
   - Alert on critical errors

## API Reference

### Ingestion Service

```python
class DataIngestionService:
    async def ingest_all(start_date, end_date) -> Dict
    async def get_ingestion_status() -> Dict
    async def validate_data_quality(source_name) -> Dict
```

### Feature Engineer

```python
class FeatureEngineer:
    async def create_features(start_date, end_date) -> pl.DataFrame
    async def store_features(features) -> int
    async def get_latest_features(days) -> pl.DataFrame
```

### Data Validator

```python
class DataValidator:
    def validate(data, data_type) -> Tuple[bool, List[str]]
```

## Next Steps

1. Set up model training pipeline
2. Configure production deployment
3. Implement real-time predictions
4. Add custom data sources 