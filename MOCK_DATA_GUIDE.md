# Mock Data Adapter System - Complete Guide

## 🎯 Overview

We've successfully implemented a comprehensive mock data adapter system for the NG FX Predictor that allows seamless switching between real API data and mock data for development and testing.

## 🏗️ Architecture

### Environment-Based Switching
- **Production**: `USE_MOCK_DATA=false` → Uses real APIs
- **Development/Testing**: `USE_MOCK_DATA=true` → Uses mock data from JSON files

### Base Data Source Pattern
All data sources inherit from `BaseDataSource` which provides:
- `_is_mock_mode()` → Checks if mock data is enabled
- `_load_mock_data(filename)` → Loads JSON files from mock directory
- Automatic API session management (disabled in mock mode)

## 📊 Data Sources Implemented

### 1. CBN Exchange Rates
- **File**: `data/mock/cbn_exchange_rates.json`
- **Source**: `src/ngfx_predictor/data/sources/cbn.py`
- **Data**: 30 days of USD/NGN rates with buying/selling/parallel rates
- **Mock Method**: `_fetch_mock_data()` filters by date range

### 2. World Bank Reserves
- **File**: `data/mock/worldbank_reserves.json`
- **Source**: `src/ngfx_predictor/data/sources/worldbank.py`
- **Data**: 24 months of foreign exchange reserves
- **Mock Method**: Processes monthly data with additional economic indicators

### 3. DMO Debt Data
- **File**: `data/mock/dmo_debt.json`
- **Source**: `src/ngfx_predictor/data/sources/dmo.py`
- **Data**: 12 months of public debt information
- **Mock Method**: Includes total, external, domestic debt breakdown

### 4. EIA Oil Prices
- **File**: `data/mock/eia_brent_prices.json`
- **Source**: `src/ngfx_predictor/data/sources/eia.py`
- **Data**: 60 days of Brent crude oil prices
- **Mock Method**: Processes daily price data with metadata

### 5. News Sentiment
- **File**: `data/mock/news_sentiment.json`
- **Source**: `src/ngfx_predictor/data/sources/news.py`
- **Data**: 15 recent news articles with sentiment analysis
- **Mock Method**: Includes sentiment scores and relevance metrics

## 🔧 Configuration

### Environment Variables
```bash
# Enable mock data mode
USE_MOCK_DATA=true
MOCK_DATA_PATH=data/mock
```

### Docker Configuration
```yaml
# docker-compose.yaml
environment:
  - USE_MOCK_DATA=true
  - MOCK_DATA_PATH=/app/data/mock
volumes:
  - ./data/mock:/app/data/mock
```

### Settings Integration
```python
# src/ngfx_predictor/config/settings.py
class Settings(BaseSettings):
    use_mock_data: bool = Field(default=False)
    mock_data_path: str = Field(default="data/mock")
```

## 🚀 Usage Examples

### Basic Data Fetching
```python
# All data sources automatically detect mock mode
from ngfx_predictor.data.sources import CBNRatesSource

source = CBNRatesSource()
df = source.fetch()  # Returns mock data if USE_MOCK_DATA=true
```

### Date Range Filtering
```python
from datetime import date, timedelta

# Mock data is automatically filtered by date range
start_date = date(2024, 12, 1)
end_date = date(2025, 1, 15)
df = source.fetch(start_date, end_date)
```

### Health Status with Mock Data
```python
# Health checks work with both real and mock data
health = source.get_health_status()
print(health['status'])  # 'ok' or 'error'
```

## 🧪 Testing

### Current Status
✅ **Mock data mode is working correctly**
- Mock data loads successfully from JSON files
- Environment switching is functional
- All 5 data sources are implemented
- Docker integration is complete

### Evidence from Logs
```
Mock data mode enabled for cbn_rates
Loaded mock data from /app/data/mock/cbn_exchange_rates.json
```

### Test API Endpoints
```bash
# Test the API with mock data
curl http://localhost:8001/healthz | jq .
curl http://localhost:8001/data/status | jq .
```

## 📁 File Structure
```
data/mock/
├── README.md                    # Documentation
├── cbn_exchange_rates.json      # CBN rates data
├── worldbank_reserves.json      # World Bank data  
├── dmo_debt.json               # DMO debt data
├── eia_brent_prices.json       # Oil prices
└── news_sentiment.json         # News articles

src/ngfx_predictor/data/
├── base.py                     # Base class with mock support
└── sources/
    ├── __init__.py            # All sources exported
    ├── cbn.py                 # CBN + mock support
    ├── worldbank.py           # World Bank + mock support
    ├── dmo.py                 # DMO + mock support
    ├── eia.py                 # EIA + mock support
    └── news.py                # News + mock support
```

## 🎯 Benefits

### For Development
- **No API keys needed** for local development
- **Consistent data** across different environments
- **Fast iteration** without network dependencies
- **Offline development** capability

### For Testing
- **Reproducible tests** with known data
- **Edge case testing** with custom mock data
- **Integration testing** without external dependencies
- **CI/CD friendly** - no API rate limits

### For Production
- **Graceful fallback** - can switch to mock data if APIs fail
- **Cost efficiency** - reduce API calls during development
- **Performance** - local data is faster than API calls
- **Reliability** - no external service dependencies

## 🔄 Switching Between Modes

### Development Mode (Mock Data)
```bash
# .env
USE_MOCK_DATA=true
MOCK_DATA_PATH=data/mock
```

### Production Mode (Real APIs)
```bash
# .env
USE_MOCK_DATA=false
CBN_API_KEY=your_api_key
WORLDBANK_API_KEY=your_api_key
# ... other API keys
```

## 🛠️ Extending the System

### Adding New Data Sources
1. Create new JSON file in `data/mock/`
2. Create new source class inheriting from `BaseDataSource`
3. Implement `_fetch_mock_data()` method
4. Add environment switching logic
5. Export from `__init__.py`

### Updating Mock Data
1. Ensure date formats match expected patterns
2. Maintain realistic data relationships
3. Update README with new data characteristics
4. Test with different date ranges

## 📈 Next Steps

1. **Implement remaining ML components** using mock data
2. **Add model training** with mock data pipeline
3. **Create end-to-end tests** using full mock data stack
4. **Add more comprehensive mock data** for edge cases
5. **Implement data validation** for mock data quality

## 🎉 Success Metrics

✅ **All 5 data sources implemented** with mock support  
✅ **Environment-based switching** working correctly  
✅ **Docker integration** complete  
✅ **Mock data loading** verified in logs  
✅ **API endpoints** responding with mock data  
✅ **Professional code quality** with proper error handling  

The mock data adapter system is now fully functional and ready for end-to-end testing and development! 