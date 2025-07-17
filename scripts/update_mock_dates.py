#!/usr/bin/env python3
"""Update mock data dates to be current."""

import json
from datetime import datetime, timedelta
from pathlib import Path


def update_cbn_dates(filepath: Path):
    """Update CBN exchange rates dates."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get today and work backwards
    today = datetime.now().date()
    
    for i, rate in enumerate(data['data']['exchange_rates']):
        new_date = today - timedelta(days=i)
        rate['date'] = new_date.strftime('%Y-%m-%d')
    
    # Update metadata
    data['data']['metadata']['last_updated'] = datetime.now().isoformat() + 'Z'
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated CBN dates: {len(data['data']['exchange_rates'])} records")


def update_worldbank_dates(filepath: Path):
    """Update World Bank reserves dates."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Start from current month and work backwards
    today = datetime.now().date()
    
    for i, record in enumerate(data['data']):
        # Monthly data - go back by months
        new_date = today.replace(day=1) - timedelta(days=30*i)
        record['date'] = new_date.strftime('%Y-%m')
    
    # Update metadata
    data['metadata']['lastupdated'] = today.strftime('%Y-%m-%d')
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated World Bank dates: {len(data['data'])} records")


def update_dmo_dates(filepath: Path):
    """Update DMO debt dates."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Update total debt date
    today = datetime.now().date()
    data['data']['total_debt']['as_at'] = today.strftime('%Y-%m-%d')
    
    # Update monthly data
    for i, record in enumerate(data['data']['monthly_debt_data']):
        new_date = today.replace(day=1) - timedelta(days=30*i)
        record['date'] = new_date.strftime('%Y-%m')
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated DMO dates: {len(data['data']['monthly_debt_data'])} records")


def update_eia_dates(filepath: Path):
    """Update EIA oil prices dates."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get today and work backwards
    today = datetime.now().date()
    
    # Update the data array
    series = data['response']['data']['series'][0]
    for i, item in enumerate(series['data']):
        new_date = today - timedelta(days=i)
        item[0] = new_date.strftime('%Y-%m-%d')
    
    # Update metadata
    series['end'] = today.strftime('%Y-%m-%d')
    series['lastHistoricalPeriod'] = today.strftime('%Y-%m-%d')
    series['updated'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S-0500')
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated EIA dates: {len(series['data'])} records")


def update_news_dates(filepath: Path):
    """Update news sentiment dates."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Get today and work backwards
    today = datetime.now()
    
    for i, article in enumerate(data['articles']):
        # News articles - spread over last 2 weeks
        new_date = today - timedelta(hours=12*i)
        article['publishedAt'] = new_date.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✅ Updated news dates: {len(data['articles'])} articles")


def main():
    """Update all mock data files."""
    print("🔄 Updating mock data dates to be current...")
    
    mock_dir = Path('data/mock')
    
    # Update each data source
    update_cbn_dates(mock_dir / 'cbn_exchange_rates.json')
    update_worldbank_dates(mock_dir / 'worldbank_reserves.json')
    update_dmo_dates(mock_dir / 'dmo_debt.json')
    update_eia_dates(mock_dir / 'eia_brent_prices.json')
    update_news_dates(mock_dir / 'news_sentiment.json')
    
    print("\n✅ All mock data dates updated!")


if __name__ == '__main__':
    main() 