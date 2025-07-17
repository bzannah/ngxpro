"""Data validators for NG FX Predictor."""

from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import polars as pl

from ..config import get_settings
from ..utils.logging import get_logger
from ..utils.exceptions import DataQualityError

logger = get_logger(__name__)


class BaseValidator(ABC):
    """Base class for data validators."""
    
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate data.
        
        Args:
            data: Data to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        pass


class DataValidator(BaseValidator):
    """Main data validator."""
    
    def __init__(self):
        """Initialize data validator."""
        self.settings = get_settings()
        
        # Initialize specific validators
        self.rate_validator = RateValidator()
        self.time_series_validator = TimeSeriesValidator()
        self.feature_validator = FeatureValidator()
    
    def validate(self, data: pl.DataFrame, data_type: str = "general") -> Tuple[bool, List[str]]:
        """Validate data based on type.
        
        Args:
            data: DataFrame to validate
            data_type: Type of data being validated
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Basic validation
        if data.is_empty():
            errors.append("Data is empty")
            return False, errors
        
        # Check for minimum rows
        if len(data) < self.settings.features.min_observations:
            errors.append(f"Insufficient data: {len(data)} rows < {self.settings.features.min_observations} required")
        
        # Type-specific validation
        if data_type == "rates":
            is_valid, rate_errors = self.rate_validator.validate(data)
            errors.extend(rate_errors)
        
        elif data_type == "time_series":
            is_valid, ts_errors = self.time_series_validator.validate(data)
            errors.extend(ts_errors)
        
        elif data_type == "features":
            is_valid, feature_errors = self.feature_validator.validate(data)
            errors.extend(feature_errors)
        
        # General data quality checks
        quality_errors = self._check_data_quality(data)
        errors.extend(quality_errors)
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Validation failed with {len(errors)} errors")
        
        return is_valid, errors
    
    def _check_data_quality(self, data: pl.DataFrame) -> List[str]:
        """Check general data quality.
        
        Args:
            data: DataFrame to check
            
        Returns:
            List of quality issues
        """
        errors = []
        
        # Check for duplicate columns
        if len(set(data.columns)) < len(data.columns):
            errors.append("Duplicate column names detected")
        
        # Check for all-null columns
        for col in data.columns:
            if data[col].null_count() == len(data):
                errors.append(f"Column '{col}' contains only null values")
        
        # Check for constant columns
        numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
        for col in numeric_cols:
            if data[col].n_unique() == 1:
                errors.append(f"Column '{col}' has constant values")
        
        return errors


class RateValidator(BaseValidator):
    """Validator for exchange rate data."""
    
    def __init__(self):
        """Initialize rate validator."""
        self.min_rate = 100.0  # Minimum reasonable rate
        self.max_rate = 2000.0  # Maximum reasonable rate
        self.max_daily_change_pct = 20.0  # Maximum 20% daily change
    
    def validate(self, data: pl.DataFrame) -> Tuple[bool, List[str]]:
        """Validate exchange rate data.
        
        Args:
            data: DataFrame with rate data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Find rate columns
        rate_cols = [col for col in data.columns if 'rate' in col.lower()]
        
        if not rate_cols:
            errors.append("No rate columns found")
            return False, errors
        
        for col in rate_cols:
            # Check for negative rates
            if (data[col] < 0).any():
                errors.append(f"Negative values found in {col}")
            
            # Check for unreasonable values
            min_val = data[col].min()
            max_val = data[col].max()
            
            if min_val is not None and min_val < self.min_rate:
                errors.append(f"{col}: Minimum value {min_val} below threshold {self.min_rate}")
            
            if max_val is not None and max_val > self.max_rate:
                errors.append(f"{col}: Maximum value {max_val} above threshold {self.max_rate}")
            
            # Check for extreme daily changes
            daily_changes = data[col].pct_change() * 100
            max_change = daily_changes.abs().max()
            
            if max_change is not None and max_change > self.max_daily_change_pct:
                errors.append(f"{col}: Daily change {max_change:.1f}% exceeds threshold {self.max_daily_change_pct}%")
            
            # Check for rate spreads (if both official and parallel exist)
            if 'official' in col and any('parallel' in c for c in rate_cols):
                parallel_col = next((c for c in rate_cols if 'parallel' in c), None)
                if parallel_col:
                    spread_pct = ((data[parallel_col] - data[col]) / data[col] * 100).abs()
                    max_spread = spread_pct.max()
                    
                    if max_spread is not None and max_spread > 100:
                        errors.append(f"Excessive spread between official and parallel rates: {max_spread:.1f}%")
        
        return len(errors) == 0, errors


class TimeSeriesValidator(BaseValidator):
    """Validator for time series data."""
    
    def validate(self, data: pl.DataFrame) -> Tuple[bool, List[str]]:
        """Validate time series data.
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for date column
        date_cols = [col for col in data.columns if col.lower() in ['date', 'datetime', 'timestamp']]
        
        if not date_cols:
            errors.append("No date column found")
            return False, errors
        
        date_col = date_cols[0]
        
        # Check for duplicate dates
        if data[date_col].n_unique() < len(data):
            errors.append(f"Duplicate dates found in {date_col}")
        
        # Check for sorted dates
        sorted_dates = data[date_col].sort()
        if not data[date_col].equals(sorted_dates):
            errors.append("Dates are not sorted in ascending order")
        
        # Check for missing dates (assuming daily data)
        if len(data) > 1:
            date_range = pl.date_range(
                data[date_col].min(),
                data[date_col].max(),
                interval="1d"
            )
            
            missing_dates = set(date_range) - set(data[date_col])
            if missing_dates:
                errors.append(f"Missing {len(missing_dates)} dates in time series")
        
        # Check for future dates
        today = datetime.now().date()
        future_dates = data.filter(pl.col(date_col) > today)
        
        if len(future_dates) > 0:
            errors.append(f"Found {len(future_dates)} future dates")
        
        # Check for stale data
        if len(data) > 0:
            latest_date = data[date_col].max()
            days_old = (today - latest_date).days
            
            if days_old > 7:
                errors.append(f"Data is {days_old} days old (latest: {latest_date})")
        
        return len(errors) == 0, errors


class FeatureValidator(BaseValidator):
    """Validator for engineered features."""
    
    def __init__(self):
        """Initialize feature validator."""
        self.settings = get_settings()
    
    def validate(self, data: pl.DataFrame) -> Tuple[bool, List[str]]:
        """Validate engineered features.
        
        Args:
            data: DataFrame with features
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check for minimum required features
        min_features = 10
        if len(data.columns) < min_features:
            errors.append(f"Too few features: {len(data.columns)} < {min_features}")
        
        # Check for highly correlated features
        numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Float32, pl.Float64]]
        
        if len(numeric_cols) > 1:
            # Calculate correlation matrix
            corr_errors = self._check_correlations(data.select(numeric_cols))
            errors.extend(corr_errors)
        
        # Check for outliers
        outlier_errors = self._check_outliers(data)
        errors.extend(outlier_errors)
        
        # Check for data leakage
        leakage_errors = self._check_data_leakage(data)
        errors.extend(leakage_errors)
        
        return len(errors) == 0, errors
    
    def _check_correlations(self, data: pl.DataFrame) -> List[str]:
        """Check for highly correlated features.
        
        Args:
            data: DataFrame with numeric features
            
        Returns:
            List of correlation issues
        """
        errors = []
        
        # Convert to numpy for correlation calculation
        corr_matrix = np.corrcoef(data.to_numpy().T)
        
        # Find highly correlated pairs (> 0.95)
        high_corr_threshold = 0.95
        n_features = len(data.columns)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs(corr_matrix[i, j]) > high_corr_threshold:
                    errors.append(
                        f"High correlation ({corr_matrix[i, j]:.3f}) between "
                        f"'{data.columns[i]}' and '{data.columns[j]}'"
                    )
        
        return errors
    
    def _check_outliers(self, data: pl.DataFrame) -> List[str]:
        """Check for outliers in features.
        
        Args:
            data: DataFrame with features
            
        Returns:
            List of outlier issues
        """
        errors = []
        
        numeric_cols = [col for col in data.columns if data[col].dtype in [pl.Float32, pl.Float64]]
        
        for col in numeric_cols:
            # Calculate z-scores
            mean = data[col].mean()
            std = data[col].std()
            
            if std is not None and std > 0:
                z_scores = (data[col] - mean) / std
                outliers = z_scores.abs() > self.settings.features.outlier_threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_pct = outlier_count / len(data) * 100
                    if outlier_pct > 5:  # More than 5% outliers
                        errors.append(
                            f"Column '{col}' has {outlier_pct:.1f}% outliers "
                            f"(threshold: {self.settings.features.outlier_threshold} std)"
                        )
        
        return errors
    
    def _check_data_leakage(self, data: pl.DataFrame) -> List[str]:
        """Check for potential data leakage.
        
        Args:
            data: DataFrame with features
            
        Returns:
            List of leakage issues
        """
        errors = []
        
        # Check for future-looking features
        future_indicators = ['future', 'next', 'tomorrow', 'forward']
        
        for col in data.columns:
            if any(indicator in col.lower() for indicator in future_indicators):
                errors.append(f"Potential future data leakage in column '{col}'")
        
        # Check for target-like features
        target_indicators = ['target', 'label', 'y_true', 'actual']
        
        for col in data.columns:
            if any(indicator in col.lower() for indicator in target_indicators):
                errors.append(f"Potential target leakage in column '{col}'")
        
        return errors 