"""Data validation utilities for NG FX Predictor."""

import re
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from pydantic import BaseModel, Field, validator

from ..config import get_settings
from ..utils.logging import get_logger
from .exceptions import ValidationError, DataQualityError

logger = get_logger(__name__)


class ValidationResult(BaseModel):
    """Validation result model."""
    
    is_valid: bool = Field(description="Whether validation passed")
    score: float = Field(ge=0.0, le=1.0, description="Validation score (0-1)")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional details")
    
    @validator('score')
    def score_must_be_valid(cls, v):
        """Validate score range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Score must be between 0.0 and 1.0')
        return v


class DataValidator:
    """Data validation and quality checking."""
    
    def __init__(self):
        """Initialize data validator."""
        self.settings = get_settings()
        logger.info("Data validator initialized")
    
    def validate_dataframe(self, df: pl.DataFrame, schema: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate a DataFrame.
        
        Args:
            df: DataFrame to validate
            schema: Optional schema definition
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        details = {}
        
        # Basic structure validation
        if df.is_empty():
            errors.append("DataFrame is empty")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                errors=errors,
                details=details
            )
        
        # Check for basic requirements
        if df.height == 0:
            errors.append("DataFrame has no rows")
        
        if df.width == 0:
            errors.append("DataFrame has no columns")
        
        # Check for null values
        null_counts = df.null_count()
        total_cells = df.height * df.width
        total_nulls = null_counts.sum_horizontal().item()
        
        if total_nulls > 0:
            null_percentage = (total_nulls / total_cells) * 100
            details['null_percentage'] = null_percentage
            
            if null_percentage > 20:  # More than 20% null values
                errors.append(f"High null percentage: {null_percentage:.1f}%")
            elif null_percentage > 10:  # More than 10% null values
                warnings.append(f"Moderate null percentage: {null_percentage:.1f}%")
        
        # Check for duplicate rows
        duplicate_count = df.height - df.unique().height
        if duplicate_count > 0:
            duplicate_percentage = (duplicate_count / df.height) * 100
            details['duplicate_percentage'] = duplicate_percentage
            
            if duplicate_percentage > 10:
                errors.append(f"High duplicate percentage: {duplicate_percentage:.1f}%")
            elif duplicate_percentage > 5:
                warnings.append(f"Moderate duplicate percentage: {duplicate_percentage:.1f}%")
        
        # Validate schema if provided
        if schema:
            schema_validation = self._validate_schema(df, schema)
            errors.extend(schema_validation.get('errors', []))
            warnings.extend(schema_validation.get('warnings', []))
            details['schema_validation'] = schema_validation
        
        # Calculate overall score
        score = self._calculate_validation_score(df, errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def validate_time_series(self, df: pl.DataFrame, date_column: str = "date") -> ValidationResult:
        """Validate time series data.
        
        Args:
            df: DataFrame with time series data
            date_column: Name of the date column
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        details = {}
        
        # Check if date column exists
        if date_column not in df.columns:
            errors.append(f"Date column '{date_column}' not found")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                errors=errors,
                details=details
            )
        
        # Check date column type
        date_dtype = df[date_column].dtype
        if date_dtype not in [pl.Date, pl.Datetime]:
            errors.append(f"Date column '{date_column}' has invalid type: {date_dtype}")
        
        # Check for missing dates
        if df[date_column].null_count() > 0:
            errors.append(f"Date column '{date_column}' contains null values")
        
        # Check date range
        try:
            min_date = df[date_column].min()
            max_date = df[date_column].max()
            
            if min_date and max_date:
                date_range = (max_date - min_date).days if hasattr(max_date - min_date, 'days') else 0
                details['date_range_days'] = date_range
                details['min_date'] = str(min_date)
                details['max_date'] = str(max_date)
                
                # Check if data is too old
                if isinstance(max_date, date):
                    days_old = (datetime.now().date() - max_date).days
                    if days_old > self.settings.data_validation.staleness_threshold_days:
                        errors.append(f"Data is {days_old} days old (threshold: {self.settings.data_validation.staleness_threshold_days})")
        except Exception as e:
            errors.append(f"Error analyzing date range: {str(e)}")
        
        # Check for duplicate dates
        duplicate_dates = df.height - df.select(date_column).unique().height
        if duplicate_dates > 0:
            warnings.append(f"Found {duplicate_dates} duplicate dates")
            details['duplicate_dates'] = duplicate_dates
        
        # Check for gaps in time series
        if df.height > 1:
            sorted_df = df.sort(date_column)
            # This is a simplified gap detection - could be enhanced
            details['total_records'] = df.height
        
        # Calculate score
        score = self._calculate_validation_score(df, errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def validate_numeric_data(self, df: pl.DataFrame, numeric_columns: List[str]) -> ValidationResult:
        """Validate numeric data columns.
        
        Args:
            df: DataFrame to validate
            numeric_columns: List of numeric column names
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        details = {}
        
        for col in numeric_columns:
            if col not in df.columns:
                errors.append(f"Numeric column '{col}' not found")
                continue
            
            col_details = {}
            
            # Check data type
            if not df[col].dtype.is_numeric():
                errors.append(f"Column '{col}' is not numeric: {df[col].dtype}")
                continue
            
            # Get basic statistics
            try:
                col_data = df[col].drop_nulls()
                if col_data.len() == 0:
                    errors.append(f"Column '{col}' contains only null values")
                    continue
                
                col_details['count'] = col_data.len()
                col_details['null_count'] = df[col].null_count()
                col_details['mean'] = col_data.mean()
                col_details['std'] = col_data.std()
                col_details['min'] = col_data.min()
                col_details['max'] = col_data.max()
                
                # Check for outliers using IQR method
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = col_data.filter((col_data < lower_bound) | (col_data > upper_bound))
                outlier_count = outliers.len()
                
                if outlier_count > 0:
                    outlier_percentage = (outlier_count / col_data.len()) * 100
                    col_details['outlier_percentage'] = outlier_percentage
                    
                    if outlier_percentage > self.settings.data_validation.quality_max_outliers * 100:
                        errors.append(f"Column '{col}' has {outlier_percentage:.1f}% outliers")
                    elif outlier_percentage > 5:
                        warnings.append(f"Column '{col}' has {outlier_percentage:.1f}% outliers")
                
                # Check for infinite values
                if col_data.is_infinite().any():
                    errors.append(f"Column '{col}' contains infinite values")
                
                # Check for extremely large values
                if col_details['std'] > 0:
                    z_scores = (col_data - col_details['mean']) / col_details['std']
                    extreme_values = z_scores.filter(z_scores.abs() > 5).len()
                    if extreme_values > 0:
                        warnings.append(f"Column '{col}' has {extreme_values} extreme values (|z-score| > 5)")
                
            except Exception as e:
                errors.append(f"Error analyzing column '{col}': {str(e)}")
            
            details[col] = col_details
        
        # Calculate score
        score = self._calculate_validation_score(df, errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def validate_categorical_data(self, df: pl.DataFrame, categorical_columns: List[str]) -> ValidationResult:
        """Validate categorical data columns.
        
        Args:
            df: DataFrame to validate
            categorical_columns: List of categorical column names
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        details = {}
        
        for col in categorical_columns:
            if col not in df.columns:
                errors.append(f"Categorical column '{col}' not found")
                continue
            
            col_details = {}
            
            # Get value counts
            try:
                value_counts = df[col].value_counts()
                col_details['unique_values'] = value_counts.height
                col_details['null_count'] = df[col].null_count()
                
                # Check for too many unique values (might indicate a problem)
                if col_details['unique_values'] > df.height * 0.5:
                    warnings.append(f"Column '{col}' has {col_details['unique_values']} unique values ({col_details['unique_values']/df.height*100:.1f}% of total)")
                
                # Check for imbalanced categories
                if col_details['unique_values'] > 1:
                    value_counts_df = value_counts.sort('count', descending=True)
                    max_count = value_counts_df['count'].max()
                    min_count = value_counts_df['count'].min()
                    
                    if max_count / min_count > 10:  # Most frequent is 10x more than least frequent
                        warnings.append(f"Column '{col}' has imbalanced categories")
                
                # Store top categories
                col_details['top_categories'] = value_counts.head(5).to_dict(as_series=False)
                
            except Exception as e:
                errors.append(f"Error analyzing categorical column '{col}': {str(e)}")
            
            details[col] = col_details
        
        # Calculate score
        score = self._calculate_validation_score(df, errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=score,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def validate_data_quality(self, df: pl.DataFrame) -> ValidationResult:
        """Comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        details = {}
        
        # Basic structure validation
        basic_validation = self.validate_dataframe(df)
        errors.extend(basic_validation.errors)
        warnings.extend(basic_validation.warnings)
        details['basic_validation'] = basic_validation.details
        
        # Completeness check
        completeness_score = self._calculate_completeness_score(df)
        details['completeness_score'] = completeness_score
        
        if completeness_score < self.settings.data_validation.quality_min_completeness:
            errors.append(f"Data completeness {completeness_score:.1%} is below threshold {self.settings.data_validation.quality_min_completeness:.1%}")
        
        # Consistency checks
        consistency_issues = self._check_consistency(df)
        if consistency_issues:
            warnings.extend(consistency_issues)
        
        # Timeliness check (if date column exists)
        if 'date' in df.columns:
            time_validation = self.validate_time_series(df)
            errors.extend(time_validation.errors)
            warnings.extend(time_validation.warnings)
            details['time_validation'] = time_validation.details
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(df, errors, warnings)
        details['quality_score'] = quality_score
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            score=quality_score,
            errors=errors,
            warnings=warnings,
            details=details
        )
    
    def _validate_schema(self, df: pl.DataFrame, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DataFrame against schema.
        
        Args:
            df: DataFrame to validate
            schema: Schema definition
            
        Returns:
            Validation results
        """
        errors = []
        warnings = []
        
        # Check required columns
        required_columns = schema.get('required_columns', [])
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Required column '{col}' is missing")
        
        # Check column types
        column_types = schema.get('column_types', {})
        for col, expected_type in column_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if actual_type != expected_type:
                    errors.append(f"Column '{col}' has type {actual_type}, expected {expected_type}")
        
        # Check column constraints
        constraints = schema.get('constraints', {})
        for col, constraint in constraints.items():
            if col in df.columns:
                if 'min_value' in constraint:
                    min_val = df[col].min()
                    if min_val < constraint['min_value']:
                        errors.append(f"Column '{col}' has value {min_val} below minimum {constraint['min_value']}")
                
                if 'max_value' in constraint:
                    max_val = df[col].max()
                    if max_val > constraint['max_value']:
                        errors.append(f"Column '{col}' has value {max_val} above maximum {constraint['max_value']}")
        
        return {
            'errors': errors,
            'warnings': warnings,
        }
    
    def _calculate_validation_score(self, df: pl.DataFrame, errors: List[str], warnings: List[str]) -> float:
        """Calculate validation score.
        
        Args:
            df: DataFrame
            errors: List of errors
            warnings: List of warnings
            
        Returns:
            Validation score (0-1)
        """
        if not df or df.is_empty():
            return 0.0
        
        # Start with perfect score
        score = 1.0
        
        # Deduct for errors (more severe)
        score -= len(errors) * 0.2
        
        # Deduct for warnings (less severe)
        score -= len(warnings) * 0.1
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, score))
    
    def _calculate_completeness_score(self, df: pl.DataFrame) -> float:
        """Calculate data completeness score.
        
        Args:
            df: DataFrame
            
        Returns:
            Completeness score (0-1)
        """
        if df.is_empty():
            return 0.0
        
        total_cells = df.height * df.width
        null_cells = df.null_count().sum_horizontal().item()
        
        return 1.0 - (null_cells / total_cells)
    
    def _calculate_quality_score(self, df: pl.DataFrame, errors: List[str], warnings: List[str]) -> float:
        """Calculate overall data quality score.
        
        Args:
            df: DataFrame
            errors: List of errors
            warnings: List of warnings
            
        Returns:
            Quality score (0-1)
        """
        if df.is_empty():
            return 0.0
        
        # Base score from completeness
        completeness_score = self._calculate_completeness_score(df)
        
        # Validation score
        validation_score = self._calculate_validation_score(df, errors, warnings)
        
        # Combined score (weighted average)
        quality_score = (completeness_score * 0.4) + (validation_score * 0.6)
        
        return quality_score
    
    def _check_consistency(self, df: pl.DataFrame) -> List[str]:
        """Check data consistency issues.
        
        Args:
            df: DataFrame
            
        Returns:
            List of consistency issues
        """
        issues = []
        
        # Check for mixed data types in object columns
        for col in df.columns:
            if df[col].dtype == pl.Object:
                issues.append(f"Column '{col}' has mixed data types")
        
        # Check for unrealistic values (example: negative prices)
        if 'price' in df.columns:
            try:
                if df['price'].min() < 0:
                    issues.append("Found negative prices")
            except:
                pass
        
        return issues


def validate_exchange_rate_data(df: pl.DataFrame) -> ValidationResult:
    """Validate exchange rate specific data.
    
    Args:
        df: DataFrame with exchange rate data
        
    Returns:
        Validation result
    """
    validator = DataValidator()
    
    # Define exchange rate schema
    schema = {
        'required_columns': ['date', 'usd_ngn'],
        'column_types': {
            'date': pl.Date,
            'usd_ngn': pl.Float64,
        },
        'constraints': {
            'usd_ngn': {
                'min_value': 0.0,
                'max_value': 10000.0,  # Reasonable upper bound
            }
        }
    }
    
    # Validate with schema
    result = validator.validate_dataframe(df, schema)
    
    # Additional exchange rate specific checks
    if 'usd_ngn' in df.columns:
        # Check for unrealistic exchange rates
        try:
            rate_data = df['usd_ngn'].drop_nulls()
            if rate_data.len() > 0:
                mean_rate = rate_data.mean()
                std_rate = rate_data.std()
                
                # Check for rates that are too far from normal range
                if mean_rate < 100 or mean_rate > 5000:
                    result.warnings.append(f"Exchange rate mean ({mean_rate:.2f}) is outside normal range")
                
                # Check for excessive volatility
                if std_rate > mean_rate * 0.5:
                    result.warnings.append(f"Exchange rate has high volatility (std: {std_rate:.2f})")
        except:
            pass
    
    return result 