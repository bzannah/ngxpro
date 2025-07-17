"""Data validation and quality assurance for NG FX Predictor."""

import asyncio
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import polars as pl
from sqlalchemy import text

from ..config import get_settings
from ..data.database import DatabaseManager
from ..data.models import RawDataModel, FeatureModel
from ..utils.logging import get_logger
from ..utils.exceptions import ValidationError
from ..utils.metrics import MetricsManager

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    check_name: str
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    passed: bool


class DataValidator:
    """Comprehensive data validation and quality assurance system."""
    
    def __init__(self):
        """Initialize data validator."""
        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.metrics = MetricsManager()
        
        # Validation thresholds
        self.thresholds = {
            'missing_data_ratio': 0.1,  # 10% missing data threshold
            'outlier_z_score': 3.0,     # Z-score for outlier detection
            'data_freshness_hours': 48,  # Data freshness requirement
            'min_records_per_source': 10, # Minimum records per source
            'max_value_change_pct': 50,   # Maximum percentage change
        }
        
        logger.info("Data validator initialized")
    
    async def validate_raw_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[ValidationResult]:
        """Validate raw data quality.
        
        Args:
            start_date: Start date for validation
            end_date: End date for validation
            
        Returns:
            List of validation results
        """
        # Set default date range
        if end_date is None:
            end_date = datetime.now().date()
        if start_date is None:
            start_date = end_date - timedelta(days=30)
        
        logger.info(f"Validating raw data from {start_date} to {end_date}")
        
        results = []
        
        with self.metrics.timer("validation.raw_data"):
            # Load raw data
            raw_data = await self._load_raw_data(start_date, end_date)
            
            if raw_data.is_empty():
                results.append(ValidationResult(
                    check_name="data_availability",
                    severity=ValidationSeverity.CRITICAL,
                    message="No raw data available for validation",
                    details={"start_date": start_date, "end_date": end_date},
                    timestamp=datetime.now(),
                    passed=False
                ))
                return results
            
            # Run validation checks
            results.extend(await self._check_data_completeness(raw_data))
            results.extend(await self._check_data_freshness(raw_data))
            results.extend(await self._check_data_consistency(raw_data))
            results.extend(await self._check_data_quality(raw_data))
            results.extend(await self._detect_outliers(raw_data))
            
            # Log validation summary
            passed_checks = sum(1 for r in results if r.passed)
            total_checks = len(results)
            logger.info(f"Raw data validation completed: {passed_checks}/{total_checks} checks passed")
        
        return results
    
    async def validate_features(
        self,
        features: pl.DataFrame
    ) -> List[ValidationResult]:
        """Validate engineered features.
        
        Args:
            features: Features DataFrame
            
        Returns:
            List of validation results
        """
        logger.info("Validating engineered features")
        
        results = []
        
        with self.metrics.timer("validation.features"):
            if features.is_empty():
                results.append(ValidationResult(
                    check_name="feature_availability",
                    severity=ValidationSeverity.CRITICAL,
                    message="No features available for validation",
                    details={},
                    timestamp=datetime.now(),
                    passed=False
                ))
                return results
            
            # Run feature validation checks
            results.extend(self._check_feature_completeness(features))
            results.extend(self._check_feature_distributions(features))
            results.extend(self._check_feature_correlations(features))
            results.extend(self._check_feature_stability(features))
            results.extend(self._detect_feature_outliers(features))
            
            # Log validation summary
            passed_checks = sum(1 for r in results if r.passed)
            total_checks = len(results)
            logger.info(f"Feature validation completed: {passed_checks}/{total_checks} checks passed")
        
        return results
    
    async def _load_raw_data(self, start_date: date, end_date: date) -> pl.DataFrame:
        """Load raw data from database.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Raw data DataFrame
        """
        async with self.db_manager.get_session() as session:
            result = session.execute(
                text("""
                SELECT source, date, data, created_at
                FROM raw_data
                WHERE date BETWEEN :start_date AND :end_date
                ORDER BY date, source
                """),
                {'start_date': start_date, 'end_date': end_date}
            )
            
            records = result.fetchall()
        
        if not records:
            return pl.DataFrame()
        
        # Convert to structured format
        data = []
        for record in records:
            raw_data = record.data
            if isinstance(raw_data, str):
                import json
                raw_data = json.loads(raw_data)
            
            data.append({
                'source': record.source,
                'date': record.date,
                'data': raw_data,
                'created_at': record.created_at
            })
        
        return pl.DataFrame(data)
    
    async def _check_data_completeness(self, raw_data: pl.DataFrame) -> List[ValidationResult]:
        """Check data completeness across sources.
        
        Args:
            raw_data: Raw data DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check records per source
        source_counts = raw_data.group_by('source').agg(pl.count().alias('record_count'))
        
        for row in source_counts.iter_rows(named=True):
            source = row['source']
            count = row['record_count']
            
            if count < self.thresholds['min_records_per_source']:
                results.append(ValidationResult(
                    check_name="data_completeness",
                    severity=ValidationSeverity.WARNING,
                    message=f"Source {source} has insufficient records",
                    details={"source": source, "count": count, "threshold": self.thresholds['min_records_per_source']},
                    timestamp=datetime.now(),
                    passed=False
                ))
            else:
                results.append(ValidationResult(
                    check_name="data_completeness",
                    severity=ValidationSeverity.INFO,
                    message=f"Source {source} has sufficient records",
                    details={"source": source, "count": count},
                    timestamp=datetime.now(),
                    passed=True
                ))
        
        return results
    
    async def _check_data_freshness(self, raw_data: pl.DataFrame) -> List[ValidationResult]:
        """Check data freshness.
        
        Args:
            raw_data: Raw data DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check latest data per source
        latest_data = raw_data.group_by('source').agg(
            pl.max('date').alias('latest_date'),
            pl.max('created_at').alias('latest_created')
        )
        
        current_time = datetime.now()
        freshness_threshold = current_time - timedelta(hours=self.thresholds['data_freshness_hours'])
        
        for row in latest_data.iter_rows(named=True):
            source = row['source']
            latest_date = row['latest_date']
            latest_created = row['latest_created']
            
            # Convert created_at to datetime if it's a string
            if isinstance(latest_created, str):
                from datetime import datetime
                latest_created = datetime.fromisoformat(latest_created.replace('Z', '+00:00'))
            
            if latest_created < freshness_threshold:
                results.append(ValidationResult(
                    check_name="data_freshness",
                    severity=ValidationSeverity.WARNING,
                    message=f"Data from {source} is stale",
                    details={
                        "source": source,
                        "latest_date": latest_date,
                        "latest_created": latest_created,
                        "hours_old": (current_time - latest_created).total_seconds() / 3600
                    },
                    timestamp=datetime.now(),
                    passed=False
                ))
            else:
                results.append(ValidationResult(
                    check_name="data_freshness",
                    severity=ValidationSeverity.INFO,
                    message=f"Data from {source} is fresh",
                    details={
                        "source": source,
                        "latest_date": latest_date,
                        "latest_created": latest_created
                    },
                    timestamp=datetime.now(),
                    passed=True
                ))
        
        return results
    
    async def _check_data_consistency(self, raw_data: pl.DataFrame) -> List[ValidationResult]:
        """Check data consistency.
        
        Args:
            raw_data: Raw data DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check for duplicate records
        duplicates = raw_data.group_by(['source', 'date']).agg(
            pl.count().alias('count')
        ).filter(pl.col('count') > 1)
        
        if len(duplicates) > 0:
            results.append(ValidationResult(
                check_name="data_consistency",
                severity=ValidationSeverity.WARNING,
                message="Duplicate records found",
                details={"duplicates": duplicates.to_dicts()},
                timestamp=datetime.now(),
                passed=False
            ))
        else:
            results.append(ValidationResult(
                check_name="data_consistency",
                severity=ValidationSeverity.INFO,
                message="No duplicate records found",
                details={},
                timestamp=datetime.now(),
                passed=True
            ))
        
        # Check date continuity (for daily data)
        for source in raw_data['source'].unique():
            source_data = raw_data.filter(pl.col('source') == source).sort('date')
            dates = source_data['date'].to_list()
            
            if len(dates) > 1:
                # Check for gaps in dates
                gaps = []
                for i in range(1, len(dates)):
                    gap_days = (dates[i] - dates[i-1]).days
                    if gap_days > 1:
                        gaps.append({
                            'start_date': dates[i-1],
                            'end_date': dates[i],
                            'gap_days': gap_days
                        })
                
                if gaps:
                    results.append(ValidationResult(
                        check_name="data_continuity",
                        severity=ValidationSeverity.WARNING,
                        message=f"Data gaps found in {source}",
                        details={"source": source, "gaps": gaps},
                        timestamp=datetime.now(),
                        passed=False
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="data_continuity",
                        severity=ValidationSeverity.INFO,
                        message=f"No data gaps in {source}",
                        details={"source": source},
                        timestamp=datetime.now(),
                        passed=True
                    ))
        
        return results
    
    async def _check_data_quality(self, raw_data: pl.DataFrame) -> List[ValidationResult]:
        """Check data quality metrics.
        
        Args:
            raw_data: Raw data DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check for null/empty data
        for source in raw_data['source'].unique():
            source_data = raw_data.filter(pl.col('source') == source)
            
            # Check for null data fields
            null_data_count = source_data.filter(pl.col('data').is_null()).height
            total_count = source_data.height
            
            if null_data_count > 0:
                null_ratio = null_data_count / total_count
                results.append(ValidationResult(
                    check_name="data_quality",
                    severity=ValidationSeverity.WARNING if null_ratio < 0.1 else ValidationSeverity.ERROR,
                    message=f"Null data found in {source}",
                    details={
                        "source": source,
                        "null_count": null_data_count,
                        "total_count": total_count,
                        "null_ratio": null_ratio
                    },
                    timestamp=datetime.now(),
                    passed=False
                ))
            else:
                results.append(ValidationResult(
                    check_name="data_quality",
                    severity=ValidationSeverity.INFO,
                    message=f"No null data in {source}",
                    details={"source": source},
                    timestamp=datetime.now(),
                    passed=True
                ))
        
        return results
    
    async def _detect_outliers(self, raw_data: pl.DataFrame) -> List[ValidationResult]:
        """Detect outliers in raw data.
        
        Args:
            raw_data: Raw data DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        # Define expected ranges for different data types
        expected_ranges = {
            'cbn_rates': {'min': 100, 'max': 2000},  # NGN/USD rates
            'eia_brent': {'min': 20, 'max': 200},    # Oil prices
            'worldbank_reserves': {'min': 1e9, 'max': 1e12},  # USD reserves
            'dmo_debt': {'min': 1e12, 'max': 1e15},  # NGN debt
            'news_sentiment': {'min': -1, 'max': 1}  # Sentiment scores
        }
        
        for source in raw_data['source'].unique():
            source_data = raw_data.filter(pl.col('source') == source)
            
            # Extract numeric values from data
            numeric_values = []
            for row in source_data.iter_rows(named=True):
                data = row['data']
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (int, float)) and not np.isnan(value):
                            numeric_values.append(value)
            
            if numeric_values:
                # Statistical outlier detection
                values = np.array(numeric_values)
                z_scores = np.abs((values - np.mean(values)) / np.std(values))
                outliers = values[z_scores > self.thresholds['outlier_z_score']]
                
                if len(outliers) > 0:
                    results.append(ValidationResult(
                        check_name="outlier_detection",
                        severity=ValidationSeverity.WARNING,
                        message=f"Statistical outliers detected in {source}",
                        details={
                            "source": source,
                            "outlier_count": len(outliers),
                            "outlier_values": outliers.tolist()[:10],  # Limit to first 10
                            "z_score_threshold": self.thresholds['outlier_z_score']
                        },
                        timestamp=datetime.now(),
                        passed=False
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="outlier_detection",
                        severity=ValidationSeverity.INFO,
                        message=f"No statistical outliers in {source}",
                        details={"source": source},
                        timestamp=datetime.now(),
                        passed=True
                    ))
                
                # Domain-specific outlier detection
                if source in expected_ranges:
                    range_info = expected_ranges[source]
                    out_of_range = values[(values < range_info['min']) | (values > range_info['max'])]
                    
                    if len(out_of_range) > 0:
                        results.append(ValidationResult(
                            check_name="range_validation",
                            severity=ValidationSeverity.ERROR,
                            message=f"Values out of expected range in {source}",
                            details={
                                "source": source,
                                "out_of_range_count": len(out_of_range),
                                "expected_range": range_info,
                                "out_of_range_values": out_of_range.tolist()[:5]
                            },
                            timestamp=datetime.now(),
                            passed=False
                        ))
                    else:
                        results.append(ValidationResult(
                            check_name="range_validation",
                            severity=ValidationSeverity.INFO,
                            message=f"All values within expected range for {source}",
                            details={"source": source, "expected_range": range_info},
                            timestamp=datetime.now(),
                            passed=True
                        ))
        
        return results
    
    def _check_feature_completeness(self, features: pl.DataFrame) -> List[ValidationResult]:
        """Check feature completeness.
        
        Args:
            features: Features DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check missing values per feature
        for col in features.columns:
            if col != 'date':
                null_count = features[col].null_count()
                total_count = features.height
                null_ratio = null_count / total_count
                
                if null_ratio > self.thresholds['missing_data_ratio']:
                    results.append(ValidationResult(
                        check_name="feature_completeness",
                        severity=ValidationSeverity.WARNING,
                        message=f"High missing data ratio in feature {col}",
                        details={
                            "feature": col,
                            "null_count": null_count,
                            "total_count": total_count,
                            "null_ratio": null_ratio,
                            "threshold": self.thresholds['missing_data_ratio']
                        },
                        timestamp=datetime.now(),
                        passed=False
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="feature_completeness",
                        severity=ValidationSeverity.INFO,
                        message=f"Feature {col} has acceptable missing data ratio",
                        details={"feature": col, "null_ratio": null_ratio},
                        timestamp=datetime.now(),
                        passed=True
                    ))
        
        return results
    
    def _check_feature_distributions(self, features: pl.DataFrame) -> List[ValidationResult]:
        """Check feature distributions for anomalies.
        
        Args:
            features: Features DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        numeric_features = [
            col for col in features.columns
            if col != 'date' and features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        for feature in numeric_features:
            values = features[feature].drop_nulls()
            
            if len(values) > 10:  # Need sufficient data for distribution check
                # Check for constant values
                if values.std() == 0:
                    results.append(ValidationResult(
                        check_name="feature_distribution",
                        severity=ValidationSeverity.WARNING,
                        message=f"Feature {feature} has constant values",
                        details={"feature": feature, "constant_value": values[0]},
                        timestamp=datetime.now(),
                        passed=False
                    ))
                else:
                    # Check for extreme skewness
                    skewness = values.skew()
                    if abs(skewness) > 2:  # Highly skewed
                        results.append(ValidationResult(
                            check_name="feature_distribution",
                            severity=ValidationSeverity.INFO,
                            message=f"Feature {feature} is highly skewed",
                            details={"feature": feature, "skewness": skewness},
                            timestamp=datetime.now(),
                            passed=True
                        ))
                    else:
                        results.append(ValidationResult(
                            check_name="feature_distribution",
                            severity=ValidationSeverity.INFO,
                            message=f"Feature {feature} has normal distribution",
                            details={"feature": feature, "skewness": skewness},
                            timestamp=datetime.now(),
                            passed=True
                        ))
        
        return results
    
    def _check_feature_correlations(self, features: pl.DataFrame) -> List[ValidationResult]:
        """Check feature correlations for multicollinearity.
        
        Args:
            features: Features DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        numeric_features = [
            col for col in features.columns
            if col != 'date' and features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        if len(numeric_features) > 1:
            # Calculate correlation matrix
            correlation_matrix = features.select(numeric_features).corr()
            
            # Check for high correlations (multicollinearity)
            high_correlations = []
            for i in range(len(numeric_features)):
                for j in range(i + 1, len(numeric_features)):
                    corr_value = correlation_matrix[i, j]
                    if abs(corr_value) > 0.95:  # High correlation threshold
                        high_correlations.append({
                            'feature1': numeric_features[i],
                            'feature2': numeric_features[j],
                            'correlation': corr_value
                        })
            
            if high_correlations:
                results.append(ValidationResult(
                    check_name="feature_correlation",
                    severity=ValidationSeverity.WARNING,
                    message="High feature correlations detected",
                    details={"high_correlations": high_correlations},
                    timestamp=datetime.now(),
                    passed=False
                ))
            else:
                results.append(ValidationResult(
                    check_name="feature_correlation",
                    severity=ValidationSeverity.INFO,
                    message="No high feature correlations detected",
                    details={},
                    timestamp=datetime.now(),
                    passed=True
                ))
        
        return results
    
    def _check_feature_stability(self, features: pl.DataFrame) -> List[ValidationResult]:
        """Check feature stability over time.
        
        Args:
            features: Features DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        if features.height < 2:
            return results
        
        # Sort by date
        features = features.sort('date')
        
        numeric_features = [
            col for col in features.columns
            if col != 'date' and features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        for feature in numeric_features:
            # Calculate rolling statistics
            rolling_mean = features[feature].rolling_mean(7)
            rolling_std = features[feature].rolling_std(7)
            
            # Check for sudden changes
            pct_changes = features[feature].pct_change()
            extreme_changes = pct_changes.filter(pl.col(feature).abs() > self.thresholds['max_value_change_pct'] / 100)
            
            if len(extreme_changes) > 0:
                results.append(ValidationResult(
                    check_name="feature_stability",
                    severity=ValidationSeverity.WARNING,
                    message=f"Extreme changes detected in feature {feature}",
                    details={
                        "feature": feature,
                        "extreme_changes": len(extreme_changes),
                        "max_change_pct": self.thresholds['max_value_change_pct']
                    },
                    timestamp=datetime.now(),
                    passed=False
                ))
            else:
                results.append(ValidationResult(
                    check_name="feature_stability",
                    severity=ValidationSeverity.INFO,
                    message=f"Feature {feature} is stable",
                    details={"feature": feature},
                    timestamp=datetime.now(),
                    passed=True
                ))
        
        return results
    
    def _detect_feature_outliers(self, features: pl.DataFrame) -> List[ValidationResult]:
        """Detect outliers in engineered features.
        
        Args:
            features: Features DataFrame
            
        Returns:
            List of validation results
        """
        results = []
        
        numeric_features = [
            col for col in features.columns
            if col != 'date' and features[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
        ]
        
        for feature in numeric_features:
            values = features[feature].drop_nulls()
            
            if len(values) > 10:
                # Statistical outlier detection using IQR method
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers_mask = (values < lower_bound) | (values > upper_bound)
                outliers = values.filter(outliers_mask)
                
                if len(outliers) > 0:
                    results.append(ValidationResult(
                        check_name="feature_outliers",
                        severity=ValidationSeverity.INFO,
                        message=f"Outliers detected in feature {feature}",
                        details={
                            "feature": feature,
                            "outlier_count": len(outliers),
                            "total_count": len(values),
                            "outlier_ratio": len(outliers) / len(values),
                            "bounds": {"lower": lower_bound, "upper": upper_bound}
                        },
                        timestamp=datetime.now(),
                        passed=len(outliers) / len(values) < 0.05  # Less than 5% outliers is acceptable
                    ))
                else:
                    results.append(ValidationResult(
                        check_name="feature_outliers",
                        severity=ValidationSeverity.INFO,
                        message=f"No outliers detected in feature {feature}",
                        details={"feature": feature},
                        timestamp=datetime.now(),
                        passed=True
                    ))
        
        return results
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate a comprehensive validation report.
        
        Args:
            results: List of validation results
            
        Returns:
            Validation report dictionary
        """
        # Categorize results by severity
        severity_counts = {
            ValidationSeverity.INFO: 0,
            ValidationSeverity.WARNING: 0,
            ValidationSeverity.ERROR: 0,
            ValidationSeverity.CRITICAL: 0
        }
        
        passed_count = 0
        failed_count = 0
        
        for result in results:
            severity_counts[result.severity] += 1
            if result.passed:
                passed_count += 1
            else:
                failed_count += 1
        
        # Calculate overall quality score
        total_checks = len(results)
        quality_score = passed_count / total_checks if total_checks > 0 else 0
        
        # Determine overall status
        if severity_counts[ValidationSeverity.CRITICAL] > 0:
            status = "CRITICAL"
        elif severity_counts[ValidationSeverity.ERROR] > 0:
            status = "ERROR"
        elif severity_counts[ValidationSeverity.WARNING] > 0:
            status = "WARNING"
        else:
            status = "PASSED"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": status,
            "quality_score": quality_score,
            "total_checks": total_checks,
            "passed_checks": passed_count,
            "failed_checks": failed_count,
            "severity_counts": {
                "info": severity_counts[ValidationSeverity.INFO],
                "warning": severity_counts[ValidationSeverity.WARNING],
                "error": severity_counts[ValidationSeverity.ERROR],
                "critical": severity_counts[ValidationSeverity.CRITICAL]
            },
            "detailed_results": [
                {
                    "check_name": result.check_name,
                    "severity": result.severity.value,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat(),
                    "passed": result.passed
                }
                for result in results
            ]
        } 