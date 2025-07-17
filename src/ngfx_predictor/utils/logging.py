"""Structured logging for NG FX Predictor."""

import logging
import sys
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from ..config import get_settings


def configure_logging(log_level: str = "INFO", json_format: bool = True) -> None:
    """Configure structured logging.
    
    Args:
        log_level: Logging level
        json_format: Whether to use JSON formatting
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


@lru_cache(maxsize=None)
def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return structlog.get_logger(name)


class RequestLogger:
    """Request logging context manager."""
    
    def __init__(self, request_id: str, logger: structlog.stdlib.BoundLogger):
        """Initialize request logger.
        
        Args:
            request_id: Unique request identifier
            logger: Base logger
        """
        self.request_id = request_id
        self.logger = logger.bind(request_id=request_id)
        self.start_time = datetime.now()
    
    def __enter__(self):
        """Enter context manager."""
        self.logger.info("Request started", request_id=self.request_id)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is not None:
            self.logger.error(
                "Request failed",
                request_id=self.request_id,
                duration_seconds=duration,
                exception=str(exc_val),
                exception_type=exc_type.__name__,
            )
        else:
            self.logger.info(
                "Request completed",
                request_id=self.request_id,
                duration_seconds=duration,
            )


class MetricsLogger:
    """Metrics logging utility."""
    
    def __init__(self, logger: structlog.stdlib.BoundLogger):
        """Initialize metrics logger.
        
        Args:
            logger: Base logger
        """
        self.logger = logger
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "count",
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Log a metric.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            tags: Optional tags
        """
        self.logger.info(
            "Metric recorded",
            metric_name=metric_name,
            metric_value=value,
            metric_unit=unit,
            metric_tags=tags or {},
            timestamp=datetime.now().isoformat(),
        )
    
    def log_performance(
        self,
        operation: str,
        duration_seconds: float,
        success: bool = True,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log performance metrics.
        
        Args:
            operation: Operation name
            duration_seconds: Duration in seconds
            success: Whether operation succeeded
            additional_info: Additional information
        """
        self.logger.info(
            "Performance metric",
            operation=operation,
            duration_seconds=duration_seconds,
            success=success,
            additional_info=additional_info or {},
            timestamp=datetime.now().isoformat(),
        )
    
    def log_business_metric(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> None:
        """Log business metrics.
        
        Args:
            event_type: Type of business event
            event_data: Event data
            user_id: Optional user identifier
        """
        self.logger.info(
            "Business metric",
            event_type=event_type,
            event_data=event_data,
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
        )


class AuditLogger:
    """Audit logging utility."""
    
    def __init__(self, logger: structlog.stdlib.BoundLogger):
        """Initialize audit logger.
        
        Args:
            logger: Base logger
        """
        self.logger = logger
    
    def log_data_access(
        self,
        table_name: str,
        operation: str,
        record_count: int,
        user_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log data access events.
        
        Args:
            table_name: Database table name
            operation: Operation type (SELECT, INSERT, UPDATE, DELETE)
            record_count: Number of records affected
            user_id: Optional user identifier
            filters: Optional filters applied
        """
        self.logger.info(
            "Data access",
            table_name=table_name,
            operation=operation,
            record_count=record_count,
            user_id=user_id,
            filters=filters or {},
            timestamp=datetime.now().isoformat(),
        )
    
    def log_model_operation(
        self,
        operation: str,
        model_name: str,
        model_version: str,
        user_id: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log model operations.
        
        Args:
            operation: Operation type (TRAIN, PREDICT, PROMOTE, ARCHIVE)
            model_name: Name of the model
            model_version: Version of the model
            user_id: Optional user identifier
            additional_info: Additional information
        """
        self.logger.info(
            "Model operation",
            operation=operation,
            model_name=model_name,
            model_version=model_version,
            user_id=user_id,
            additional_info=additional_info or {},
            timestamp=datetime.now().isoformat(),
        )
    
    def log_security_event(
        self,
        event_type: str,
        event_description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        """Log security events.
        
        Args:
            event_type: Type of security event
            event_description: Description of the event
            user_id: Optional user identifier
            ip_address: Optional IP address
            user_agent: Optional user agent
        """
        self.logger.warning(
            "Security event",
            event_type=event_type,
            event_description=event_description,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now().isoformat(),
        )


class DataQualityLogger:
    """Data quality logging utility."""
    
    def __init__(self, logger: structlog.stdlib.BoundLogger):
        """Initialize data quality logger.
        
        Args:
            logger: Base logger
        """
        self.logger = logger
    
    def log_data_quality_check(
        self,
        source_name: str,
        check_type: str,
        quality_score: float,
        passed: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log data quality checks.
        
        Args:
            source_name: Name of the data source
            check_type: Type of quality check
            quality_score: Quality score (0-1)
            passed: Whether check passed
            details: Optional details
        """
        log_level = "info" if passed else "warning"
        
        getattr(self.logger, log_level)(
            "Data quality check",
            source_name=source_name,
            check_type=check_type,
            quality_score=quality_score,
            passed=passed,
            details=details or {},
            timestamp=datetime.now().isoformat(),
        )
    
    def log_data_anomaly(
        self,
        source_name: str,
        anomaly_type: str,
        anomaly_description: str,
        severity: str = "medium",
        affected_records: Optional[int] = None,
    ) -> None:
        """Log data anomalies.
        
        Args:
            source_name: Name of the data source
            anomaly_type: Type of anomaly
            anomaly_description: Description of the anomaly
            severity: Severity level (low, medium, high)
            affected_records: Number of affected records
        """
        log_level = "warning" if severity in ["medium", "high"] else "info"
        
        getattr(self.logger, log_level)(
            "Data anomaly detected",
            source_name=source_name,
            anomaly_type=anomaly_type,
            anomaly_description=anomaly_description,
            severity=severity,
            affected_records=affected_records,
            timestamp=datetime.now().isoformat(),
        )


def mask_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive data in log entries.
    
    Args:
        data: Data dictionary
        
    Returns:
        Data dictionary with sensitive fields masked
    """
    sensitive_fields = {
        "password", "secret", "token", "key", "credential",
        "authorization", "cookie", "session", "api_key"
    }
    
    masked_data = {}
    for key, value in data.items():
        if any(sensitive_field in key.lower() for sensitive_field in sensitive_fields):
            masked_data[key] = "***"
        elif isinstance(value, dict):
            masked_data[key] = mask_sensitive_data(value)
        else:
            masked_data[key] = value
    
    return masked_data


# Initialize logging on import
settings = get_settings()
configure_logging(
    log_level=settings.log_level,
    json_format=not settings.debug
) 