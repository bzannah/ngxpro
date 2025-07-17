"""Metrics management for NG FX Predictor."""

import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..config import get_settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MetricsManager:
    """Metrics collection and management."""
    
    _instance = None
    _prometheus_initialized = False
    
    def __new__(cls):
        """Singleton pattern to prevent multiple Prometheus registrations."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize metrics manager."""
        # Skip initialization if already done (singleton)
        if hasattr(self, '_initialized'):
            return
            
        self.settings = get_settings()
        self.metrics: Dict[str, Any] = {}
        self.custom_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        if PROMETHEUS_AVAILABLE and self.settings.monitoring.metrics_enabled:
            if not self._prometheus_initialized:
                self._init_prometheus_metrics()
                MetricsManager._prometheus_initialized = True
        
        self._initialized = True
        logger.info("Metrics manager initialized")
    
    def timer(self, operation_name: str, **labels):
        """Create a timer context manager.
        
        Args:
            operation_name: Name of the operation
            **labels: Additional labels
            
        Returns:
            PerformanceTimer context manager
        """
        return PerformanceTimer(self, operation_name, **labels)
    
    def increment(self, metric_name: str, value: int = 1, **labels):
        """Increment a counter metric.
        
        Args:
            metric_name: Name of the metric
            value: Value to increment by
            **labels: Additional labels
        """
        timestamp = datetime.utcnow().isoformat()
        self.custom_metrics[metric_name].append({
            'timestamp': timestamp,
            'value': value,
            'labels': labels
        })
        
        # Log the increment
        logger.debug(f"Incremented {metric_name} by {value}")
    
    def gauge(self, metric_name: str, value: float, **labels):
        """Set a gauge metric value.
        
        Args:
            metric_name: Name of the metric
            value: Value to set
            **labels: Additional labels
        """
        timestamp = datetime.utcnow().isoformat()
        self.custom_metrics[metric_name].append({
            'timestamp': timestamp,
            'value': value,
            'labels': labels
        })
        
        # Log the gauge
        logger.debug(f"Set gauge {metric_name} to {value}")
    
    def _init_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        # Request metrics
        self.request_count = Counter(
            'ngfx_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.request_duration = Histogram(
            'ngfx_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint']
        )
        
        # Data source metrics
        self.data_fetch_count = Counter(
            'ngfx_data_fetch_total',
            'Total number of data fetches',
            ['source', 'status']
        )
        
        self.data_fetch_duration = Histogram(
            'ngfx_data_fetch_duration_seconds',
            'Data fetch duration in seconds',
            ['source']
        )
        
        self.data_quality_score = Gauge(
            'ngfx_data_quality_score',
            'Data quality score',
            ['source']
        )
        
        # Model metrics
        self.model_training_duration = Histogram(
            'ngfx_model_training_duration_seconds',
            'Model training duration in seconds',
            ['horizon', 'model_type']
        )
        
        self.model_inference_duration = Histogram(
            'ngfx_model_inference_duration_seconds',
            'Model inference duration in seconds',
            ['horizon', 'model_type']
        )
        
        self.model_accuracy = Gauge(
            'ngfx_model_accuracy',
            'Model accuracy score',
            ['horizon', 'model_type', 'metric']
        )
        
        # Prediction metrics
        self.predictions_count = Counter(
            'ngfx_predictions_total',
            'Total number of predictions',
            ['horizon', 'status']
        )
        
        self.explanation_count = Counter(
            'ngfx_explanations_total',
            'Total number of explanations',
            ['status']
        )
        
        # System metrics
        self.system_info = Info(
            'ngfx_system_info',
            'System information'
        )
        
        self.active_connections = Gauge(
            'ngfx_active_connections',
            'Number of active connections',
            ['type']
        )
        
        # Set system info
        self.system_info.info({
            'version': self.settings.app_version,
            'environment': 'production' if self.settings.is_production else 'development',
        })
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ) -> None:
        """Record request metrics.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            status_code: HTTP status code
            duration_seconds: Request duration
        """
        if PROMETHEUS_AVAILABLE and self.settings.monitoring.metrics_enabled:
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status=str(status_code)
            ).inc()
            
            self.request_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration_seconds)
        
        # Store custom metrics
        self.custom_metrics['requests'].append({
            'timestamp': datetime.now().isoformat(),
            'method': method,
            'endpoint': endpoint,
            'status_code': status_code,
            'duration_seconds': duration_seconds,
        })
    
    def record_data_fetch(
        self,
        source: str,
        status: str,
        duration_seconds: float,
        quality_score: Optional[float] = None
    ) -> None:
        """Record data fetch metrics.
        
        Args:
            source: Data source name
            status: Fetch status (success, failure, timeout)
            duration_seconds: Fetch duration
            quality_score: Optional data quality score
        """
        if PROMETHEUS_AVAILABLE and self.settings.monitoring.metrics_enabled:
            self.data_fetch_count.labels(
                source=source,
                status=status
            ).inc()
            
            self.data_fetch_duration.labels(
                source=source
            ).observe(duration_seconds)
            
            if quality_score is not None:
                self.data_quality_score.labels(
                    source=source
                ).set(quality_score)
        
        # Store custom metrics
        self.custom_metrics['data_fetches'].append({
            'timestamp': datetime.now().isoformat(),
            'source': source,
            'status': status,
            'duration_seconds': duration_seconds,
            'quality_score': quality_score,
        })
    
    def record_model_training(
        self,
        horizon: int,
        model_type: str,
        duration_seconds: float,
        accuracy_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """Record model training metrics.
        
        Args:
            horizon: Forecast horizon
            model_type: Type of model
            duration_seconds: Training duration
            accuracy_metrics: Optional accuracy metrics
        """
        if PROMETHEUS_AVAILABLE and self.settings.monitoring.metrics_enabled:
            self.model_training_duration.labels(
                horizon=str(horizon),
                model_type=model_type
            ).observe(duration_seconds)
            
            if accuracy_metrics:
                for metric_name, value in accuracy_metrics.items():
                    self.model_accuracy.labels(
                        horizon=str(horizon),
                        model_type=model_type,
                        metric=metric_name
                    ).set(value)
        
        # Store custom metrics
        self.custom_metrics['model_training'].append({
            'timestamp': datetime.now().isoformat(),
            'horizon': horizon,
            'model_type': model_type,
            'duration_seconds': duration_seconds,
            'accuracy_metrics': accuracy_metrics or {},
        })
    
    def record_model_inference(
        self,
        horizon: int,
        model_type: str,
        duration_seconds: float,
        status: str = "success"
    ) -> None:
        """Record model inference metrics.
        
        Args:
            horizon: Forecast horizon
            model_type: Type of model
            duration_seconds: Inference duration
            status: Inference status
        """
        if PROMETHEUS_AVAILABLE and self.settings.monitoring.metrics_enabled:
            self.model_inference_duration.labels(
                horizon=str(horizon),
                model_type=model_type
            ).observe(duration_seconds)
        
        # Store custom metrics
        self.custom_metrics['model_inference'].append({
            'timestamp': datetime.now().isoformat(),
            'horizon': horizon,
            'model_type': model_type,
            'duration_seconds': duration_seconds,
            'status': status,
        })
    
    def record_prediction(self, horizon: int, status: str = "success") -> None:
        """Record prediction metrics.
        
        Args:
            horizon: Forecast horizon
            status: Prediction status
        """
        if PROMETHEUS_AVAILABLE and self.settings.monitoring.metrics_enabled:
            self.predictions_count.labels(
                horizon=str(horizon),
                status=status
            ).inc()
        
        # Store custom metrics
        self.custom_metrics['predictions'].append({
            'timestamp': datetime.now().isoformat(),
            'horizon': horizon,
            'status': status,
        })
    
    def record_explanation(self, status: str = "success") -> None:
        """Record explanation metrics.
        
        Args:
            status: Explanation status
        """
        if PROMETHEUS_AVAILABLE and self.settings.monitoring.metrics_enabled:
            self.explanation_count.labels(
                status=status
            ).inc()
        
        # Store custom metrics
        self.custom_metrics['explanations'].append({
            'timestamp': datetime.now().isoformat(),
            'status': status,
        })
    
    def record_custom_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            labels: Optional labels
            timestamp: Optional timestamp
        """
        metric_data = {
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'value': value,
            'labels': labels or {},
        }
        
        self.custom_metrics[name].append(metric_data)
    
    def set_active_connections(self, connection_type: str, count: int) -> None:
        """Set active connections count.
        
        Args:
            connection_type: Type of connection (database, redis, etc.)
            count: Number of active connections
        """
        if PROMETHEUS_AVAILABLE and self.settings.monitoring.metrics_enabled:
            self.active_connections.labels(
                type=connection_type
            ).set(count)
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format.
        
        Returns:
            Prometheus metrics text
        """
        if not PROMETHEUS_AVAILABLE:
            return "# Prometheus not available\n"
        
        return generate_latest().decode('utf-8')
    
    def get_custom_metrics(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get custom metrics.
        
        Returns:
            Custom metrics dictionary
        """
        return dict(self.custom_metrics)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary.
        
        Returns:
            Metrics summary
        """
        summary = {
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'metrics_enabled': self.settings.monitoring.metrics_enabled,
            'custom_metrics_count': {
                name: len(metrics) for name, metrics in self.custom_metrics.items()
            },
            'last_updated': datetime.now().isoformat(),
        }
        
        return summary
    
    def clear_custom_metrics(self, older_than_hours: int = 24) -> int:
        """Clear old custom metrics.
        
        Args:
            older_than_hours: Clear metrics older than this many hours
            
        Returns:
            Number of metrics cleared
        """
        cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)
        cleared_count = 0
        
        for metric_name in list(self.custom_metrics.keys()):
            original_count = len(self.custom_metrics[metric_name])
            
            self.custom_metrics[metric_name] = [
                metric for metric in self.custom_metrics[metric_name]
                if datetime.fromisoformat(metric['timestamp']).timestamp() > cutoff_time
            ]
            
            cleared_count += original_count - len(self.custom_metrics[metric_name])
        
        logger.info(f"Cleared {cleared_count} old custom metrics")
        return cleared_count


class PerformanceTimer:
    """Context manager for measuring performance."""
    
    def __init__(self, metrics_manager: MetricsManager, operation_name: str, **labels):
        """Initialize performance timer.
        
        Args:
            metrics_manager: Metrics manager instance
            operation_name: Name of the operation
            **labels: Additional labels
        """
        self.metrics_manager = metrics_manager
        self.operation_name = operation_name
        self.labels = labels
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metrics."""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        # Record the timing
        self.metrics_manager.record_custom_metric(
            f"{self.operation_name}_duration_seconds",
            duration,
            self.labels
        )
        
        # Record success/failure
        status = "failure" if exc_type is not None else "success"
        self.metrics_manager.record_custom_metric(
            f"{self.operation_name}_status",
            1,
            {**self.labels, 'status': status}
        )
    
    @property
    def duration(self) -> Optional[float]:
        """Get duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


def timed_operation(metrics_manager: MetricsManager, operation_name: str, **labels):
    """Decorator for timing operations.
    
    Args:
        metrics_manager: Metrics manager instance
        operation_name: Name of the operation
        **labels: Additional labels
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with PerformanceTimer(metrics_manager, operation_name, **labels):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Global metrics manager instance
_metrics_manager = None


def get_metrics_manager() -> MetricsManager:
    """Get global metrics manager instance."""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager 