"""Custom exceptions for NG FX Predictor."""

from typing import Any, Dict, Optional


class NGFXPredictorError(Exception):
    """Base exception for NG FX Predictor."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize exception.
        
        Args:
            message: Error message
            error_code: Optional error code
            details: Optional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
        }


class DataSourceError(NGFXPredictorError):
    """Exception for data source related errors."""
    
    def __init__(self, message: str, source_name: Optional[str] = None, **kwargs):
        """Initialize data source error.
        
        Args:
            message: Error message
            source_name: Name of the data source
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.source_name = source_name
        if source_name:
            self.details["source_name"] = source_name


class DataQualityError(NGFXPredictorError):
    """Exception for data quality issues."""
    
    def __init__(self, message: str, quality_score: Optional[float] = None, **kwargs):
        """Initialize data quality error.
        
        Args:
            message: Error message
            quality_score: Quality score (0-1)
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.quality_score = quality_score
        if quality_score is not None:
            self.details["quality_score"] = quality_score


class ModelError(NGFXPredictorError):
    """Exception for model related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, model_version: Optional[str] = None, **kwargs):
        """Initialize model error.
        
        Args:
            message: Error message
            model_name: Name of the model
            model_version: Version of the model
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.model_name = model_name
        self.model_version = model_version
        if model_name:
            self.details["model_name"] = model_name
        if model_version:
            self.details["model_version"] = model_version


class InferenceError(NGFXPredictorError):
    """Exception for inference related errors."""
    
    def __init__(self, message: str, inference_type: Optional[str] = None, **kwargs):
        """Initialize inference error.
        
        Args:
            message: Error message
            inference_type: Type of inference (prediction, explanation)
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.inference_type = inference_type
        if inference_type:
            self.details["inference_type"] = inference_type


class ConfigurationError(NGFXPredictorError):
    """Exception for configuration related errors."""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_key: Configuration key that caused the error
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.config_key = config_key
        if config_key:
            self.details["config_key"] = config_key


class ValidationError(NGFXPredictorError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, validation_type: Optional[str] = None, **kwargs):
        """Initialize validation error.
        
        Args:
            message: Error message
            validation_type: Type of validation that failed
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.validation_type = validation_type
        if validation_type:
            self.details["validation_type"] = validation_type


class FeatureEngineeringError(NGFXPredictorError):
    """Exception for feature engineering related errors."""
    
    def __init__(self, message: str, feature_name: Optional[str] = None, **kwargs):
        """Initialize feature engineering error.
        
        Args:
            message: Error message
            feature_name: Name of the feature that caused the error
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.feature_name = feature_name
        if feature_name:
            self.details["feature_name"] = feature_name


class TrainingError(NGFXPredictorError):
    """Exception for model training related errors."""
    
    def __init__(self, message: str, horizon: Optional[int] = None, **kwargs):
        """Initialize training error.
        
        Args:
            message: Error message
            horizon: Forecast horizon
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.horizon = horizon
        if horizon is not None:
            self.details["horizon"] = horizon


class APIError(NGFXPredictorError):
    """Exception for API related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        """Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.status_code = status_code
        if status_code is not None:
            self.details["status_code"] = status_code


class CacheError(NGFXPredictorError):
    """Exception for cache related errors."""
    
    def __init__(self, message: str, cache_key: Optional[str] = None, **kwargs):
        """Initialize cache error.
        
        Args:
            message: Error message
            cache_key: Cache key that caused the error
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.cache_key = cache_key
        if cache_key:
            self.details["cache_key"] = cache_key


class OrchestrationError(NGFXPredictorError):
    """Exception for orchestration related errors."""
    
    def __init__(self, message: str, flow_name: Optional[str] = None, task_name: Optional[str] = None, **kwargs):
        """Initialize orchestration error.
        
        Args:
            message: Error message
            flow_name: Name of the flow
            task_name: Name of the task
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.flow_name = flow_name
        self.task_name = task_name
        if flow_name:
            self.details["flow_name"] = flow_name
        if task_name:
            self.details["task_name"] = task_name


class DataIngestionError(NGFXPredictorError):
    """Exception for data ingestion failures."""
    
    def __init__(self, message: str, source_name: Optional[str] = None, **kwargs):
        """Initialize data ingestion error.
        
        Args:
            message: Error message
            source_name: Name of the data source
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.source_name = source_name
        if source_name:
            self.details["source_name"] = source_name


class FeatureEngineeringError(NGFXPredictorError):
    """Exception for feature engineering failures."""
    
    def __init__(self, message: str, feature_name: Optional[str] = None, **kwargs):
        """Initialize feature engineering error.
        
        Args:
            message: Error message
            feature_name: Name of the feature
            **kwargs: Additional arguments
        """
        super().__init__(message, **kwargs)
        self.feature_name = feature_name
        if feature_name:
            self.details["feature_name"] = feature_name 