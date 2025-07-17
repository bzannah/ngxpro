"""Configuration settings for NG FX Predictor."""

import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseModel):
    """Database configuration settings."""
    
    url: str = Field(
        default="postgresql://ngfx_user:ngfx_password@localhost:5432/ngfx_db",
        description="Database connection URL",
        env="DATABASE_URL"
    )
    pool_size: int = Field(default=20, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=50)
    pool_timeout: int = Field(default=30, ge=1, le=300)
    pool_recycle: int = Field(default=3600, ge=300, le=7200)


class MLflowSettings(BaseModel):
    """MLflow configuration settings."""
    
    tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI"
    )
    experiment_name: str = Field(
        default="ngfx-predictor",
        description="MLflow experiment name"
    )
    artifact_root: str = Field(
        default="./mlruns",
        description="MLflow artifact storage root"
    )


class PrefectSettings(BaseModel):
    """Prefect orchestration settings."""
    
    api_url: str = Field(
        default="http://localhost:4200/api",
        description="Prefect API URL"
    )
    logging_level: str = Field(
        default="INFO",
        description="Prefect logging level"
    )
    cloud_api_key: Optional[str] = Field(
        default=None,
        description="Prefect Cloud API key"
    )


class ModelSettings(BaseModel):
    """Model configuration settings."""
    
    registry_path: str = Field(
        default="./models",
        description="Model registry file path"
    )
    cache_size: int = Field(default=100, ge=1, le=1000)
    retrain_interval_hours: int = Field(default=24, ge=1, le=168)
    stale_threshold_hours: int = Field(default=48, ge=1, le=336)
    default_forecast_horizon: int = Field(default=5, ge=1, le=5)
    max_forecast_horizon: int = Field(default=5, ge=1, le=10)
    min_forecast_horizon: int = Field(default=1, ge=1, le=5)


class FeatureSettings(BaseModel):
    """Feature engineering settings."""
    
    window_days: int = Field(default=30, ge=1, le=365)
    min_observations: int = Field(default=10, ge=1, le=100)
    outlier_threshold: float = Field(default=3.0, ge=1.0, le=10.0)
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    version: str = Field(default="1.0.0", description="Feature version")
    lag_days: List[int] = Field(default=[1, 2, 3, 5, 7, 14, 30])
    rolling_windows: List[int] = Field(default=[7, 14, 30])
    enable_technical_indicators: bool = Field(default=True)
    enable_sentiment_features: bool = Field(default=True)


class DataSourceSettings(BaseModel):
    """Data source configuration settings."""
    
    cbn_api_base_url: str = Field(
        default="https://api.cbn.gov.ng",
        description="CBN API base URL"
    )
    cbn_api_key: Optional[str] = Field(
        default=None,
        description="CBN API key"
    )
    
    worldbank_api_base_url: str = Field(
        default="https://api.worldbank.org/v2",
        description="World Bank API base URL"
    )
    
    dmo_api_base_url: str = Field(
        default="https://www.dmo.gov.ng",
        description="DMO API base URL"
    )
    
    eia_api_base_url: str = Field(
        default="https://api.eia.gov",
        description="EIA API base URL"
    )
    eia_api_key: Optional[str] = Field(
        default=None,
        description="EIA API key"
    )
    
    news_api_base_url: str = Field(
        default="https://newsapi.org/v2",
        description="News API base URL"
    )
    news_api_key: Optional[str] = Field(
        default=None,
        description="News API key"
    )
    news_sentiment_model: str = Field(
        default="ProsusAI/finbert",
        description="News sentiment analysis model"
    )
    
    api_timeout: int = Field(default=30, ge=1, le=300)
    api_retries: int = Field(default=3, ge=0, le=10)


class DataValidationSettings(BaseModel):
    """Data validation settings."""
    
    staleness_threshold_days: int = Field(default=7, ge=1, le=30)
    quality_min_completeness: float = Field(default=0.8, ge=0.0, le=1.0)
    quality_max_outliers: float = Field(default=0.1, ge=0.0, le=1.0)
    validation_enabled: bool = Field(default=True)


class CachingSettings(BaseModel):
    """Caching configuration settings."""
    
    enabled: bool = Field(default=True)
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    max_size: int = Field(default=1000, ge=1, le=10000)
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )


class SchedulerSettings(BaseModel):
    """Scheduler settings."""
    
    daily_ingestion_hour: int = Field(default=2, ge=0, le=23)
    daily_ingestion_minute: int = Field(default=0, ge=0, le=59)
    enable_auto_ingestion: bool = Field(default=True)


class MonitoringSettings(BaseModel):
    """Monitoring and observability settings."""
    
    metrics_enabled: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    metrics_path: str = Field(default="/metrics")
    tracing_enabled: bool = Field(default=True)
    tracing_jaeger_host: str = Field(default="localhost")
    tracing_jaeger_port: int = Field(default=14268, ge=1024, le=65535)


class SecuritySettings(BaseModel):
    """Security configuration settings."""
    
    secret_key: str = Field(
        default="your-secret-key-change-this-in-production",
        description="Application secret key"
    )
    jwt_secret_key: str = Field(
        default="your-jwt-secret-key-change-this-in-production",
        description="JWT secret key"
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_minutes: int = Field(default=30, ge=1, le=1440)
    api_key_header: str = Field(default="X-API-Key")
    rate_limit_requests: int = Field(default=100, ge=1, le=10000)
    rate_limit_window_seconds: int = Field(default=60, ge=1, le=3600)


class TrainingSettings(BaseModel):
    """Model training configuration settings."""
    
    enabled: bool = Field(default=True)
    schedule: str = Field(
        default="0 2 * * *",
        description="Training schedule (cron format)"
    )
    timeout_minutes: int = Field(default=120, ge=1, le=1440)
    optuna_trials: int = Field(default=10, ge=1, le=100)
    optuna_timeout_seconds: int = Field(default=3600, ge=60, le=86400)
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5)
    test_split: float = Field(default=0.1, ge=0.05, le=0.3)
    random_seed: int = Field(default=42, ge=0)


class LightGBMSettings(BaseModel):
    """LightGBM model configuration settings."""
    
    boosting_type: str = Field(default="gbdt")
    objective: str = Field(default="regression")
    metric: str = Field(default="rmse")
    verbose: int = Field(default=-1)
    seed: int = Field(default=42, ge=0)


class SHAPSettings(BaseModel):
    """SHAP explainer configuration settings."""
    
    explainer_type: str = Field(default="tree")
    max_features: int = Field(default=10, ge=1, le=50)
    cache_size: int = Field(default=100, ge=1, le=1000)
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class ChatSettings(BaseModel):
    """Chat interface configuration settings."""
    
    enabled: bool = Field(default=True)
    max_tokens: int = Field(default=500, ge=1, le=4000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    timeout_seconds: int = Field(default=30, ge=1, le=300)


class Settings(BaseSettings):
    """Main application settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore",
    )
    
    # Application settings
    app_name: str = Field(default="NG FX Predictor")
    app_version: str = Field(default="0.1.0")
    app_description: str = Field(
        default="Production-ready Nigerian FX forecasting platform"
    )
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)
    reload: bool = Field(default=False)
    workers: int = Field(default=4, ge=1, le=32)
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    cors_allow_headers: List[str] = Field(default=["*"])
    
    # Database URL (override for nested settings)
    database_url: str = Field(
        default="postgresql://ngfx_user:ngfx_password@localhost:5432/ngfx_db",
        description="Database connection URL"
    )
    
    # Mock data configuration
    use_mock_data: bool = Field(
        default=False,
        description="Whether to use mock data instead of real APIs"
    )
    mock_data_path: str = Field(
        default="data/mock",
        description="Path to mock data directory"
    )
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    prefect: PrefectSettings = Field(default_factory=PrefectSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    data_source: DataSourceSettings = Field(default_factory=DataSourceSettings)
    data_validation: DataValidationSettings = Field(
        default_factory=DataValidationSettings
    )
    caching: CachingSettings = Field(default_factory=CachingSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    lightgbm: LightGBMSettings = Field(default_factory=LightGBMSettings)
    shap: SHAPSettings = Field(default_factory=SHAPSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    
    def model_post_init(self, __context: Any) -> None:
        """Override nested settings with main settings values."""
        # Override database URL in nested settings
        self.database.url = self.database_url
    
    @validator("cors_origins", pre=True)
    def parse_cors_origins(cls, v: Any) -> List[str]:
        """Parse CORS origins from environment variable."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @validator("cors_allow_methods", pre=True)
    def parse_cors_methods(cls, v: Any) -> List[str]:
        """Parse CORS methods from environment variable."""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")]
        return v
    
    @validator("cors_allow_headers", pre=True)
    def parse_cors_headers(cls, v: Any) -> List[str]:
        """Parse CORS headers from environment variable."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return os.getenv("DEPLOYMENT_ENVIRONMENT", "development") == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return not self.is_production
    
    def get_database_url(self, test: bool = False) -> str:
        """Get database URL for specified environment."""
        if test:
            return os.getenv("TEST_DATABASE_URL", self.database.url)
        return self.database.url
    
    def get_redis_url(self, test: bool = False) -> str:
        """Get Redis URL for specified environment."""
        if test:
            return os.getenv("TEST_REDIS_URL", self.caching.redis_url)
        return self.caching.redis_url


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings() 