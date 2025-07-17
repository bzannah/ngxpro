# Implementation Plan

- [ ] 1. Set up project structure and core configuration
  - Create directory structure following the specified layout
  - Implement configuration management with Pydantic settings
  - Set up pyproject.toml with all dependencies and development tools
  - Create .env.example with all required environment variables
  - _Requirements: 6.2, 8.1_

- [ ] 2. Implement base data infrastructure
  - [ ] 2.1 Create abstract BaseDataSource class
    - Define interface with fetch() method returning Polars DataFrame
    - Implement error handling, retry logic, and logging
    - Add data validation and caching mechanisms
    - _Requirements: 4.1, 4.3_

  - [ ] 2.2 Implement database models and connection utilities
    - Create PostgreSQL connection management with connection pooling
    - Define SQLAlchemy models for raw_data, features, and predictions tables
    - Implement database initialization and migration utilities
    - Write unit tests for database operations
    - _Requirements: 4.3, 6.2_

- [ ] 3. Implement data source connectors
  - [ ] 3.1 Create CBN exchange rates data source
    - Implement CBNRatesSource class extending BaseDataSource
    - Handle CBN API authentication and rate limiting
    - Create VCR.py cassettes for offline testing
    - Write unit tests with 3-day fixture data
    - _Requirements: 4.1, 4.2, 10.3_

  - [ ] 3.2 Create World Bank reserves data source
    - Implement WorldBankReservesSource with API integration
    - Handle World Bank API pagination and data formatting
    - Create test fixtures and VCR cassettes
    - Write comprehensive unit tests
    - _Requirements: 4.1, 4.2, 10.3_

  - [ ] 3.3 Create DMO debt data source (PDF processing stub)
    - Implement DMODebtSource with PDF parsing capabilities
    - Create stub implementation that processes CSV fallback
    - Add error handling for PDF parsing failures
    - Write tests with sample PDF and CSV fixtures
    - _Requirements: 4.1, 4.2, 10.3_

  - [ ] 3.4 Create EIA Brent oil price data source
    - Implement EIABrentSource with EIA API integration
    - Handle API key authentication and data transformation
    - Create VCR cassettes and test fixtures
    - Write unit tests for data validation
    - _Requirements: 4.1, 4.2, 10.3_

  - [ ] 3.5 Create news sentiment analysis data source
    - Implement NewsSentimentSource with FinBERT integration
    - Create sentiment scoring pipeline (negative/neutral/positive)
    - Add text preprocessing and batch processing capabilities
    - Write tests with sample news articles and expected sentiments
    - _Requirements: 4.1, 4.2, 10.3_

- [ ] 4. Implement feature engineering pipeline
  - [ ] 4.1 Create FeatureSet Pydantic schema
    - Define comprehensive schema with all required features
    - Add validation rules for data quality checks
    - Implement serialization/deserialization methods
    - Write schema validation tests
    - _Requirements: 4.3, 5.3_

  - [ ] 4.2 Implement FeatureEngineer class
    - Create feature transformation methods (rolling returns, regime flags)
    - Implement days-since-last-update calculations
    - Add technical indicators and derived features
    - Write unit tests for each feature transformation
    - _Requirements: 4.3, 5.3_

  - [ ] 4.3 Create feature validation and quality checks
    - Implement data freshness validation (7-day staleness check)
    - Add outlier detection and data quality metrics
    - Create feature completeness validation
    - Write tests for edge cases and validation failures
    - _Requirements: 4.4, 7.3_

- [ ] 5. Implement model training infrastructure
  - [ ] 5.1 Create ModelTrainer base class
    - Define training interface with hyperparameter optimization
    - Implement MLflow integration for experiment tracking
    - Add model evaluation and validation methods
    - Write unit tests with synthetic training data
    - _Requirements: 5.1, 5.2_

  - [ ] 5.2 Implement LightGBM model trainer
    - Create LightGBMTrainer extending ModelTrainer
    - Implement Optuna hyperparameter optimization (10 trials)
    - Add cross-validation and model evaluation metrics
    - Write tests for training pipeline with known datasets
    - _Requirements: 5.1, 5.2_

  - [ ] 5.3 Create model export and registry utilities
    - Implement ONNX and pickle model export functionality
    - Create MLflow model registration and versioning
    - Add champion model promotion logic
    - Write tests for model serialization and loading
    - _Requirements: 5.2, 5.3_

  - [ ] 5.4 Implement multi-horizon training pipeline
    - Create training orchestration for horizons T+1 to T+5
    - Add parallel training capabilities for different horizons
    - Implement model comparison and selection logic
    - Write integration tests for complete training workflow
    - _Requirements: 5.1, 5.3_

- [ ] 6. Implement model inference and serving
  - [ ] 6.1 Create model inference engine
    - Implement ModelInference class with ONNX runtime
    - Add model loading and caching mechanisms
    - Create prediction interval calculation methods
    - Write unit tests for inference accuracy and performance
    - _Requirements: 1.1, 1.2_

  - [ ] 6.2 Implement SHAP explainer service
    - Create SHAPExplainer class for model interpretability
    - Add top-10 feature contribution calculation
    - Implement explanation caching for performance
    - Write tests for explanation consistency and accuracy
    - _Requirements: 2.1, 2.2, 2.3_

- [ ] 7. Implement FastAPI web service
  - [ ] 7.1 Create core FastAPI application structure
    - Set up FastAPI app with middleware and CORS configuration
    - Implement structured logging with request IDs
    - Add health check and metrics endpoints
    - Write tests for application startup and configuration
    - _Requirements: 6.1, 7.1, 7.2_

  - [ ] 7.2 Implement prediction API endpoints
    - Create POST /predict endpoint with horizon validation
    - Implement ForecastResponse schema and validation
    - Add error handling for invalid requests and model failures
    - Write API tests for all prediction scenarios
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [ ] 7.3 Implement explanation API endpoints
    - Create GET /explain endpoint with date parameter validation
    - Implement ExplanationResponse schema
    - Add error handling for missing predictions and invalid dates
    - Write API tests for explanation functionality
    - _Requirements: 2.1, 2.2, 2.3_

  - [ ] 7.4 Implement chat interface endpoints
    - Create POST /chat endpoint with natural language processing
    - Implement pattern-based intent recognition
    - Add response formatting for different query types
    - Write tests for various chat scenarios and edge cases
    - _Requirements: 3.1, 3.2, 3.3_

- [ ] 8. Implement orchestration with Prefect
  - [ ] 8.1 Create data ingestion Prefect tasks
    - Implement refresh_data task with parallel source processing
    - Add error handling and retry logic for failed sources
    - Create task dependencies and flow orchestration
    - Write tests for task execution and error scenarios
    - _Requirements: 4.1, 4.2, 4.4_

  - [ ] 8.2 Create model training Prefect flow
    - Implement daily_ingest_train flow with task dependencies
    - Add feature_engineer and train_models tasks
    - Create promote_champion task for model deployment
    - Write integration tests for complete flow execution
    - _Requirements: 5.1, 5.3_

  - [ ] 8.3 Implement flow scheduling and monitoring
    - Set up cron scheduling for 2 AM Abuja time (0 2 * * *)
    - Add flow monitoring and alerting capabilities
    - Implement graceful error handling and recovery
    - Write tests for scheduling and monitoring functionality
    - _Requirements: 5.1, 7.3_

- [ ] 9. Implement containerization and deployment
  - [ ] 9.1 Create Dockerfile and Docker Compose configuration
    - Write multi-stage Dockerfile with security best practices
    - Create docker-compose.yaml with all required services
    - Configure service networking and volume management
    - Test container builds and service startup
    - _Requirements: 6.1, 6.2_

  - [ ] 9.2 Implement environment configuration and secrets
    - Create comprehensive .env.example with all variables
    - Implement secure defaults and environment variable validation
    - Add Docker secrets integration for production
    - Write tests for configuration loading and validation
    - _Requirements: 6.2, 8.1, 8.3_

- [ ] 10. Implement monitoring and observability
  - [ ] 10.1 Create structured logging infrastructure
    - Implement JSON logging with ISO timestamps using structlog
    - Add correlation IDs and request tracing
    - Create log filtering and sensitive data masking
    - Write tests for logging functionality and format validation
    - _Requirements: 7.1, 7.3, 8.4_

  - [ ] 10.2 Implement metrics and health monitoring
    - Create Prometheus-compatible /metrics endpoint
    - Add application performance and business metrics
    - Implement health check logic with dependency validation
    - Write tests for metrics collection and health status
    - _Requirements: 7.2, 6.1_

- [ ] 11. Implement comprehensive testing suite
  - [ ] 11.1 Create unit test infrastructure
    - Set up pytest configuration with coverage reporting
    - Create test fixtures for all data sources (3-day samples)
    - Implement VCR.py cassettes for HTTP mocking
    - Write property-based tests using Hypothesis
    - _Requirements: 10.1, 10.3, 10.4_

  - [ ] 11.2 Create integration tests with Testcontainers
    - Implement end-to-end test using Testcontainers-Python
    - Create Docker service integration tests
    - Add API integration tests with real model artifacts
    - Write database integration tests with test data
    - _Requirements: 10.2, 6.1_

  - [ ] 11.3 Implement CI/CD pipeline and quality gates
    - Create GitHub Actions workflow for CI/CD
    - Add Trivy security scanning for container images
    - Implement code quality checks (ruff, black, isort, mypy)
    - Set up coverage reporting and quality thresholds (≥80%)
    - _Requirements: 6.3, 8.2, 10.1_

- [ ] 12. Create documentation and developer experience
  - [ ] 12.1 Write comprehensive README with quick-start guide
    - Create Mermaid architecture diagram
    - Write 15-minute quick-start tutorial
    - Add installation instructions for Windows/macOS/Linux
    - Include troubleshooting section and FAQ
    - _Requirements: 9.1, 9.2_

  - [ ] 12.2 Create developer documentation
    - Write CONTRIBUTING.md with development guidelines
    - Create docs/architecture.md with technical details
    - Add docs/setup_vm.md for production deployment
    - Write API documentation and usage examples
    - _Requirements: 9.3, 9.4_

  - [ ] 12.3 Implement Makefile and development utilities
    - Create Makefile with dev_up, ci, and example_pred targets
    - Add development environment setup scripts
    - Implement sample prediction utility for testing
    - Write helper scripts for common development tasks
    - _Requirements: 6.3, 6.1_