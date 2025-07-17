# Requirements Document

## Introduction

The NG FX Predictor is a production-ready forecasting platform that ingests Nigerian macroeconomic and foreign exchange data daily, retrains machine learning models nightly, and serves USD/NGN exchange rate forecasts with explanations through REST APIs and chat interfaces. The system is designed to run on a single VM with Docker Compose and scale to Kubernetes, providing 1-5 day forecast horizons with automated data pipelines and comprehensive observability.

## Requirements

### Requirement 1

**User Story:** As a financial analyst, I want to get USD/NGN exchange rate forecasts for 1-5 days ahead, so that I can make informed trading and risk management decisions.

#### Acceptance Criteria

1. WHEN I send a POST request to /predict with horizon parameter THEN the system SHALL return a JSON forecast with date, predicted rate, and 80% prediction interval
2. WHEN I request a forecast horizon between 1-5 days THEN the system SHALL provide accurate predictions based on the latest trained model
3. WHEN I request a forecast horizon outside 1-5 days THEN the system SHALL return an error with appropriate message
4. IF the model has not been trained in the last 48 hours THEN the system SHALL indicate stale model status in the response

### Requirement 2

**User Story:** As a financial analyst, I want to understand why the model made specific predictions, so that I can validate the forecast logic and build confidence in the results.

#### Acceptance Criteria

1. WHEN I send a GET request to /explain with a specific date THEN the system SHALL return the top-10 SHAP feature contributions for that prediction
2. WHEN I request explanations for a date without available predictions THEN the system SHALL return an appropriate error message
3. WHEN explanations are generated THEN the system SHALL include feature names, contribution values, and directional impact

### Requirement 3

**User Story:** As a business user, I want to interact with the forecasting system through natural language, so that I can get predictions without learning API syntax.

#### Acceptance Criteria

1. WHEN I send a POST request to /chat with "What's the forecast for next week?" THEN the system SHALL interpret this as a 5-day horizon request and return formatted predictions
2. WHEN I ask "Why" questions in chat THEN the system SHALL provide explanation summaries in natural language
3. WHEN I use ambiguous time references THEN the system SHALL ask for clarification or use reasonable defaults

### Requirement 4

**User Story:** As a data engineer, I want the system to automatically ingest fresh data daily from multiple sources, so that forecasts remain accurate and up-to-date.

#### Acceptance Criteria

1. WHEN the daily ingestion pipeline runs THEN the system SHALL fetch data from CBN rates, World Bank reserves, DMO debt, EIA Brent oil, and news sentiment sources
2. WHEN any data source fails THEN the system SHALL log the error and continue with available sources
3. WHEN new data is ingested THEN the system SHALL validate data quality and freshness before processing
4. IF data is more than 7 days stale THEN the system SHALL raise alerts and mark the source as degraded

### Requirement 5

**User Story:** As a data scientist, I want the system to automatically retrain models nightly with hyperparameter optimization, so that predictions adapt to changing market conditions.

#### Acceptance Criteria

1. WHEN the nightly training pipeline runs THEN the system SHALL perform Optuna hyperparameter search with 10 trials for each forecast horizon
2. WHEN training completes successfully THEN the system SHALL log the best model to MLflow and export to ONNX and pickle formats
3. WHEN a new model performs better than the current champion THEN the system SHALL automatically promote it to production
4. IF training fails THEN the system SHALL retain the previous model and alert administrators

### Requirement 6

**User Story:** As a DevOps engineer, I want the system to be containerized and easily deployable, so that I can run it consistently across different environments.

#### Acceptance Criteria

1. WHEN I run `docker compose up` THEN the system SHALL start all services (API, MLflow, Prefect, Postgres) without manual configuration
2. WHEN the system starts for the first time THEN it SHALL use sane defaults and not require pre-existing configuration files
3. WHEN I run `make ci` THEN the system SHALL pass all linting, type checking, tests, and achieve ≥80% code coverage
4. WHEN I access http://localhost:8000/healthz THEN the system SHALL return {"status":"ok"} with HTTP 200

### Requirement 7

**User Story:** As a system administrator, I want comprehensive observability and monitoring, so that I can maintain system health and troubleshoot issues effectively.

#### Acceptance Criteria

1. WHEN any component logs events THEN the system SHALL use structured JSON logging with ISO timestamps
2. WHEN I access /metrics THEN the system SHALL provide Prometheus-compatible metrics
3. WHEN errors occur THEN the system SHALL log detailed context including stack traces and request IDs
4. WHEN system health degrades THEN monitoring SHALL detect and alert on key metrics

### Requirement 8

**User Story:** As a security-conscious operator, I want proper secrets management and security scanning, so that the system meets production security standards.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL load configuration from environment variables with secure defaults
2. WHEN Docker images are built THEN they SHALL be scanned with Trivy for vulnerabilities
3. WHEN API requests are made THEN CORS SHALL be properly configured based on environment settings
4. IF sensitive data is logged THEN it SHALL be masked or excluded from log output

### Requirement 9

**User Story:** As a new developer, I want clear documentation and setup instructions, so that I can contribute to the project quickly.

#### Acceptance Criteria

1. WHEN I read the README THEN it SHALL include a Mermaid architecture diagram and 15-minute quick-start guide
2. WHEN I follow the setup instructions THEN I SHALL be able to run the system without prior knowledge of the tech stack
3. WHEN I want to contribute THEN CONTRIBUTING.md SHALL explain branch naming, commit conventions, and pre-commit setup
4. WHEN I need help with deployment THEN docs/setup_vm.md SHALL provide VM configuration guidance

### Requirement 10

**User Story:** As a quality assurance engineer, I want comprehensive automated testing, so that I can ensure system reliability and catch regressions early.

#### Acceptance Criteria

1. WHEN tests run THEN they SHALL achieve ≥80% code coverage across all modules
2. WHEN integration tests execute THEN they SHALL use Testcontainers to spin up real services
3. WHEN unit tests run THEN they SHALL use VCR.py cassettes for offline HTTP testing
4. WHEN edge cases are tested THEN Hypothesis property-based testing SHALL generate diverse test scenarios