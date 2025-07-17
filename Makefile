# NG FX Predictor Makefile

.PHONY: help dev_up dev_down build clean test lint format ci example_pred

# Default target
help:
	@echo "NG FX Predictor - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  dev_up       - Start development environment"
	@echo "  dev_down     - Stop development environment"
	@echo "  build        - Build Docker images"
	@echo "  clean        - Clean up containers and volumes"
	@echo ""
	@echo "Code Quality:"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linters"
	@echo "  format       - Format code"
	@echo "  ci           - Run CI checks (lint, test, coverage)"
	@echo ""
	@echo "Examples:"
	@echo "  example_pred - Run example prediction"
	@echo "  logs         - Show application logs"
	@echo "  status       - Show system status"

# Development environment
dev_up:
	@echo "Starting NG FX Predictor development environment..."
	docker-compose up -d postgres redis mlflow prefect
	@echo "Waiting for services to be ready..."
	sleep 10
	docker-compose up -d ngfx_api
	@echo "Development environment started!"
	@echo "Access the API at: http://localhost:8000"
	@echo "Access MLflow at: http://localhost:5000"
	@echo "Access Prefect at: http://localhost:4200"

dev_down:
	@echo "Stopping development environment..."
	docker-compose down

# Build Docker images
build:
	@echo "Building Docker images..."
	docker-compose build

# Clean up
clean:
	@echo "Cleaning up containers and volumes..."
	docker-compose down -v --remove-orphans
	docker system prune -f

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -e ".[dev]"

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src/ngfx_predictor --cov-report=html --cov-report=term-missing

# Linting
lint:
	@echo "Running linters..."
	ruff check src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/
	mypy src/

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/
	ruff check --fix src/ tests/

# CI checks
ci: lint test
	@echo "All CI checks passed!"

# Example prediction
example_pred:
	@echo "Running example prediction..."
	curl -X POST "http://localhost:8000/predict?horizon=1" -H "Content-Type: application/json" | jq '.'

# Show logs
logs:
	docker-compose logs -f ngfx_api

# Show system status
status:
	@echo "System Status:"
	@echo "============="
	@echo "API Health:"
	curl -s http://localhost:8000/healthz | jq '.'
	@echo ""
	@echo "Data Sources:"
	curl -s http://localhost:8000/data/status | jq '.'

# Run full system
full_up:
	@echo "Starting full system with monitoring..."
	docker-compose up -d

# Database operations
db_migrate:
	@echo "Running database migrations..."
	docker-compose exec ngfx_api python -m alembic upgrade head

db_shell:
	@echo "Opening database shell..."
	docker-compose exec postgres psql -U ngfx_user -d ngfx_db

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "MLflow: http://localhost:5000"
	@echo "Prefect: http://localhost:4200"

# Documentation
docs:
	@echo "Building documentation..."
	mkdocs build

docs_serve:
	@echo "Serving documentation..."
	mkdocs serve

# Security scan
security_scan:
	@echo "Running security scan..."
	docker run --rm -v $(PWD):/app -w /app aquasec/trivy fs .

# Performance test
perf_test:
	@echo "Running performance tests..."
	ab -n 100 -c 10 http://localhost:8000/healthz 