"""Basic API tests for NG FX Predictor."""

import pytest
from fastapi.testclient import TestClient

from src.ngfx_predictor.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "NG FX Predictor API"
    assert "version" in data
    assert "endpoints" in data


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/healthz")
    assert response.status_code in [200, 503]  # May be degraded without full setup
    data = response.json()
    assert "status" in data
    assert "timestamp" in data
    assert "version" in data


def test_prediction_endpoint():
    """Test the prediction endpoint."""
    response = client.post("/predict?horizon=1")
    assert response.status_code == 200
    data = response.json()
    assert "usd_ngn" in data
    assert "pi80" in data
    assert "confidence" in data
    assert "model_version" in data
    assert data["horizon"] == 1


def test_prediction_invalid_horizon():
    """Test prediction with invalid horizon."""
    response = client.post("/predict?horizon=10")
    assert response.status_code == 400


def test_explanation_endpoint():
    """Test the explanation endpoint."""
    response = client.get("/explain?date=2024-01-15")
    assert response.status_code == 200
    data = response.json()
    assert "date" in data
    assert "top_features" in data
    assert "model_version" in data


def test_explanation_invalid_date():
    """Test explanation with invalid date."""
    response = client.get("/explain?date=invalid")
    assert response.status_code == 400


def test_chat_endpoint():
    """Test the chat endpoint."""
    response = client.post("/chat", json={"message": "What's the forecast for tomorrow?"})
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "response" in data
    assert "timestamp" in data


def test_data_status_endpoint():
    """Test the data status endpoint."""
    response = client.get("/data/status")
    assert response.status_code == 200
    data = response.json()
    assert "data_sources" in data
    assert "database" in data


def test_metrics_endpoint():
    """Test the metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data 