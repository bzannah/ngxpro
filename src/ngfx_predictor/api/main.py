"""Main FastAPI application for NG FX Predictor."""

import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from ..config import get_settings
from ..data.database import get_database_manager
from ..data.sources.cbn import CBNRatesSource
from ..utils.logging import get_logger
from ..utils.metrics import get_metrics_manager
from ..utils.exceptions import NGFXPredictorError

logger = get_logger(__name__)
settings = get_settings()
metrics_manager = get_metrics_manager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting NG FX Predictor API")
    
    # Initialize database
    try:
        db_manager = get_database_manager()
        db_manager.create_tables()
        logger.info("Database tables created/verified")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down NG FX Predictor API")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add request ID to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


# Error handlers
@app.exception_handler(NGFXPredictorError)
async def ngfx_error_handler(request: Request, exc: NGFXPredictorError):
    """Handle NG FX Predictor specific errors."""
    logger.error(f"NG FX Predictor error: {exc.message}", extra=exc.details)
    
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.to_dict(),
            "request_id": getattr(request.state, "request_id", None),
        }
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle general errors."""
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "InternalServerError",
                "message": "An unexpected error occurred",
            },
            "request_id": getattr(request.state, "request_id", None),
        }
    )


# Health check endpoint
@app.get("/healthz")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        db_manager = get_database_manager()
        db_healthy = db_manager.check_connection()
        
        # Check data sources
        cbn_source = CBNRatesSource()
        cbn_health = cbn_source.get_health_status()
        
        health_status = {
            "status": "ok" if db_healthy and cbn_health.get("is_healthy", False) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "version": settings.app_version,
            "environment": "production" if settings.is_production else "development",
            "checks": {
                "database": "ok" if db_healthy else "error",
                "cbn_source": "ok" if cbn_health.get("is_healthy", False) else "error",
            }
        }
        
        status_code = 200 if health_status["status"] == "ok" else 503
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
            }
        )


# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    try:
        metrics_text = metrics_manager.get_prometheus_metrics()
        return JSONResponse(
            content={"metrics": metrics_text},
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NG FX Predictor API",
        "version": settings.app_version,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/healthz",
            "metrics": "/metrics",
            "docs": "/docs",
            "redoc": "/redoc",
        }
    }


# Data source status endpoint
@app.get("/data/status")
async def get_data_status():
    """Get data source status."""
    try:
        # Check CBN source
        cbn_source = CBNRatesSource()
        cbn_status = cbn_source.get_health_status()
        
        # Get database table counts
        db_manager = get_database_manager()
        table_counts = db_manager.get_table_counts()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data_sources": {
                "cbn_rates": cbn_status,
            },
            "database": {
                "connected": db_manager.check_connection(),
                "table_counts": table_counts,
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get data status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get data status")


# Sample prediction endpoint (MVP - returns demo data)
@app.post("/predict")
async def predict(horizon: int = 1):
    """Get USD/NGN exchange rate prediction."""
    try:
        # Validate horizon
        if not (1 <= horizon <= 5):
            raise HTTPException(
                status_code=400,
                detail="Horizon must be between 1 and 5 days"
            )
        
        # For MVP, return sample prediction
        # In full implementation, this would use trained models
        cbn_source = CBNRatesSource()
        latest_rate = cbn_source.get_latest_rate()
        
        if latest_rate is None:
            raise HTTPException(
                status_code=503,
                detail="Unable to get latest exchange rate"
            )
        
        # Generate sample prediction (trend + noise)
        import random
        trend = horizon * 0.5  # Slight upward trend
        noise = random.uniform(-10, 10)  # Random variation
        predicted_rate = latest_rate + trend + noise
        
        # Generate prediction interval (±2% for demo)
        margin = predicted_rate * 0.02
        prediction_interval = [
            predicted_rate - margin,
            predicted_rate + margin
        ]
        
        prediction = {
            "date": (datetime.now().date()).isoformat(),
            "horizon": horizon,
            "usd_ngn": round(predicted_rate, 2),
            "pi80": [round(x, 2) for x in prediction_interval],
            "confidence": 0.8,
            "model_version": "demo-v1.0",
            "timestamp": datetime.now().isoformat(),
        }
        
        # Record metrics
        metrics_manager.record_prediction(horizon, "success")
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        metrics_manager.record_prediction(horizon, "error")
        raise HTTPException(status_code=500, detail="Prediction failed")


# Sample explanation endpoint (MVP - returns demo data)
@app.get("/explain")
async def explain(date: str):
    """Get prediction explanation."""
    try:
        # Validate date format
        try:
            datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format. Use YYYY-MM-DD"
            )
        
        # For MVP, return sample explanation
        # In full implementation, this would use SHAP
        sample_explanation = {
            "date": date,
            "model_version": "demo-v1.0",
            "top_features": [
                {"feature": "previous_rate", "contribution": 0.45, "direction": "positive"},
                {"feature": "oil_price", "contribution": 0.23, "direction": "positive"},
                {"feature": "foreign_reserves", "contribution": -0.18, "direction": "negative"},
                {"feature": "sentiment_score", "contribution": 0.12, "direction": "positive"},
                {"feature": "day_of_week", "contribution": -0.08, "direction": "negative"},
            ],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Record metrics
        metrics_manager.record_explanation("success")
        
        return sample_explanation
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        metrics_manager.record_explanation("error")
        raise HTTPException(status_code=500, detail="Explanation failed")


# Sample chat endpoint (MVP - basic pattern matching)
@app.post("/chat")
async def chat(message: str):
    """Chat interface for natural language queries."""
    try:
        message_lower = message.lower()
        
        # Simple pattern matching for MVP
        if "forecast" in message_lower or "prediction" in message_lower:
            if "week" in message_lower:
                horizon = 5
            elif "tomorrow" in message_lower:
                horizon = 1
            else:
                horizon = 1
            
            # Get prediction
            prediction = await predict(horizon)
            
            response = f"Based on current data, I predict the USD/NGN exchange rate will be approximately ₦{prediction['usd_ngn']} in {horizon} day(s). The 80% confidence interval is ₦{prediction['pi80'][0]} to ₦{prediction['pi80'][1]}."
            
        elif "why" in message_lower or "explain" in message_lower:
            response = "To get explanations for predictions, please use the /explain endpoint with a specific date. For example: /explain?date=2024-01-15"
            
        elif "help" in message_lower:
            response = """I can help you with FX predictions! Here are some things you can ask:
            - "What's the forecast for tomorrow?"
            - "Predict the exchange rate for next week"
            - "Why did the rate change?"
            
            You can also use the API endpoints directly:
            - POST /predict?horizon=1 (for predictions)
            - GET /explain?date=2024-01-15 (for explanations)
            """
            
        else:
            response = "I understand you're asking about FX predictions. Try asking 'What's the forecast for tomorrow?' or 'help' for more options."
        
        return {
            "message": message,
            "response": response,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.reload,
    ) 