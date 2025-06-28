from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from router import router
import os

# Create FastAPI application
app = FastAPI(
    title="Heart Sound Prediction API",
    description="REST API for heart sound classification using deep learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the router
app.include_router(router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Heart Sound Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/api/v1/health",
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict-batch",
            "model_info": "/api/v1/model-info",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )

# Health check for the entire application
@app.get("/health")
async def app_health():
    """Application health check"""
    return {
        "status": "healthy",
        "service": "Heart Sound Prediction API",
        "model_file_exists": os.path.exists("model_fold_1.keras")
    }

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 