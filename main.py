from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import gdown

# Download models from Google Drive if they don't exist locally
def download_models():
    """Download model files from Google Drive if they don't exist locally"""
    
    # Model URLs and file paths
    models = [
        {
            "url": "https://drive.google.com/uc?id=1F60Octn-RTJzvBw56jZL9S7Km0gCunpO",
            "file_path": "model_fold_1.keras",
            "name": "Heart Sound Model"
        },
        {
            "url": "https://drive.google.com/uc?id=1yg1OrvN_47FCYj6pt00fecc8LoG2ltWn", 
            "file_path": "xray_pneumonia_model.keras",
            "name": "X-ray Pneumonia Model"
        }
    ]
    
    for model in models:
        if not os.path.exists(model["file_path"]):
            print(f"Downloading {model['name']} from Google Drive...")
            try:
                gdown.download(model["url"], model["file_path"], quiet=False)
                print(f"✓ Successfully downloaded {model['name']}")
            except Exception as e:
                print(f"✗ Error downloading {model['name']}: {str(e)}")
                print(f"  Please manually download from: {model['url']}")
        else:
            print(f"✓ {model['name']} already exists at {model['file_path']}")

# Download models at startup
print("=== Model Download Check ===")
download_models()
print("=== Starting API Server ===")

from router import router

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