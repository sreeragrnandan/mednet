from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import Dict, Any
from handler import HeartSoundHandler
from xray_handler import XRayPneumoniaHandler

# Create router instance
router = APIRouter(
    prefix="/api/v1",
    tags=["medical-prediction"]
)

# Model URLs from Google Drive (to be replaced with your actual shared URLs)
HEART_MODEL_URL = os.getenv("HEART_MODEL_URL", None)  # Set this in environment variables
XRAY_MODEL_URL = os.getenv("XRAY_MODEL_URL", None)   # Set this in environment variables

# Initialize handlers with Google Drive URLs
heart_handler = HeartSoundHandler(model_url=HEART_MODEL_URL)
xray_handler = XRayPneumoniaHandler(model_url=XRAY_MODEL_URL)

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "service": "Medical Prediction API"}

# Heart Sound Prediction Endpoints
@router.post("/predict/heart-sound")
async def predict_heart_sound(
    audio_file: UploadFile = File(..., description="Audio file (.wav format)")
) -> Dict[str, Any]:
    """
    Predict heart sound classification from uploaded audio file
    
    Args:
        audio_file: WAV audio file to analyze
        
    Returns:
        JSON response with prediction result
    """
    
    # Validate file type
    if not audio_file.filename.lower().endswith('.wav'):
        raise HTTPException(
            status_code=400, 
            detail="Only WAV files are supported"
        )
    
    # Check file size (limit to 10MB)
    if audio_file.size and audio_file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum size is 10MB"
        )
    
    try:
        # Create temporary file to save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            # Read and save uploaded file
            content = await audio_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Get prediction from handler
            result = await heart_handler.predict_audio(temp_file_path)
            
            return {
                "success": True,
                "filename": audio_file.filename,
                "file_size": len(content),
                "prediction": result["prediction"],
                "confidence": result.get("confidence"),
                "processing_time": result.get("processing_time"),
                "message": "Heart sound analysis completed successfully"
            }
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )

# X-ray Pneumonia Prediction Endpoints
@router.post("/predict/xray-pneumonia")
async def predict_xray_pneumonia(
    image_file: UploadFile = File(..., description="X-ray image file (.jpg, .jpeg, .png)")
) -> Dict[str, Any]:
    """
    Predict pneumonia from uploaded X-ray image
    
    Args:
        image_file: X-ray image file to analyze
        
    Returns:
        JSON response with prediction result
    """
    
    # Validate file type
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    file_extension = os.path.splitext(image_file.filename.lower())[1]
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (limit to 10MB)
    if image_file.size and image_file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum size is 10MB"
        )
    
    try:
        # Read the uploaded image file
        image_content = await image_file.read()
        
        # Get prediction from handler using image bytes
        result = await xray_handler.predict_xray(image_bytes=image_content)
        
        if result["prediction"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Error during prediction: {result.get('error', 'Unknown error')}"
            )
        
        return {
            "success": True,
            "filename": image_file.filename,
            "file_size": len(image_content),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probability_pneumonia": result["probability_pneumonia"],
            "probability_normal": result["probability_normal"],
            "processing_time": result["processing_time"],
            "message": "X-ray pneumonia analysis completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing X-ray image: {str(e)}"
        )

@router.post("/predict-batch")
async def predict_batch_heart_sounds(
    audio_files: list[UploadFile] = File(..., description="Multiple WAV audio files")
) -> Dict[str, Any]:
    """
    Predict heart sound classification for multiple audio files
    
    Args:
        audio_files: List of WAV audio files to analyze
        
    Returns:
        JSON response with batch prediction results
    """
    
    if len(audio_files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch"
        )
    
    results = []
    errors = []
    
    for audio_file in audio_files:
        try:
            # Validate file type
            if not audio_file.filename.lower().endswith('.wav'):
                errors.append({
                    "filename": audio_file.filename,
                    "error": "Only WAV files are supported"
                })
                continue
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                content = await audio_file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Get prediction
                result = await heart_handler.predict_audio(temp_file_path)
                
                results.append({
                    "filename": audio_file.filename,
                    "file_size": len(content),
                    "prediction": result["prediction"],
                    "confidence": result.get("confidence"),
                    "processing_time": result.get("processing_time")
                })
                
            finally:
                # Clean up
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            errors.append({
                "filename": audio_file.filename,
                "error": str(e)
            })
    
    return {
        "success": len(errors) == 0,
        "total_files": len(audio_files),
        "successful_predictions": len(results),
        "failed_predictions": len(errors),
        "results": results,
        "errors": errors if errors else None
    }

@router.get("/model-info/heart-sound")
async def get_heart_sound_model_info() -> Dict[str, Any]:
    """Get information about the loaded heart sound model"""
    try:
        info = await heart_handler.get_model_info()
        return {
            "success": True,
            "model_info": info
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting heart sound model info: {str(e)}"
        )

@router.get("/model-info/xray-pneumonia")
async def get_xray_model_info() -> Dict[str, Any]:
    """Get information about the loaded X-ray pneumonia model"""
    try:
        info = await xray_handler.get_model_info()
        return {
            "success": True,
            "model_info": info
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting X-ray model info: {str(e)}"
        )

@router.get("/health/heart-sound")
async def heart_sound_health_check() -> Dict[str, Any]:
    """Health check for heart sound prediction service"""
    try:
        health_info = await heart_handler.health_check()
        return health_info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking heart sound service health: {str(e)}"
        )

@router.get("/health/xray-pneumonia")
async def xray_health_check() -> Dict[str, Any]:
    """Health check for X-ray pneumonia prediction service"""
    try:
        health_info = await xray_handler.health_check()
        return health_info
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking X-ray service health: {str(e)}"
        ) 