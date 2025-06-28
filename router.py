from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from typing import Dict, Any
from handler import HeartSoundHandler

# Create router instance
router = APIRouter(
    prefix="/api/v1",
    tags=["heart-sound-prediction"]
)

# Initialize handler
handler = HeartSoundHandler()

@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "service": "Heart Sound Prediction API"}

@router.post("/predict")
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
            result = await handler.predict_audio(temp_file_path)
            
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
                result = await handler.predict_audio(temp_file_path)
                
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

@router.get("/model-info")
async def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model"""
    try:
        info = await handler.get_model_info()
        return {
            "success": True,
            "model_info": info
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        ) 