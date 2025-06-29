import asyncio
import time
from typing import Dict, Any
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import io

class XRayPneumoniaHandler:
    """Handler class for X-ray pneumonia prediction operations"""
    
    def __init__(self, model_path: str = "xray_pneumonia_model.keras"):
        """
        Initialize the handler with model path
        
        Args:
            model_path: Path to the trained Keras model for X-ray pneumonia classification
        """
        self.model_path = model_path
        self.model = None
        self.target_size = (180, 180)  # Input size expected by the model
        self._load_model()
    
    def _load_model(self):
        """Load the Keras model on initialization"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                print(f"X-ray pneumonia model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found: {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading X-ray model: {e}")
            self.model = None
    
    async def predict_xray(self, image_file_path: str = None, image_bytes: bytes = None) -> Dict[str, Any]:
        """
        Predict pneumonia from X-ray image
        
        Args:
            image_file_path: Path to the image file (optional)
            image_bytes: Raw image bytes (optional)
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        try:
            if self.model is None:
                raise RuntimeError("X-ray pneumonia model not loaded")
            
            # Validate input
            if image_file_path is None and image_bytes is None:
                raise ValueError("Either image_file_path or image_bytes must be provided")
            
            if image_file_path and not os.path.exists(image_file_path):
                raise FileNotFoundError(f"Image file not found: {image_file_path}")
            
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                None, 
                self._predict_sync, 
                image_file_path,
                image_bytes
            )
            
            processing_time = time.time() - start_time
            
            return {
                "prediction": prediction["class"],
                "confidence": prediction["confidence"],
                "probability_pneumonia": prediction["probability_pneumonia"],
                "probability_normal": prediction["probability_normal"],
                "processing_time": round(processing_time, 3)
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "prediction": "error",
                "confidence": 0.0,
                "probability_pneumonia": 0.0,
                "probability_normal": 0.0,
                "processing_time": round(processing_time, 3),
                "error": str(e)
            }
    
    def _predict_sync(self, image_file_path: str = None, image_bytes: bytes = None) -> Dict[str, Any]:
        """
        Synchronous prediction method
        
        Args:
            image_file_path: Path to the image file
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load and preprocess image
            if image_bytes is not None:
                # Load from bytes
                image = Image.open(io.BytesIO(image_bytes))
            else:
                # Load from file path
                image = Image.open(image_file_path)
            
            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to target size (180, 180) as used in training
            image = image.resize(self.target_size)
            
            # Convert to numpy array and normalize (rescale by 1./255 as in training)
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0
            
            # Add batch dimension: (1, 180, 180, 3)
            image_array = np.expand_dims(image_array, axis=0)
            
            # Make prediction
            prediction = self.model.predict(image_array, verbose=0)
            
            # The model uses sigmoid activation for binary classification
            # Output is probability of pneumonia (class 1)
            probability_pneumonia = float(prediction[0][0])
            probability_normal = 1.0 - probability_pneumonia
            
            # Classify based on threshold (0.5)
            predicted_class = "pneumonia" if probability_pneumonia > 0.5 else "normal"
            confidence = max(probability_pneumonia, probability_normal)
            
            return {
                "class": predicted_class,
                "confidence": confidence,
                "probability_pneumonia": probability_pneumonia,
                "probability_normal": probability_normal
            }
            
        except Exception as e:
            raise RuntimeError(f"Error during X-ray prediction: {str(e)}")
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded X-ray model
        
        Returns:
            Dictionary with model information
        """
        try:
            if self.model is None:
                return {
                    "model_loaded": False,
                    "error": "Model not loaded"
                }
            
            # Get model summary info
            model_config = self.model.get_config()
            
            return {
                "model_loaded": True,
                "model_path": self.model_path,
                "input_shape": self.model.input_shape,
                "target_size": self.target_size,
                "model_type": "X-ray Pneumonia Classification",
                "classes": ["normal", "pneumonia"],
                "preprocessing": "Resize to 180x180, RGB conversion, normalization (0-1)"
            }
            
        except Exception as e:
            return {
                "model_loaded": self.model is not None,
                "error": f"Error getting model info: {str(e)}"
            }
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for X-ray prediction service"""
        return {
            "service": "X-ray Pneumonia Prediction",
            "status": "healthy" if self.is_model_loaded() else "model_not_loaded",
            "model_path": self.model_path,
            "model_loaded": self.is_model_loaded()
        } 