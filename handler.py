import asyncio
import time
from typing import Dict, Any, Optional
import os
import numpy as np
from predict import predict_heart_sound, extract_features
import librosa
import keras

class HeartSoundHandler:
    """Handler class for heart sound prediction operations"""
    
    def __init__(self, model_path: str = "model_fold_1.keras"):
        """
        Initialize the handler with model path
        
        Args:
            model_path: Path to the trained Keras model
        """
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Keras model on initialization"""
        try:
            from predict import F1Score
            
            custom_objects = {
                'F1Score': F1Score,
                'f1_score': F1Score
            }
            
            # Try to load the model
            try:
                self.model = keras.models.load_model(self.model_path, custom_objects=custom_objects)
                print(f"Model loaded successfully from {self.model_path}")
            except Exception as load_error:
                # Fallback: load without compilation
                self.model = keras.models.load_model(
                    self.model_path, 
                    custom_objects=custom_objects, 
                    compile=False
                )
                print(f"Model loaded without compilation from {self.model_path}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    async def predict_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Predict heart sound classification for an audio file
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing prediction results
        """
        start_time = time.time()
        
        try:
            # Validate file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                None, 
                self._predict_sync, 
                audio_file_path
            )
            
            processing_time = time.time() - start_time
            
            # Get additional audio info
            audio_info = await self._get_audio_info(audio_file_path)
            
            return {
                "prediction": prediction["class"],
                "confidence": prediction["confidence"],
                "probabilities": prediction["probabilities"],
                "processing_time": round(processing_time, 3),
                "audio_info": audio_info
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "prediction": "error",
                "confidence": 0.0,
                "probabilities": None,
                "processing_time": round(processing_time, 3),
                "error": str(e),
                "audio_info": None
            }
    
    def _predict_sync(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Synchronous prediction method
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if self.model is None:
                raise RuntimeError("Model not loaded")
            
            # Load and preprocess audio
            audio, sr = librosa.load(audio_file_path, sr=2000)
            
            # Ensure 5 seconds (10,000 samples at 2000 Hz)
            segment_length = 5 * 2000
            if len(audio) < segment_length:
                audio = np.pad(audio, (0, segment_length - len(audio)), mode='constant')
            segment = audio[:segment_length]
            
            # Extract features
            features = extract_features(segment, sr=2000)
            
            # Add batch dimension: (1, 32, 79, 6)
            features = np.expand_dims(features, axis=0)
            
            # Make prediction
            prediction = self.model.predict(features, verbose=0)
            predicted_class_idx = np.argmax(prediction, axis=1)[0]
            confidence = float(np.max(prediction))
            
            # Class labels
            class_labels = ['normal', 'abnormal', 'uncertain']
            predicted_class = class_labels[predicted_class_idx]
            
            # Get all probabilities
            probabilities = {
                class_labels[i]: float(prediction[0][i]) 
                for i in range(len(class_labels))
            }
            
            return {
                "class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities
            }
            
        except Exception as e:
            # Fallback to simple prediction function
            simple_prediction = predict_heart_sound(audio_file_path, self.model_path)
            return {
                "class": simple_prediction,
                "confidence": 0.0,
                "probabilities": None
            }
    
    async def _get_audio_info(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Get audio file information
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            loop = asyncio.get_event_loop()
            audio_info = await loop.run_in_executor(
                None,
                self._get_audio_info_sync,
                audio_file_path
            )
            return audio_info
        except Exception as e:
            return {
                "error": f"Could not get audio info: {str(e)}"
            }
    
    def _get_audio_info_sync(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Synchronous method to get audio information
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            # Load audio to get info
            audio, sr = librosa.load(audio_file_path, sr=None)  # Keep original sample rate
            
            duration = len(audio) / sr
            file_size = os.path.getsize(audio_file_path)
            
            return {
                "duration_seconds": round(duration, 2),
                "sample_rate": sr,
                "samples": len(audio),
                "file_size_bytes": file_size,
                "channels": 1  # librosa loads as mono by default
            }
        except Exception as e:
            return {
                "error": f"Could not analyze audio: {str(e)}"
            }
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        try:
            if self.model is None:
                return {
                    "status": "not_loaded",
                    "error": "Model not loaded"
                }
            
            # Get model info
            input_shape = self.model.input_shape if hasattr(self.model, 'input_shape') else None
            output_shape = self.model.output_shape if hasattr(self.model, 'output_shape') else None
            
            # Count parameters
            total_params = self.model.count_params() if hasattr(self.model, 'count_params') else None
            
            return {
                "status": "loaded",
                "model_path": self.model_path,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "total_parameters": total_params,
                "model_type": str(type(self.model).__name__),
                "class_labels": ["normal", "abnormal", "uncertain"]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the handler
        
        Returns:
            Dictionary with health status
        """
        return {
            "handler_status": "healthy",
            "model_loaded": self.is_model_loaded(),
            "model_path": self.model_path,
            "supported_formats": [".wav"],
            "max_file_size_mb": 10
        } 