import asyncio
import time
from typing import Dict, Any, Optional
import os
import numpy as np
from predict import predict_heart_sound, extract_features
import librosa
import keras
import gdown
import uuid
from datetime import datetime

class HeartSoundHandler:
    """Handler class for heart sound prediction operations"""
    
    def __init__(self, model_path: str = "model_fold_1.keras", model_url: str = None):
        """
        Initialize the handler with model path and optional Google Drive URL
        
        Args:
            model_path: Path to save/load the trained Keras model
            model_url: Optional Google Drive URL to download the model from
        """
        self.model_path = model_path
        self.model_url = model_url
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
            
            # Download model from Google Drive if URL is provided and model doesn't exist
            if self.model_url and not os.path.exists(self.model_path):
                print(f"Downloading model from {self.model_url}")
                gdown.download(self.model_url, self.model_path, quiet=False)
            
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
    
    def _calculate_risk_level(self, probabilities: Dict[str, float]) -> tuple[str, float]:
        """
        Calculate risk level and score based on prediction probabilities
        
        Args:
            probabilities: Dictionary of class probabilities
            
        Returns:
            Tuple of (risk_level, risk_score)
        """
        # Calculate risk score (0-100)
        risk_score = (probabilities.get('abnormal', 0) * 100 + 
                     probabilities.get('uncertain', 0) * 50)
        
        # Determine risk level
        if risk_score < 30:
            risk_level = "low"
        elif risk_score < 70:
            risk_level = "moderate"
        else:
            risk_level = "high"
            
        return risk_level, risk_score

    def _calculate_audio_metrics(self, audio_segment: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Calculate various audio quality and characteristic metrics
        
        Args:
            audio_segment: Audio signal array
            sr: Sample rate
            
        Returns:
            Dictionary of audio metrics
        """
        try:
            # Calculate rhythm metric using tempo detection
            tempo, _ = librosa.beat.beat_track(y=audio_segment, sr=sr)
            rhythm_score = min(100, max(0, 100 - abs(tempo - 60) / 2))  # Normalize around 60 BPM
            
            # Calculate clarity using signal-to-noise ratio estimation
            noise_floor = np.mean(np.abs(audio_segment[audio_segment < np.mean(audio_segment)]))
            signal_power = np.mean(np.abs(audio_segment))
            clarity_score = min(100, max(0, 100 * (signal_power / (noise_floor + 1e-6))))
            
            # Calculate irregularity using zero crossing rate
            zero_crossings = librosa.zero_crossings(audio_segment)
            irregularity_score = min(100, max(0, 100 * np.mean(zero_crossings)))
            
            # Calculate murmur likelihood using spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr)
            murmur_score = min(100, max(0, np.mean(rolloff) / 100))
            
            return {
                "irregularity": round(irregularity_score / 100, 3),
                "murmur": round(murmur_score / 100, 3),
                "clarity": round(clarity_score / 100, 3),
                "rhythm": round(rhythm_score / 100, 3)
            }
            
        except Exception as e:
            print(f"Error calculating audio metrics: {e}")
            return {
                "irregularity": 0.0,
                "murmur": 0.0,
                "clarity": 0.0,
                "rhythm": 0.0
            }

    async def predict_audio(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Predict heart sound classification for an audio file
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            Dictionary containing detailed prediction results
        """
        start_time = time.time()
        
        try:
            # Validate file exists
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Load audio
            audio, sr = librosa.load(audio_file_path, sr=2000)
            
            # Run prediction in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            prediction = await loop.run_in_executor(
                None, 
                self._predict_sync, 
                audio_file_path
            )
            
            # Calculate risk level and score
            risk_level, risk_score = self._calculate_risk_level(prediction["probabilities"])
            
            # Calculate audio metrics
            metrics = self._calculate_audio_metrics(audio, sr)
            
            # Generate notes based on findings
            notes = self._generate_notes(prediction["class"], risk_level, metrics)
            

            
            processing_time = time.time() - start_time
            
            # Generate analysis ID with timestamp and UUID
            timestamp_ms = int(time.time() * 1000)
            analysis_id = f"analysis-{timestamp_ms}-{str(uuid.uuid4())[:12]}"
            
            return {
                "id": analysis_id,
                "timestamp": timestamp_ms,
                "riskLevel": risk_level,
                "riskScore": round(risk_score / 100, 3),  # Convert to decimal (0-1)
                "confidence": round(prediction["confidence"], 3),  # Keep as decimal (0-1)
                "metrics": metrics,
                "notes": notes
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Generate analysis ID with timestamp and UUID for error case
            timestamp_ms = int(time.time() * 1000)
            analysis_id = f"analysis-{timestamp_ms}-{str(uuid.uuid4())[:12]}"
            
            return {
                "id": analysis_id,
                "timestamp": timestamp_ms,
                "riskLevel": "error",
                "riskScore": 0.0,
                "confidence": 0.0,
                "metrics": {
                    "irregularity": 0.0,
                    "murmur": 0.0,
                    "clarity": 0.0,
                    "rhythm": 0.0
                },
                "notes": f"Error during prediction: {str(e)}"
            }

    def _generate_notes(self, prediction_class: str, risk_level: str, metrics: Dict[str, float]) -> str:
        """Generate descriptive notes based on the analysis results"""
        notes = []
        
        # Add prediction class note
        notes.append(f"Heart sound classified as {prediction_class}.")
        
        # Add risk level note
        risk_notes = {
            "low": "No significant abnormalities detected.",
            "moderate": "Some irregular patterns detected, follow-up recommended.",
            "high": "Significant abnormalities detected, immediate medical attention recommended."
        }
        notes.append(risk_notes.get(risk_level, ""))
        
        # Add specific metric notes
        if metrics["murmur"] > 0.5:
            notes.append("Potential heart murmur detected.")
        if metrics["irregularity"] > 0.5:
            notes.append("Irregular heart rhythm patterns observed.")
        if metrics["clarity"] < 0.5:
            notes.append("Note: Recording quality is suboptimal.")
        
        return " ".join(notes)
    
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