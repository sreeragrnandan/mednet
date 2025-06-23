import os
import warnings

# Suppress warnings (must be set before importing TensorFlow/Keras)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN warnings
warnings.filterwarnings('ignore')

import numpy as np
import librosa
import keras

# Custom F1Score metric class (matching training implementation)
class F1Score(keras.metrics.Metric):
    """Custom F1 Score metric implementation"""
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = keras.metrics.Precision()
        self.recall = keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + keras.backend.epsilon()))
    
    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

def extract_features(audio_segment, sr=2000):
    """Extract features exactly as done in heart_sound_classifier.py"""
    # Configuration parameters (matching heart_sound_classifier.py)
    n_mels = 32
    n_mfcc = 20
    n_fft = 512
    hop_length = 128
    n_frames = 79  # Expected time frames: int(5.0 * 2000 / 128) + 1 = 79
    
    # Extract all features
    # Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_segment,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=20,  # Minimum frequency for heart sounds
        fmax=800  # Maximum frequency for heart sounds
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # MFCCs
    mfccs = librosa.feature.mfcc(
        y=audio_segment,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=20,
        fmax=800
    )
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(
        y=audio_segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        fmin=20
    )
    
    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio_segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(
        y=audio_segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio_segment,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Ensure all features have the same dimensions (32, 79)
    target_shape = (n_mels, n_frames)
    
    # Reshape each feature to target shape
    mel_spec_db = librosa.util.fix_length(mel_spec_db, size=target_shape[1], axis=1)
    mfccs = librosa.util.fix_length(mfccs, size=target_shape[1], axis=1)
    contrast = librosa.util.fix_length(contrast, size=target_shape[1], axis=1)
    centroid = librosa.util.fix_length(centroid, size=target_shape[1], axis=1)
    bandwidth = librosa.util.fix_length(bandwidth, size=target_shape[1], axis=1)
    rolloff = librosa.util.fix_length(rolloff, size=target_shape[1], axis=1)
    
    # Ensure all features have the same number of frequency bins
    mel_spec_db = librosa.util.fix_length(mel_spec_db, size=target_shape[0], axis=0)
    mfccs = librosa.util.fix_length(mfccs, size=target_shape[0], axis=0)
    contrast = librosa.util.fix_length(contrast, size=target_shape[0], axis=0)
    centroid = librosa.util.fix_length(centroid, size=target_shape[0], axis=0)
    bandwidth = librosa.util.fix_length(bandwidth, size=target_shape[0], axis=0)
    rolloff = librosa.util.fix_length(rolloff, size=target_shape[0], axis=0)
    
    # Stack features along the channel dimension (32, 79, 6)
    combined_features = np.stack([
        mel_spec_db,
        mfccs,
        contrast,
        centroid,
        bandwidth,
        rolloff
    ], axis=-1)
    
    return combined_features

def predict_heart_sound(wav_path, model_path="model_fold_1.keras"):
    """Predict heart sound using the trained Keras model"""
    try:
        # Try to load the trained model with custom objects
        custom_objects = {
            'F1Score': F1Score,
            'f1_score': F1Score
        }
        
        try:
            model = keras.models.load_model(model_path, custom_objects=custom_objects)
        except Exception as load_error:
            # Try loading without compilation
            model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        
        # Load audio
        audio, sr = librosa.load(wav_path, sr=2000)
        
        # Ensure 5 seconds (10,000 samples at 2000 Hz)
        segment_length = 5 * 2000  # 10,000 samples
        if len(audio) < segment_length:
            audio = np.pad(audio, (0, segment_length - len(audio)), mode='constant')
        segment = audio[:segment_length]
        
        # Extract features using the same method as training
        features = extract_features(segment, sr=2000)
        
        # Add batch dimension: (1, 32, 79, 6)
        features = np.expand_dims(features, axis=0)
        
        # Make prediction
        prediction = model.predict(features, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)
        
        # Class labels (matching the training setup)
        class_labels = ['normal', 'abnormal', 'uncertain']
        
        return class_labels[predicted_class]
        
    except Exception as e:
        # Fallback to error if model loading or prediction fails
        return "error"

if __name__ == "__main__":
    result = predict_heart_sound("a0007.wav", "model_fold_1.keras")
    print(result) 