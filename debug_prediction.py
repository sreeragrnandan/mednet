import numpy as np
import keras
from predict import predict_heart_sound, extract_features, F1Score
import librosa

def debug_prediction(audio_file_path="a0007.wav"):
    """Debug the prediction pipeline step by step"""
    print("=== DEBUGGING HEART SOUND PREDICTION ===")
    
    try:
        # Step 1: Load model
        print("\n1. Loading model...")
        custom_objects = {
            'F1Score': F1Score,
            'f1_score': F1Score
        }
        
        try:
            model = keras.models.load_model("model_fold_1.keras", custom_objects=custom_objects)
            print("✓ Model loaded successfully")
        except Exception as load_error:
            print(f"⚠ Model load error, trying without compilation: {load_error}")
            model = keras.models.load_model("model_fold_1.keras", custom_objects=custom_objects, compile=False)
            print("✓ Model loaded without compilation")
        
        # Print model info
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        
        # Step 2: Load and preprocess audio
        print(f"\n2. Loading audio file: {audio_file_path}")
        audio, sr = librosa.load(audio_file_path, sr=2000)
        print(f"✓ Audio loaded - Shape: {audio.shape}, Sample rate: {sr}")
        
        # Ensure 5 seconds (10,000 samples at 2000 Hz)
        segment_length = 5 * 2000
        if len(audio) < segment_length:
            print(f"⚠ Audio too short ({len(audio)} samples), padding to {segment_length}")
            audio = np.pad(audio, (0, segment_length - len(audio)), mode='constant')
        segment = audio[:segment_length]
        print(f"✓ Audio segment prepared - Shape: {segment.shape}")
        
        # Step 3: Extract features
        print("\n3. Extracting features...")
        features = extract_features(segment, sr=2000)
        print(f"✓ Features extracted - Shape: {features.shape}")
        print(f"Feature stats - Min: {np.min(features):.3f}, Max: {np.max(features):.3f}, Mean: {np.mean(features):.3f}")
        
        # Check for NaN or infinite values
        if np.isnan(features).any():
            print("⚠ WARNING: NaN values found in features!")
        if np.isinf(features).any():
            print("⚠ WARNING: Infinite values found in features!")
        
        # Step 4: Add batch dimension and predict
        print("\n4. Making prediction...")
        features_batch = np.expand_dims(features, axis=0)
        print(f"Features batch shape: {features_batch.shape}")
        
        # Make prediction
        prediction = model.predict(features_batch, verbose=1)
        print(f"✓ Raw prediction output: {prediction}")
        print(f"Prediction shape: {prediction.shape}")
        
        # Step 5: Process prediction
        print("\n5. Processing prediction...")
        predicted_class_idx = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        class_labels = ['normal', 'abnormal', 'uncertain']
        predicted_class = class_labels[predicted_class_idx]
        
        # Get all probabilities
        probabilities = {
            class_labels[i]: float(prediction[0][i]) 
            for i in range(len(class_labels))
        }
        
        print(f"✓ Predicted class index: {predicted_class_idx}")
        print(f"✓ Predicted class: {predicted_class}")
        print(f"✓ Confidence: {confidence:.4f}")
        print(f"✓ All probabilities: {probabilities}")
        
        return {
            "class": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities,
            "raw_prediction": prediction.tolist()
        }
        
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    # Test with the sample audio file
    result = debug_prediction("a0007.wav") #Result should be normal
    result = debug_prediction("a0001.wav") #Result should be abnormal
    print(f"\n=== FINAL RESULT ===")
    print(result) 