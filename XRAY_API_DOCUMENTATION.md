# X-ray Pneumonia Prediction API Documentation

This document describes the X-ray pneumonia prediction endpoints added to the Medical Prediction API.

## Overview

The X-ray pneumonia prediction service uses a trained convolutional neural network to classify chest X-ray images as either "normal" or "pneumonia". The model was trained using the preprocessing pipeline described in `xRayKerasModel.py`.

## Model Details

- **Input Size**: 180x180 RGB images
- **Classes**: `normal`, `pneumonia`
- **Architecture**: CNN with binary classification (sigmoid output)
- **Preprocessing**: Resize to 180x180, RGB conversion, normalization (0-1 scale)

## API Endpoints

### 1. X-ray Pneumonia Prediction

**Endpoint**: `POST /api/v1/predict/xray-pneumonia`

**Description**: Predict pneumonia from an uploaded X-ray image.

**Request**:

- Method: POST
- Content-Type: multipart/form-data
- Body: Form data with image file

**Supported File Types**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`
**File Size Limit**: 10MB

**Example Request**:

```bash
curl -X POST "http://localhost:8000/api/v1/predict/xray-pneumonia" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "image_file=@chest_xray.jpg"
```

**Response**:

```json
{
  "success": true,
  "filename": "chest_xray.jpg",
  "file_size": 45678,
  "prediction": "pneumonia",
  "confidence": 0.87,
  "probability_pneumonia": 0.87,
  "probability_normal": 0.13,
  "processing_time": 0.234,
  "message": "X-ray pneumonia analysis completed successfully"
}
```

**Response Fields**:

- `success`: Boolean indicating if the prediction was successful
- `filename`: Original filename of the uploaded image
- `file_size`: Size of the uploaded file in bytes
- `prediction`: Predicted class (`"normal"` or `"pneumonia"`)
- `confidence`: Confidence score (0-1) for the prediction
- `probability_pneumonia`: Probability of pneumonia (0-1)
- `probability_normal`: Probability of normal (0-1)
- `processing_time`: Time taken for processing in seconds
- `message`: Status message

### 2. X-ray Model Information

**Endpoint**: `GET /api/v1/model-info/xray-pneumonia`

**Description**: Get information about the loaded X-ray pneumonia model.

**Response**:

```json
{
  "success": true,
  "model_info": {
    "model_loaded": true,
    "model_path": "xray_pneumonia_model.keras",
    "input_shape": [null, 180, 180, 3],
    "target_size": [180, 180],
    "model_type": "X-ray Pneumonia Classification",
    "classes": ["normal", "pneumonia"],
    "preprocessing": "Resize to 180x180, RGB conversion, normalization (0-1)"
  }
}
```

### 3. X-ray Service Health Check

**Endpoint**: `GET /api/v1/health/xray-pneumonia`

**Description**: Check the health status of the X-ray pneumonia prediction service.

**Response**:

```json
{
  "service": "X-ray Pneumonia Prediction",
  "status": "healthy",
  "model_path": "xray_pneumonia_model.keras",
  "model_loaded": true
}
```

## Error Responses

### File Type Error (400)

```json
{
  "detail": "Unsupported file type. Allowed types: .jpg, .jpeg, .png, .bmp, .tiff"
}
```

### File Size Error (400)

```json
{
  "detail": "File size too large. Maximum size is 10MB"
}
```

### Processing Error (500)

```json
{
  "detail": "Error processing X-ray image: [error details]"
}
```

## Usage Examples

### Python with requests

```python
import requests

# Predict pneumonia from X-ray
url = "http://localhost:8000/api/v1/predict/xray-pneumonia"
files = {"image_file": open("chest_xray.jpg", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Pneumonia probability: {result['probability_pneumonia']:.2f}")
```

### JavaScript with fetch

```javascript
const formData = new FormData();
formData.append("image_file", fileInput.files[0]);

fetch("/api/v1/predict/xray-pneumonia", {
  method: "POST",
  body: formData,
})
  .then((response) => response.json())
  .then((data) => {
    console.log("Prediction:", data.prediction);
    console.log("Confidence:", data.confidence);
    console.log("Pneumonia probability:", data.probability_pneumonia);
  });
```

## Setup and Installation

1. Ensure all dependencies are installed:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure the trained model file `xray_pneumonia_model.keras` is in the backend directory.

3. Start the server:

   ```bash
   uvicorn main:app --reload
   ```

4. Test the endpoint:
   ```bash
   python test_xray_prediction.py
   ```

## Model Training

The model was trained using the code in `xRayKerasModel.py` with the following key characteristics:

- **Dataset**: Chest X-ray images (NORMAL vs PNEUMONIA)
- **Input preprocessing**: Resize to 180x180, normalize pixel values to 0-1 range
- **Architecture**: Convolutional Neural Network with dropout regularization
- **Loss function**: Binary crossentropy
- **Optimizer**: RMSprop with learning rate 0.0001

## Notes

- The model expects RGB images and will automatically convert grayscale images to RGB
- Images are automatically resized to 180x180 pixels to match the training data
- The prediction threshold is 0.5 (probability > 0.5 = pneumonia, otherwise normal)
- Processing time varies based on image size and server performance
