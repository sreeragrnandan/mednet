#!/usr/bin/env python3
"""
Test script for X-ray pneumonia prediction functionality
"""

import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path to import our handlers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from xray_handler import XRayPneumoniaHandler

async def test_xray_handler():
    """Test the X-ray pneumonia handler functionality"""
    print("Testing X-ray pneumonia prediction handler...")
    print("-" * 50)
    
    # Initialize handler
    handler = XRayPneumoniaHandler()
    
    # Test model loading
    print(f"Model loaded: {handler.is_model_loaded()}")
    if not handler.is_model_loaded():
        print("Warning: Model not loaded. Make sure xray_pneumonia_model.keras exists in the backend directory.")
        return
    
    # Test model info
    print("\nModel Info:")
    model_info = await handler.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Test health check
    print("\nHealth Check:")
    health = await handler.health_check()
    for key, value in health.items():
        print(f"  {key}: {value}")
    
    # If you have a test image, uncomment and modify the path below
    # test_image_path = "path/to/your/test/xray/image.jpg"
    # if os.path.exists(test_image_path):
    #     print(f"\nTesting prediction with {test_image_path}:")
    #     result = await handler.predict_xray(image_file_path=test_image_path)
    #     for key, value in result.items():
    #         print(f"  {key}: {value}")
    # else:
    #     print(f"\nTest image not found at {test_image_path}")
    
    print("\nTest completed!")

def test_model_requirements():
    """Test if all required files and dependencies are available"""
    print("Checking model requirements...")
    print("-" * 30)
    
    # Check if model file exists
    model_path = "xray_pneumonia_model.keras"
    model_exists = os.path.exists(model_path)
    print(f"Model file ({model_path}): {'✓ Found' if model_exists else '✗ Not found'}")
    
    # Check required packages
    required_packages = [
        'tensorflow',
        'PIL',
        'numpy',
        'asyncio'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"Package {package}: ✓ Installed")
        except ImportError:
            print(f"Package {package}: ✗ Not installed")
    
    return model_exists

if __name__ == "__main__":
    print("X-ray Pneumonia Prediction Test")
    print("=" * 50)
    
    # Test requirements first
    requirements_ok = test_model_requirements()
    print()
    
    if requirements_ok:
        # Run async test
        asyncio.run(test_xray_handler())
    else:
        print("Some requirements are missing. Please install required packages and ensure the model file is present.")
        print("\nTo install required packages:")
        print("pip install -r requirements.txt")
        print("\nMake sure xray_pneumonia_model.keras is in the backend directory.") 