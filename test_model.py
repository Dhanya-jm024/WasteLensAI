"""
RealWaste Classifier - Verification & Test Script
Executes inference on sample images from sampleInput/ and validates model accuracy.
"""

import os
import sys
import glob
import numpy as np
from PIL import Image

# Suppress TensorFlow verbose logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Model', 'waste_classifier.h5')
SAMPLE_INPUT_DIR = os.path.join(PROJECT_ROOT, 'sampleInput')
SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'sample_images')

CLASS_NAMES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]

def run_tests():
    print("=" * 92)
    print("  ♻️  RealWaste Deep Learning Model Verification (High Accuracy Test Suite)")
    print("=" * 92)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        sys.exit(1)
        
    print(f"📦 Loading Model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!\n")
    
    target_dir = SAMPLE_INPUT_DIR if os.path.exists(SAMPLE_INPUT_DIR) else SAMPLES_DIR
    print(f"📁 Evaluating Sample Inputs from: {os.path.relpath(target_dir, PROJECT_ROOT)}/\n")
    
    print(f"{'Input File':28s} | {'Target Class':20s} | {'Predicted':20s} | {'Confidence':10s} | Status")
    print("-" * 92)
    
    correct = 0
    total = len(CLASS_NAMES)
    
    for i, cls in enumerate(CLASS_NAMES, start=1):
        numbered_name = f"{i:02d}_{cls.replace(' ', '_')}.jpg"
        slug_name = f"{cls.lower().replace(' ', '_')}.jpg"
        
        path_numbered = os.path.join(target_dir, numbered_name)
        path_slug = os.path.join(target_dir, slug_name)
        
        if os.path.exists(path_numbered):
            img_path = path_numbered
            display_name = numbered_name
        elif os.path.exists(path_slug):
            img_path = path_slug
            display_name = slug_name
        else:
            alt_path = os.path.join(SAMPLES_DIR, slug_name)
            if os.path.exists(alt_path):
                img_path = alt_path
                display_name = slug_name
            else:
                print(f"⚠️ Sample image missing for {cls}")
                continue
            
        img = Image.open(img_path).convert('RGB').resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.expand_dims(arr, axis=0)
        
        predictions = model.predict(arr, verbose=0)[0]
        pred_idx = np.argmax(predictions)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(predictions[pred_idx]) * 100
        
        is_correct = (pred_class == cls)
        status = "✅ PASS" if is_correct else "❌ FAIL"
        if is_correct:
            correct += 1
            
        print(f"{display_name:28s} | {cls:20s} | {pred_class:20s} | {confidence:6.2f}%    | {status}")
        
    print("-" * 92)
    accuracy = (correct / total) * 100
    print(f"🎯 Overall Test Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    print("=" * 92)

if __name__ == '__main__':
    run_tests()

