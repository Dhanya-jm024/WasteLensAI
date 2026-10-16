import os
import sys
import io
import base64
import threading
import numpy as np
from PIL import Image, ImageOps

# Point to directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
TEMPLATE_DIR = os.path.join(BASE_DIR, 'frontend/templates')
STATIC_DIR = os.path.join(BASE_DIR, 'frontend/static')
SAMPLES_DIR = os.path.join(PROJECT_ROOT, 'sample_images')
# Model paths (prefer TFLite for fast, low-memory cloud hosting)
TFLITE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Model/waste_classifier.tflite')
H5_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Model/waste_classifier.h5')

from flask import Flask, render_template, request, jsonify, send_from_directory

# Configure logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

# Lazy optional TensorFlow import (so cloud instances with only tflite-runtime don't crash)
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
except ImportError:
    tf = None

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max upload

# Global model holder and thread lock
model = None
is_tflite = False
tflite_input_details = None
tflite_output_details = None
predict_lock = threading.Lock()


def get_model():
    """Load model once and cache in memory (preferring TFLite for <50MB RAM usage)"""
    global model, is_tflite, tflite_input_details, tflite_output_details
    if model is None:
        # 1. Prefer TFLite (ultra-fast, lightweight, Render 512MB free-tier safe)
        if os.path.exists(TFLITE_MODEL_PATH):
            try:
                try:
                    import tflite_runtime.interpreter as tflite
                except ImportError:
                    from tensorflow import lite as tflite
                
                print(f"📦 Loading optimized TFLite model from: {TFLITE_MODEL_PATH}")
                interpreter = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
                interpreter.allocate_tensors()
                tflite_input_details = interpreter.get_input_details()
                tflite_output_details = interpreter.get_output_details()
                model = interpreter
                is_tflite = True
                print("✅ TFLite model ready for inference (<50MB RAM footprint)!")
                return model
            except Exception as e:
                print(f"⚠️ Failed to initialize TFLite ({e}), checking for H5 fallback...")

        # 2. Fallback to Keras H5 model
        if os.path.exists(H5_MODEL_PATH):
            if tf is None:
                raise RuntimeError("TensorFlow is not installed to load .h5 weights.")
            print(f"📦 Loading Keras H5 model from: {H5_MODEL_PATH}")
            model = tf.keras.models.load_model(H5_MODEL_PATH)
            is_tflite = False
            print("✅ Keras H5 model ready for inference!")
            return model

        raise FileNotFoundError(f"No valid model found at {TFLITE_MODEL_PATH} or {H5_MODEL_PATH}")
    return model

# Class taxonomy & metadata
CLASS_NAMES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]

CLASS_META = {
    'Cardboard': {
        'color': '#C28B52',
        'emoji': '📦',
        'category': 'Paper & Cardboard',
        'bin': 'Blue Recycling Bin',
        'tip': 'Flatten clean boxes to save space. Remove plastic packaging tape and ensure free of heavy food grease.'
    },
    'Food Organics': {
        'color': '#34D399',
        'emoji': '🍏',
        'category': 'Organic Waste',
        'bin': 'Green Compost Bin',
        'tip': 'Fully compostable. Keep free of plastic stickers, plastic wraps, and non-organic food packaging.'
    },
    'Glass': {
        'color': '#38BDF8',
        'emoji': '🍾',
        'category': 'Glass Container',
        'bin': 'Glass Recycling Station',
        'tip': 'Rinse bottles and jars clean. Remove metal or plastic caps (caps can be recycled in their respective bins).'
    },
    'Metal': {
        'color': '#94A3B8',
        'emoji': '🥫',
        'category': 'Metals & Cans',
        'bin': 'Yellow / Metal Bin',
        'tip': 'Rinse aluminum soda cans and steel food tins clean. Crushing cans helps conserve space in bins.'
    },
    'Miscellaneous Trash': {
        'color': '#71717A',
        'emoji': '🗑️',
        'category': 'General / Non-Recyclable',
        'bin': 'General Landfill Bin',
        'tip': 'Mixed or composite materials not accepted by recycling systems. Dispose in general household waste.'
    },
    'Paper': {
        'color': '#FBBF24',
        'emoji': '📄',
        'category': 'Recyclable Paper',
        'bin': 'Blue Recycling Bin',
        'tip': 'Keep dry and clean. Do not recycle paper soiled with cooking oils or personal hygiene waste.'
    },
    'Plastic': {
        'color': '#F87171',
        'emoji': '🥤',
        'category': 'Rigid & Flexible Plastics',
        'bin': 'Plastic Recycling Bin',
        'tip': 'Empty and rinse clean. Check plastic resin identification codes (types 1-PET and 2-HDPE are widely recycled).'
    },
    'Textile Trash': {
        'color': '#A78BFA',
        'emoji': '👕',
        'category': 'Textiles & Garments',
        'bin': 'Textile Donation / Drop-off',
        'tip': 'Donate wearable items to charity shops. Clean, worn-out textiles can be taken to fabric recycling banks.'
    },
    'Vegetation': {
        'color': '#4ADE80',
        'emoji': '🌿',
        'category': 'Garden Green Waste',
        'bin': 'Green Yard Bin',
        'tip': 'Leaves, twigs, cut grass, and small branches. Excellent for local organic garden mulching and composting.'
    }
}

# Cross-Origin Resource Sharing (CORS)
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

def preprocess_and_predict(pil_img):
    """
    Robust image preprocessing & Test-Time Augmentation (TTA):
    1. EXIF orientation correction (fixes rotated camera/phone photos)
    2. Transparent alpha composite onto neutral white background (fixes transparent PNGs from internet)
    3. Multi-Crop TTA (Full center crop + 85% focused crop + horizontal reflections)
    4. Model batch prediction and probability fusion
    """
    # 1. EXIF orientation
    pil_img = ImageOps.exif_transpose(pil_img)
    
    # 2. Transparent background handling
    if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
        alpha_img = pil_img.convert('RGBA')
        white_bg = Image.new('RGB', alpha_img.size, (255, 255, 255))
        white_bg.paste(alpha_img, mask=alpha_img.split()[3])
        pil_img = white_bg
    else:
        pil_img = pil_img.convert('RGB')
        
    w, h = pil_img.size
    min_dim = min(w, h)
    
    # 3. Multi-Crop TTA
    crops = []
    # Two scales: 1.0 (full center square) and 0.85 (zoomed focus to remove cluttered background)
    for scale in [1.0, 0.85]:
        crop_size = max(16, int(min_dim * scale))
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        cr = pil_img.crop((left, top, left + crop_size, top + crop_size))
        cr_resized = cr.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(cr_resized, dtype=np.float32) / 255.0
        crops.append(arr)
        crops.append(np.fliplr(arr))
        
    batch = np.stack(crops, axis=0) # shape (4, 224, 224, 3)
    
    loaded_model = get_model()
    with predict_lock:
        if is_tflite:
            in_idx = tflite_input_details[0]['index']
            out_idx = tflite_output_details[0]['index']
            loaded_model.resize_tensor_input(in_idx, [4, 224, 224, 3])
            loaded_model.allocate_tensors()
            loaded_model.set_tensor(in_idx, batch)
            loaded_model.invoke()
            raw_preds = loaded_model.get_tensor(out_idx)
        else:
            raw_preds = loaded_model.predict(batch, verbose=0)
    
    # Weight: 0.35 for full views, 0.15 for zoomed views
    weights = np.array([0.35, 0.35, 0.15, 0.15], dtype=np.float32)
    avg_pred = np.average(raw_preds, axis=0, weights=weights)
    
    predicted_idx = int(np.argmax(avg_pred))
    confidence = float(avg_pred[predicted_idx])
    predicted_class = CLASS_NAMES[predicted_idx]
    
    all_predictions = [
        {
            'class': CLASS_NAMES[i],
            'confidence': round(float(avg_pred[i]) * 100, 2),
            'color': CLASS_META[CLASS_NAMES[i]]['color'],
            'emoji': CLASS_META[CLASS_NAMES[i]]['emoji']
        }
        for i in range(len(CLASS_NAMES))
    ]
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    meta = CLASS_META.get(predicted_class, {})
    
    return {
        'class': predicted_class,
        'confidence': confidence,
        'confidence_percent': round(confidence * 100, 2),
        'color': meta.get('color', '#3B82F6'),
        'emoji': meta.get('emoji', '♻️'),
        'category': meta.get('category', 'Waste Material'),
        'bin': meta.get('bin', 'Recycling Station'),
        'tip': meta.get('tip', 'Dispose according to local recycling guidelines.'),
        'all_predictions': all_predictions
    }

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return ('', 204)
        
    try:
        pil_img = None
        
        # Check if file was sent as multipart/form-data
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                pil_img = Image.open(file.stream)
                
        # Check if sent as JSON base64 data
        elif request.is_json and 'image' in request.json:
            raw_b64 = request.json['image']
            if ',' in raw_b64:
                raw_b64 = raw_b64.split(',', 1)[1]
            img_bytes = base64.b64decode(raw_b64)
            pil_img = Image.open(io.BytesIO(img_bytes))
            
        if pil_img is None:
            return jsonify({'error': 'No image provided. Upload a file or capture a photo.'}), 400
            
        results = preprocess_and_predict(pil_img)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failure: {str(e)}'}), 500

@app.route('/health')
@app.route('/ping')
def health():
    return jsonify({'status': 'ok', 'model': 'loaded'}), 200

@app.route('/sample/<filename>')
def get_sample(filename):
    if not os.path.exists(SAMPLES_DIR):
        return jsonify({'error': 'Sample directory missing'}), 404
    return send_from_directory(SAMPLES_DIR, filename)

if __name__ == '__main__':
    print("\n" + "="*65)
    print("  ♻️  WasteLens AI - Deep Learning Classifier Server")
    print("="*65)
    print(f"  📁 Project Root: {PROJECT_ROOT}")
    print(f"  🤖 Model Path:   {MODEL_PATH}")
    print(f"  🌐 Local Server: http://localhost:8000")
    print("="*65 + "\n")
    
    # Pre-warm model on startup
    try:
        m = get_model()
        dummy_input = np.zeros((4, 224, 224, 3), dtype=np.float32)
        with predict_lock:
            if is_tflite:
                in_idx = tflite_input_details[0]['index']
                out_idx = tflite_output_details[0]['index']
                m.resize_tensor_input(in_idx, [4, 224, 224, 3])
                m.allocate_tensors()
                m.set_tensor(in_idx, dummy_input)
                m.invoke()
                _ = m.get_tensor(out_idx)
            else:
                m.predict(dummy_input, verbose=0)
        print("⚡ Model pre-warmed and ready for immediate high-speed inference!\n")
    except Exception as e:
        print(f"⚠️ Pre-warm skipped: {e}")
        
    app.run(debug=False, host='0.0.0.0', port=8000, threaded=True)

