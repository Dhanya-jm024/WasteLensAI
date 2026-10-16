"""
WasteLens AI - Multi-Dataset Training Pipeline
Trains MobileNetV2 on combined real-world waste datasets:
1. Kaggle Garbage Classification (asdasdasasdas/garbage-classification)
2. RealWaste Dataset (joebeachcapital/realwaste)
Total: >7,200 images across 9 waste categories.
"""

import os
import sys
import glob
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, 'Model', 'waste_classifier.h5')
HISTORY_PLOT_PATH = os.path.join(PROJECT_ROOT, 'outputs', 'combined_dataset_training_history.png')

CANONICAL_CLASSES = [
    'Cardboard', 'Food Organics', 'Glass', 'Metal', 
    'Miscellaneous Trash', 'Paper', 'Plastic', 'Textile Trash', 'Vegetation'
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CANONICAL_CLASSES)}

# Mapping various directory names to canonical class names
CLASS_ALIAS_MAP = {
    'cardboard': 'Cardboard',
    'glass': 'Glass',
    'metal': 'Metal',
    'paper': 'Paper',
    'plastic': 'Plastic',
    'trash': 'Miscellaneous Trash',
    'food organics': 'Food Organics',
    'food_organics': 'Food Organics',
    'foodorganics': 'Food Organics',
    'textile trash': 'Textile Trash',
    'textile_trash': 'Textile Trash',
    'textiletrash': 'Textile Trash',
    'textiles': 'Textile Trash',
    'vegetation': 'Vegetation',
    'miscellaneous trash': 'Miscellaneous Trash',
    'miscellaneous_trash': 'Miscellaneous Trash',
    'miscellaneoustrash': 'Miscellaneous Trash',
    
    # PhenomSG & GarciaCoding categories
    'cans_all_type': 'Metal',
    'plastic_bottles': 'Plastic',
    'paper_products': 'Paper',
    'glass_containers': 'Glass',
    'egg_shells': 'Food Organics',
    'kitchen_waste': 'Food Organics',
    'food_scraps': 'Food Organics',
    'coffee_tea_bags': 'Food Organics',
    'yard_trimmings': 'Vegetation',
    'platics_bags_wrappers': 'Plastic',
    'stroform_product': 'Miscellaneous Trash',
    'ceramic_product': 'Miscellaneous Trash',
    'sanitary_napkin': 'Miscellaneous Trash',
    'diapers': 'Miscellaneous Trash',
    'batteries': 'Miscellaneous Trash',
    'paints': 'Miscellaneous Trash',
    'e-waste': 'Miscellaneous Trash',
    'pesticides': 'Miscellaneous Trash'
}

def collect_images_from_directory(base_dir, dataset_name):
    """Recursively search for class folders and collect (file_path, class_idx) tuples"""
    samples = []
    if not os.path.exists(base_dir):
        print(f"⚠️ Directory not found: {base_dir}")
        return samples

    for root, dirs, files in os.walk(base_dir):
        folder_name = os.path.basename(root).lower().strip()
        canonical = CLASS_ALIAS_MAP.get(folder_name)
        if canonical:
            cls_idx = CLASS_TO_IDX[canonical]
            valid_files = [
                os.path.join(root, f) for f in files 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) and not f.startswith('.')
            ]
            for vf in valid_files:
                samples.append((vf, cls_idx, canonical))
                
    print(f"✓ Found {len(samples)} valid images in {dataset_name}")
    return samples

def build_data_generator(samples, batch_size=32, target_size=(224, 224), augment=True, class_weights=None):
    """Custom streaming generator that performs EXIF rotation, smart crop, and yields sample weights"""
    num_samples = len(samples)
    indices = np.arange(num_samples)
    
    while True:
        if augment:
            np.random.shuffle(indices)
            
        for offset in range(0, num_samples, batch_size):
            batch_indices = indices[offset:offset + batch_size]
            batch_images = []
            batch_labels = []
            batch_weights = []
            
            for i in batch_indices:
                img_path, cls_idx, _ = samples[i]
                try:
                    with Image.open(img_path) as pil_img:
                        pil_img = ImageOps.exif_transpose(pil_img)
                        
                        # Handle transparency
                        if pil_img.mode in ('RGBA', 'LA') or (pil_img.mode == 'P' and 'transparency' in pil_img.info):
                            alpha_img = pil_img.convert('RGBA')
                            bg = Image.new('RGB', alpha_img.size, (255, 255, 255))
                            bg.paste(alpha_img, mask=alpha_img.split()[3])
                            pil_img = bg
                        else:
                            pil_img = pil_img.convert('RGB')
                            
                        w, h = pil_img.size
                        min_dim = min(w, h)
                        left = (w - min_dim) // 2
                        top = (h - min_dim) // 2
                        cropped = pil_img.crop((left, top, left + min_dim, top + min_dim))
                        resized = cropped.resize(target_size, Image.Resampling.BILINEAR)
                        
                        arr = np.array(resized, dtype=np.float32) / 255.0
                        
                        if augment and np.random.rand() > 0.5:
                            arr = np.fliplr(arr)
                                
                        batch_images.append(arr)
                        batch_labels.append(cls_idx)
                        if class_weights is not None:
                            batch_weights.append(class_weights.get(cls_idx, 1.0))
                except Exception:
                    continue
                    
            if len(batch_images) > 0:
                imgs = np.array(batch_images, dtype=np.float32)
                lbls = np.array(batch_labels, dtype=np.int32)
                if class_weights is not None:
                    wts = np.array(batch_weights, dtype=np.float32)
                    yield imgs, lbls, wts
                else:
                    yield imgs, lbls

def train(epochs=6, batch_size=32, lr=0.0003):
    print("=" * 70)
    print("  🚀 WasteLens AI - Multi-Dataset CNN Training Pipeline")
    print("=" * 70)

    # 1. Dataset directories & loaders
    realwaste_dir = os.path.expanduser("~/.cache/kagglehub/datasets/joebeachcapital/realwaste/versions/1")
    kaggle_dir = os.path.expanduser("~/.cache/kagglehub/datasets/asdasdasasdas/garbage-classification/versions/2")
    garciacoding_dir = os.path.expanduser("~/.cache/kagglehub/datasets/garciacoding/waste-classification/versions/2")
    phenomsg_dir = os.path.expanduser("~/.cache/kagglehub/datasets/phenomsg/waste-classification/versions/1")
    shubham_base = os.path.expanduser("~/.cache/kagglehub/datasets/shubhamdivakar/waste-classification-dataset/versions/1")
    ecodetect_base = os.path.expanduser("~/.cache/kagglehub/datasets/ahsan71/ecodetect-recyclable-waste-detection-dataset/versions/5")
    
    all_samples = []
    all_samples.extend(collect_images_from_directory(realwaste_dir, "RealWaste Dataset"))
    all_samples.extend(collect_images_from_directory(kaggle_dir, "Kaggle Garbage Classification"))
    all_samples.extend(collect_images_from_directory(garciacoding_dir, "GarciaCoding Waste"))
    all_samples.extend(collect_images_from_directory(phenomsg_dir, "PhenomSG Waste"))
    
    # 5. ShubhamDivakar: Balanced sampling of Food Organics + E-waste
    if os.path.exists(shubham_base):
        o_files = glob.glob(os.path.join(shubham_base, 'TRAIN/O/*.jpg'))
        import random
        random.seed(42)
        sampled_o = random.sample(o_files, min(1200, len(o_files)))
        for f in sampled_o:
            all_samples.append((f, CLASS_TO_IDX['Food Organics'], 'Food Organics'))
        e_files = glob.glob(os.path.join(shubham_base, 'TRAIN/E/*.jpg'))
        for f in e_files:
            all_samples.append((f, CLASS_TO_IDX['Miscellaneous Trash'], 'Miscellaneous Trash'))
        print(f"✓ Found {len(sampled_o) + len(e_files)} valid images in ShubhamDivakar (1200 Organics + 124 E-waste)")

    # 6. EcoDetect: Single-class annotated images (Metal, Paper, Plastic)
    if os.path.exists(ecodetect_base):
        eco_count = 0
        eco_map = {0: 'Metal', 1: 'Paper', 2: 'Plastic'}
        for split in ['train', 'valid', 'test']:
            lbl_dir = os.path.join(ecodetect_base, split, 'labels')
            img_dir = os.path.join(ecodetect_base, split, 'images')
            if not os.path.exists(lbl_dir): continue
            for lf in glob.glob(os.path.join(lbl_dir, '*.txt')):
                with open(lf) as f:
                    cids = set(l.strip().split()[0] for l in f if l.strip())
                    if len(cids) == 1:
                        cid = int(list(cids)[0])
                        if cid in eco_map:
                            target_cls = eco_map[cid]
                            stem = os.path.splitext(os.path.basename(lf))[0]
                            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                                candidate = os.path.join(img_dir, stem + ext)
                                if os.path.exists(candidate):
                                    all_samples.append((candidate, CLASS_TO_IDX[target_cls], target_cls))
                                    eco_count += 1
                                    break
        print(f"✓ Found {eco_count} valid images in EcoDetect (Single-Class Images)")
    
    if len(all_samples) == 0:
        print("❌ No images found to train on. Check dataset paths.")
        return

    # Class distribution summary
    class_counts = {c: 0 for c in CANONICAL_CLASSES}
    for _, _, canonical in all_samples:
        class_counts[canonical] += 1
        
    print("\n📊 Combined Class Distribution:")
    print("-" * 45)
    for cls in CANONICAL_CLASSES:
        print(f"  {cls:22s}: {class_counts[cls]:5d} images")
    print("-" * 45)
    print(f"  Total Images: {len(all_samples):5d}\n")

    # 2. Train / Val split (85% train, 15% validation)
    labels = [s[1] for s in all_samples]
    train_samples, val_samples = train_test_split(all_samples, test_size=0.15, stratify=labels, random_state=42)
    print(f"✓ Train set: {len(train_samples)} images | Val set: {len(val_samples)} images")

    # 3. Class weights
    train_labels = [s[1] for s in train_samples]
    unique_classes = np.unique(train_labels)
    class_weights_arr = compute_class_weight('balanced', classes=unique_classes, y=train_labels)
    class_weight_dict = {int(cls): float(weight) for cls, weight in zip(unique_classes, class_weights_arr)}

    # 4. Load or build model
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\n📦 Loading existing model from {MODEL_SAVE_PATH} to fine-tune...")
        model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    else:
        print("\n🔨 Building new MobileNetV2 Transfer Learning Model...")
        base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base.layers:
            layer.trainable = False
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        out = tf.keras.layers.Dense(len(CANONICAL_CLASSES), activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base.input, outputs=out)

    # Compile with Adam
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 5. Data generators with per-sample class weights
    train_gen = build_data_generator(train_samples, batch_size=batch_size, augment=True, class_weights=class_weight_dict)
    val_gen = build_data_generator(val_samples, batch_size=batch_size, augment=False, class_weights=None)
    
    steps_per_epoch = len(train_samples) // batch_size
    val_steps = max(1, len(val_samples) // batch_size)

    # 6. Callbacks
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_PLOT_PATH), exist_ok=True)
    
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    print("\n🎯 Starting Training...")
    start_time = time.time()
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    elapsed = time.time() - start_time
    print(f"\n🎉 Training finished in {elapsed/60:.2f} minutes!")
    
    # Save the final / restored best weights
    model.save(MODEL_SAVE_PATH)
    print(f"✓ Model saved to {MODEL_SAVE_PATH}")

    # 7. Plot and save training history
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='#10b981', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', color='#06b6d4', linewidth=2)
    plt.title('Combined Model Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='#f59e0b', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val Loss', color='#ef4444', linewidth=2)
    plt.title('Combined Model Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_PATH, dpi=200)
    print(f"✓ Training plot saved to {HISTORY_PLOT_PATH}")
    print(f"✓ Best model checkpoint saved to {MODEL_SAVE_PATH}")
    print("=" * 70)

if __name__ == '__main__':
    train()
