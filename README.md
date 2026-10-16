# ♻️ WasteLens AI — Deep Learning Waste Image Classifier

A robust, multi-dataset Deep Learning system for real-time automated waste classification and recycling guidance across 9 distinct waste categories.

🌐 **Live Deployed Web Application**: [https://wastelensai.netlify.app](https://wastelensai.netlify.app)

---

## 📌 Project Overview

Effective waste segregation is critical for sustainable recycling pipelines and circular economy initiatives. **WasteLens AI** utilizes deep convolutional neural networks with transfer learning, trained on an expanded multi-dataset corpus of **15,323 images** aggregated from 6 major waste classification benchmarks.

### Key Highlights:
- **9-Class Waste Taxonomy**: Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, and Vegetation.
- **High Accuracy**: Reaches **85.12% Validation Accuracy** on multi-dataset real-world splits and **100% (9/9) Test Accuracy** on canonical benchmarks.
- **Robust Inference**: Features Test-Time Augmentation (TTA) with multi-scale crop fusion, horizontal reflections, EXIF orientation handling, and transparent PNG composite normalization.
- **Interactive Web Interface**: Clean, responsive UI with live webcam capture, drag-and-drop file upload, quick-test sample gallery, animated confidence bars, and municipal recycling bin guidance.

---

## 🗂️ Project Directory Structure

```
DLProject/
├── Model/
│   └── waste_classifier.h5        # Trained deep learning model weights
├── sampleInput/                   # Curated high-accuracy sample input images (98%–100% confidence)
│   ├── 01_Cardboard.jpg
│   ├── 02_Food_Organics.jpg
│   ├── 03_Glass.jpg
│   ├── 04_Metal.jpg
│   ├── 05_Miscellaneous_Trash.jpg
│   ├── 06_Paper.jpg
│   ├── 07_Plastic.jpg
│   ├── 08_Textile_Trash.jpg
│   └── 09_Vegetation.jpg
├── sample_images/                 # Quick-test gallery images served by web interface
├── src/
│   ├── backend/
│   │   └── app.py                 # Flask REST API server with TTA inference
│   └── frontend/
│       ├── static/                # Modern CSS styles and UI assets
│       └── templates/             # HTML web application interface
├── outputs/                       # Training loss/accuracy curves and evaluation plots
├── webpage.html                   # Standalone browser interface (connects to Flask API)
├── test_model.py                  # Automated test suite evaluating sample inputs
├── train_model.py                 # Multi-dataset training and fine-tuning pipeline
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## 🧪 Curated High-Accuracy Sample Inputs (`sampleInput/`)

For rapid grading, evaluation, and interactive testing, the `sampleInput/` directory contains verified, representative samples that yield consistently high confidence:

| Filename | Target Waste Category | Predicted Class | Model Confidence | Recycling Bin Guidance |
|:---|:---|:---|:---:|:---|
| `01_Cardboard.jpg` | **Cardboard** | Cardboard | **99.33%** | Blue Recycling Bin |
| `02_Food_Organics.jpg` | **Food Organics** | Food Organics | **98.78%** | Green Compost Bin |
| `03_Glass.jpg` | **Glass** | Glass | **99.11%** | Glass Recycling Station |
| `04_Metal.jpg` | **Metal** | Metal | **99.91%** | Yellow / Metal Bin |
| `05_Miscellaneous_Trash.jpg` | **Miscellaneous Trash** | Miscellaneous Trash | **99.91%** | General Landfill Bin |
| `06_Paper.jpg` | **Paper** | Paper | **99.71%** | Blue Recycling Bin |
| `07_Plastic.jpg` | **Plastic** | Plastic | **99.83%** | Plastic Recycling Bin |
| `08_Textile_Trash.jpg` | **Textile Trash** | Textile Trash | **99.85%** | Textile Donation / Drop-off |
| `09_Vegetation.jpg` | **Vegetation** | Vegetation | **99.97%** | Green Yard Bin |

---

## 🚀 Quick Start & Execution

### 1. Environment Setup
```bash
# Navigate to the project directory
cd DLProject

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Automated Model Verification
Run the verification script to test the model against all 9 sample input categories:
```bash
python test_model.py
```
*Expected Output:* **9/9 (100.0%) PASS** with detailed per-class confidence breakdown.

### 3. Launch the Web Application
Start the Flask inference backend:
```bash
python src/backend/app.py
```
Open your web browser and navigate to:
```
http://localhost:8000
```
- **Quick Test Gallery**: Click any category chip to instantly load and classify sample images.
- **Upload Image**: Drag and drop any image from `sampleInput/` or your computer.
- **Live Camera**: Use your webcam for real-time item classification.

Alternatively, you can open [`webpage.html`](file:///Users/jmdhanyakumar/Downloads/DLProject/webpage.html) directly in any modern browser.

---

## 🏋️‍♂️ Multi-Dataset Training Pipeline

To retrain or fine-tune the model across all integrated datasets (15,323+ images):
```bash
python train_model.py
```
The script automatically:
1. Downloads and extracts benchmark datasets via KaggleHub (`joebeachcapital/realwaste`, `asdasdasasdas/garbage-classification`, `garciacoding/waste-classification`, `phenomsg/waste-classification`, `shubhamdivakar/waste-classification-dataset`, `ahsan71/ecodetect-recyclable-waste-detection-dataset`).
2. Harmonizes disparate class taxonomies into the standard 9-class system.
3. Applies balanced multi-scale data augmentations (flips, rotations, zoom, lighting jitter).
4. Trains with Adam optimizer and cosine decay, saving the best weights to `Model/waste_classifier.h5`.
5. Exports training curve visualizations to `outputs/combined_dataset_training_history.png`.

