# Project Documentation: Medical Image Analysis Platform

This document explains all the code and files in this project so you can understand how everything works and explain it to others.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [File-by-File Explanation](#file-by-file-explanation)
3. [How the Application Works](#how-the-application-works)
4. [Key Concepts Explained](#key-concepts-explained)
5. [Files You Can Ignore](#files-you-can-ignore)

---

## Project Overview

This project is a web application that:
1. Takes a chest X-ray image as input
2. Lets you select a region of interest (like a suspicious area)
3. Uses AI to automatically segment (outline) structures in that region
4. Generates a written diagnostic report describing what the AI sees

**Two main AI models are used:**
- **SAM (Segment Anything Model)** by Meta: Does the visual segmentation (drawing outlines around objects)
- **BLIP** by Salesforce: Generates text descriptions of what it sees in images

---

## File-by-File Explanation

### Root Directory Files

#### `config.py`
**Purpose**: Central configuration file for all settings.

**What it contains**:
- `PROJECT_ROOT`: Base path of the project
- `DATA_DIR`, `MODELS_DIR`, `OUTPUT_DIR`: Where data, models, and outputs are stored
- `SAM_CONFIG`: Settings for SAM model (which version, download URL)
- `BLIP_CONFIG`: Settings for BLIP model (model name, max text length)
- `DEVICE`: Automatically detects if you have a GPU or need to use CPU
- `IMAGE_CONFIG`: Image sizes and supported formats
- `UI_CONFIG`: Colors for the interface
- `NIH_DATASET_CONFIG`: The 14 disease categories in the dataset

**Key function**: `get_device()` checks if CUDA (GPU) is available and returns the appropriate device.

---

#### `requirements.txt`
**Purpose**: Lists all Python packages needed to run the project.

**Key packages**:
- `torch`, `torchvision`: PyTorch deep learning framework
- `segment-anything`: Meta's SAM model
- `transformers`: HuggingFace library for BLIP
- `gradio`: Creates the web interface
- `opencv-python`: Image processing
- `numpy`, `pandas`: Data handling

---

#### `requirements.lock`
**Purpose**: Records the EXACT versions of every package installed in your environment.

**Why it matters**: If someone else installs from this file, they get the exact same versions, avoiding compatibility issues.

---

#### `setup.py`
**Purpose**: Automated setup script that:
1. Installs all requirements
2. Downloads SAM model weights (375 MB file)
3. Verifies everything is installed correctly

**How to use**: Run `python setup.py --all` to do everything, or use individual flags like `--download-weights`.

---

#### `Dockerfile` and `docker-compose.yml`
**Purpose**: Configuration for running the app in a Docker container.

**When to use**: If you want to deploy the app on a server or share it with someone who doesn't want to install Python packages manually.

---

### The `app/` Directory

#### `app/app.py`
**Purpose**: This is the MAIN file that runs the web application.

**Key components**:

1. **State Management (`AnalysisState` class)**:
   - Keeps track of the current image, segmentation mask, and selection
   - `rect_start` and `rect_end`: Store the two corners of the user's rectangle selection
   - `selection_step`: Tracks whether user is clicking first or second corner

2. **Image Processing Functions**:
   - `process_uploaded_image()`: Called when user uploads an image. Applies enhancement and prepares it for SAM.
   - `draw_selection_rectangle()`: Draws the Windows-style selection rectangle with darkened background.

3. **Segmentation Functions**:
   - `handle_segmentation_click()`: Called when user clicks on the image. First click = start corner, second click = end corner + run segmentation.
   - `clear_segmentation()`: Resets the selection to start over.

4. **Report Functions**:
   - `generate_report()`: Calls BLIP to generate diagnostic text for the segmented region.
   - `export_results()`: Packages segmented image, comparison view, and report for download.

5. **UI Creation (`create_app()`)**:
   - Uses Gradio to build the interface
   - Three columns: Upload, Segmentation, Results
   - Custom CSS (`CUSTOM_CSS`) creates the dark medical theme

6. **Event Handlers**:
   - Connect UI elements to functions (e.g., when image uploaded, call `process_uploaded_image`)

---

### The `src/` Directory

This contains all the "backend" logic separated into modules.

---

#### `src/preprocessing/image_loader.py`
**Purpose**: Functions to load images from files.

**Key functions**:
- `load_image()`: Basic image loading from file path
- `load_dicom()`: Loads medical DICOM format files
- `load_xray()`: Specialized loader for X-rays that applies CLAHE enhancement
- `load_from_array()`: Processes images that come from the Gradio upload (already in memory as numpy arrays)
- `apply_clahe_enhancement()`: Improves contrast in medical images

**What is CLAHE?**: Contrast Limited Adaptive Histogram Equalization. It's a technique that improves local contrast in images, making details more visible in X-rays.

---

#### `src/preprocessing/transforms.py`
**Purpose**: Image transformation functions.

**Key functions**:
- `normalize_xray()`: Scales pixel values to expected ranges for AI models
- `apply_windowing()`: Simulates the "window/level" controls radiologists use to adjust brightness/contrast
- `resize_for_sam()`: Resizes images to 1024x1024 (SAM's expected input size) while preserving aspect ratio
- `transform_coordinates()`: Converts coordinates between original image size and SAM size
- `prepare_for_display()`: Resizes images for showing in the UI (512x512)

---

#### `src/segmentation/sam_predictor.py`
**Purpose**: Wrapper around Meta's SAM model.

**Key class**: `SAMSegmenter`

**Key methods**:
- `__init__()`: Loads the SAM model and downloads weights if needed
- `set_image()`: Gives SAM an image to analyze (computes internal "embedding")
- `segment_point()`: Segments based on clicked points (foreground/background)
- `segment_box()`: Segments based on a bounding box (what we use)
- `get_best_mask()`: Returns the single best segmentation from multiple options
- `get_mask_overlay()`: Creates a colored overlay to show the segmentation

**How SAM works**:
1. You give it an image
2. It creates an "embedding" (internal representation)
3. You give it a "prompt" (points or box)
4. It outputs one or more possible segmentation masks
5. Each mask has a confidence score

---

#### `src/report_generation/blip_reporter.py`
**Purpose**: Generates text descriptions using the BLIP vision-language model.

**Key class**: `ReportGenerator`

**Key methods**:
- `__init__()`: Loads BLIP model from HuggingFace
- `generate_caption()`: Basic function - give it an image, get back a text description
- `generate_medical_report()`: Creates a structured report with multiple sections
- `_extract_roi()`: Crops the image to just the segmented region
- `_enhance_medical_terminology()`: Replaces casual terms with medical ones (e.g., "white area" -> "hyperdense opacity")
- `_generate_recommendations()`: Adds clinical recommendations based on findings
- `format_report()`: Formats the report dictionary into readable text

**Important**: When a mask is provided, the report ONLY analyzes the masked region, not the whole image. This was a key fix we made.

---

#### `src/utils/visualization.py`
**Purpose**: Functions for creating visual outputs.

**Key functions**:
- `overlay_mask()`: Draws a colored overlay on the image showing the segmentation
- `create_comparison_view()`: Creates side-by-side before/after images
- `draw_point_marker()`: Draws circular markers at click points
- `create_heatmap_overlay()`: For attention visualization (not currently used in UI)
- `add_text_annotation()`: Adds text labels to images

---

#### `src/utils/metrics.py`
**Purpose**: Evaluation metrics for measuring segmentation quality.

**Key functions**:
- `calculate_dice_score()`: Measures overlap between predicted and ground truth masks (0-1, higher is better)
- `calculate_iou()`: Intersection over Union - another overlap metric
- `calculate_precision_recall()`: How many predicted pixels are correct, how many actual pixels were found
- `calculate_mask_stats()`: Statistics about a mask (number of regions, area, centroid)

**What is Dice Score?**: If you have two shapes (predicted vs actual), Dice measures how much they overlap. Formula: 2 * (intersection) / (sum of both areas). Perfect overlap = 1.0.

---

### The `notebooks/` Directory

#### `dataset_exploration.py`
**Purpose**: Script to explore and understand the NIH dataset.

**What it does**:
- Loads the dataset labels CSV
- Analyzes distribution of diseases
- Displays sample X-ray images
- Shows preprocessing pipeline effects

---

#### `model_demo.py`
**Purpose**: Demonstrates how the SAM and BLIP models work.

**What it does**:
- Loads both models
- Shows step-by-step segmentation
- Demonstrates report generation
- Can be run standalone to test models

---

### The `data/` Directory

#### `data/raw/`
- `images/`: Where you put the X-ray PNG files (5000 files in your case)
- `Data_Entry_2017.csv`: Labels file from NIH dataset

#### `data/processed/`
- Where processed outputs would be saved (currently empty)

---

### The `models/` Directory

#### `models/weights/sam_vit_b_01ec64.pth`
**Purpose**: The pre-trained SAM model weights (375 MB file).

**How it got there**: Downloaded automatically by `setup.py`.

---

## How the Application Works

### Step-by-Step Flow

1. **User uploads image** (app.py: `process_uploaded_image()`)
   - Image is converted to numpy array by Gradio
   - CLAHE enhancement applied (improves contrast)
   - Resized to 1024x1024 for SAM
   - SAM's `set_image()` computes the image embedding
   - Display version (512x512) shown to user

2. **User clicks first corner** (app.py: `handle_segmentation_click()`)
   - Coordinates stored in `state.rect_start`
   - Visual marker drawn on image
   - State changes to wait for second click

3. **User clicks second corner** (app.py: `handle_segmentation_click()`)
   - Coordinates stored in `state.rect_end`
   - Bounding box created from both corners
   - SAM's `segment_box()` called with the box
   - Returns mask (binary image showing segmented region)
   - Overlay created showing segmentation
   - Comparison image created

4. **User clicks Generate Report** (app.py: `generate_report()`)
   - BLIP's `generate_medical_report()` called
   - The ROI (region of interest) is extracted based on mask
   - BLIP generates descriptions of ONLY the ROI
   - Report formatted and displayed

5. **User exports** (app.py: `export_results()`)
   - Segmented image, comparison, and report packaged
   - User can right-click to save images

---

## Key Concepts Explained

### What is SAM (Segment Anything Model)?
SAM is a model by Meta that can segment (draw outlines around) any object in any image. You give it a "prompt" (a point, multiple points, or a box), and it figures out what object you're pointing at and draws its boundary. It was trained on millions of images and can work on images it's never seen before.

### What is BLIP?
BLIP (Bootstrapping Language-Image Pre-training) is a model that understands both images and text. You can give it an image and a text prompt like "This chest X-ray shows..." and it will complete the sentence with a description of what it sees.

### What is a Mask?
In image segmentation, a "mask" is a binary image (only 0s and 1s, or black and white) where white pixels indicate the segmented region and black pixels are background. The mask has the same dimensions as the original image.

### What is CLAHE?
Medical images often have low contrast. CLAHE improves local contrast by:
1. Dividing the image into small tiles
2. Equalizing histogram in each tile separately
3. Limiting contrast to prevent noise amplification
4. Blending tiles together smoothly

### What is Window/Level?
In radiology, "window" controls the range of pixel values displayed (contrast), and "level" controls the center point (brightness). This lets radiologists focus on different tissue types (bone, soft tissue, lungs).

---

## Files You Can Ignore

These are auto-generated and not part of the actual codebase:

1. **`__pycache__/` folders**: Python's compiled bytecode cache. Speeds up imports but not needed in repo.

2. **`venv/` folder**: Your Python virtual environment. Contains all installed packages. Should NOT be committed to GitHub (it's huge and system-specific).

3. **`.gitkeep` files**: Empty files that make Git track otherwise-empty folders.

4. **`notebooks/01_dataset_exploration.ipynb`**: An incomplete Jupyter notebook attempt. Use the `.py` version instead.

5. **`src/venv/`**: Appears to be a nested/duplicate venv that shouldn't exist. Can be deleted.

---

## What to Include in GitHub Repository

**INCLUDE these files:**
- `app/app.py`
- `src/` directory (all .py files, excluding __pycache__)
- `notebooks/*.py` (the Python scripts)
- `config.py`
- `requirements.txt`
- `requirements.lock`
- `setup.py`
- `README.md`
- `PROJECT_DOCUMENTATION.md`
- `Dockerfile`
- `docker-compose.yml`
- `.gitignore`
- `data/raw/.gitkeep` and `data/processed/.gitkeep`
- `models/weights/.gitkeep`
- `outputs/.gitkeep`
- `logs/.gitkeep`

**DO NOT INCLUDE:**
- `venv/` folder (virtual environment)
- `__pycache__/` folders
- `data/raw/images/` (the actual X-ray images - too large, users download separately)
- `data/raw/Data_Entry_2017.csv` (users download separately)
- `models/weights/sam_vit_b_01ec64.pth` (375MB file - users download via setup.py)
- Any `.pyc` files

---

## Summary

This project demonstrates:

1. **Multi-model AI pipeline**: Combining SAM (segmentation) with BLIP (language) in one application
2. **Medical image preprocessing**: CLAHE, windowing, proper normalization
3. **Interactive web interface**: Gradio-based UI with custom styling
4. **Software engineering practices**: Modular code, configuration management, Docker support
5. **Evaluation methodology**: Dice scores, IoU, mask statistics

The main innovation is the combination of box-based interactive segmentation with automatic report generation that focuses only on the selected region, creating a practical tool for medical image analysis research.

