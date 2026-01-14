"""
Configuration settings for Medical Image Analysis Platform
"""

from pathlib import Path
import torch

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# SAM Model Settings
SAM_CONFIG = {
    "model_type": "vit_b",  # Options: "vit_h", "vit_l", "vit_b" (base is faster for 6GB GPU)
    "checkpoint_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    "checkpoint_name": "sam_vit_b_01ec64.pth",
}

# BLIP Model Settings (for report generation)
BLIP_CONFIG = {
    "model_name": "Salesforce/blip-image-captioning-base",
    "max_length": 150,
    "num_beams": 5,
}

# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

def get_device():
    """Get the best available device for computation."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    return device

DEVICE = get_device()

# =============================================================================
# IMAGE PROCESSING SETTINGS
# =============================================================================

IMAGE_CONFIG = {
    "target_size": (1024, 1024),  # SAM expects 1024x1024
    "display_size": (512, 512),   # For UI display
    "supported_formats": [".png", ".jpg", ".jpeg", ".bmp", ".dcm"],
}

# X-Ray specific windowing (for proper visualization)
XRAY_WINDOW = {
    "window_center": 127,
    "window_width": 255,
}

# =============================================================================
# UI CONFIGURATION
# =============================================================================

UI_CONFIG = {
    "theme": "dark",
    "primary_color": "#0EA5E9",    # Medical blue
    "secondary_color": "#06B6D4",  # Teal accent
    "background_color": "#0F172A", # Dark slate
    "surface_color": "#1E293B",    # Lighter slate
    "text_color": "#F8FAFC",       # Almost white
    "error_color": "#EF4444",      # Red for warnings
    "success_color": "#22C55E",    # Green for success
}

# =============================================================================
# NIH CHEST X-RAY DATASET CONFIGURATION
# =============================================================================

NIH_DATASET_CONFIG = {
    "num_classes": 14,
    "class_names": [
        "Atelectasis",
        "Cardiomegaly", 
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ],
    "no_finding_label": "No Finding",
}

# =============================================================================
# EVALUATION METRICS
# =============================================================================

EVAL_CONFIG = {
    "dice_threshold": 0.5,
    "iou_threshold": 0.5,
}

