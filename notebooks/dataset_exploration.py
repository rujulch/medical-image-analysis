"""
NIH Chest X-Ray Dataset Exploration Script
Can be run as a script or converted to Jupyter notebook using jupytext.

Dataset Overview:
- Source: NIH Clinical Center (https://www.kaggle.com/datasets/nih-chest-xrays/data)
- Size: 112,120 chest X-ray images
- Patients: 30,805 unique patients
- Labels: 14 disease categories + "No Finding"
"""

# %% [markdown]
# # NIH Chest X-Ray Dataset Exploration

# %%
# Import required libraries
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

# Set style
plt.style.use('dark_background')

print("Libraries imported successfully!")

# %%
# Configuration
DATA_DIR = project_root / 'data' / 'raw'
LABELS_FILE = DATA_DIR / 'Data_Entry_2017.csv'

# Disease labels in the NIH dataset
DISEASE_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural_Thickening', 'Hernia'
]

print(f"Data directory: {DATA_DIR}")
print(f"Labels file: {LABELS_FILE}")

# %%
# Load and analyze labels
def load_labels():
    """Load the NIH dataset labels."""
    if LABELS_FILE.exists():
        df = pd.read_csv(LABELS_FILE)
        print(f"Loaded {len(df)} records")
        return df
    else:
        print(f"Labels file not found: {LABELS_FILE}")
        print("\nTo use this script:")
        print("1. Download NIH Chest X-Ray dataset from Kaggle")
        print("2. Place Data_Entry_2017.csv in data/raw/")
        return None

df = load_labels()

# %%
# Analyze label distribution
def analyze_labels(df):
    """Analyze the distribution of disease labels."""
    if df is None:
        return None
    
    # Parse pipe-separated labels
    all_labels = []
    for labels in df['Finding Labels']:
        if pd.notna(labels):
            all_labels.extend(labels.split('|'))
    
    label_counts = pd.Series(all_labels).value_counts()
    return label_counts

if df is not None:
    label_counts = analyze_labels(df)
    print("\nLabel Distribution:")
    print(label_counts)

# %%
# Visualize sample X-ray
def display_sample_xray(image_path):
    """Display a sample X-ray with preprocessing steps."""
    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        return
    
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    # Apply CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)
    
    # Display
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title('CLAHE Enhanced')
    axes[1].axis('off')
    
    axes[2].hist(img.flatten(), bins=256, color='cyan', alpha=0.7, label='Original')
    axes[2].hist(enhanced.flatten(), bins=256, color='magenta', alpha=0.7, label='Enhanced')
    axes[2].set_title('Pixel Distribution')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(project_root / 'outputs' / 'sample_xray_analysis.png', dpi=150)
    plt.show()
    
    return img

# Try to display a sample
images_dir = DATA_DIR / 'images'
if images_dir.exists():
    samples = list(images_dir.glob('*.png'))[:1]
    if samples:
        display_sample_xray(samples[0])

# %%
# Summary
print("\n" + "="*60)
print("Dataset Summary")
print("="*60)
print(f"Dataset: NIH Chest X-Ray")
print(f"Disease categories: {len(DISEASE_LABELS)}")
print(f"Labels: {', '.join(DISEASE_LABELS[:7])}...")
print("="*60)

