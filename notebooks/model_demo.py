"""
SAM + BLIP Model Demonstration Script

Demonstrates the core AI models used in the Medical Image Analysis Platform:
1. SAM (Segment Anything Model): For precise medical image segmentation
2. BLIP (Vision-Language Model): For generating diagnostic descriptions
"""

# %% [markdown]
# # SAM + BLIP Model Demonstration

# %%
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

plt.style.use('dark_background')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
# Load SAM Model
print("\n" + "="*60)
print("Loading SAM Model")
print("="*60)

from src.segmentation.sam_predictor import SAMSegmenter

segmenter = SAMSegmenter()
print("SAM loaded successfully!")

# %%
# Load BLIP Model
print("\n" + "="*60)
print("Loading BLIP Model")
print("="*60)

from src.report_generation.blip_reporter import ReportGenerator

reporter = ReportGenerator()
print("BLIP loaded successfully!")

# %%
# Segmentation Demo
from src.preprocessing.image_loader import load_xray
from src.preprocessing.transforms import resize_for_sam
from src.utils.visualization import overlay_mask

def demo_segmentation(image_path, click_points):
    """Demonstrate SAM segmentation."""
    print(f"\nLoading: {image_path}")
    
    # Load and preprocess
    image = load_xray(image_path)
    sam_image, resize_info = resize_for_sam(image)
    
    # Set image and segment
    segmenter.set_image(sam_image)
    mask, score = segmenter.get_best_mask(click_points)
    
    print(f"Segmentation score: {score:.3f}")
    print(f"Mask area: {np.sum(mask):,} pixels")
    
    # Visualize
    overlay = overlay_mask(sam_image.copy(), mask)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(sam_image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(mask, cmap='Blues')
    axes[1].set_title(f'Mask (Score: {score:.3f})')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    for x, y in click_points:
        axes[2].scatter(x, y, c='lime', s=100, marker='*', edgecolors='white')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(project_root / 'outputs' / 'segmentation_demo.png', dpi=150)
    plt.show()
    
    return sam_image, mask

# %%
# Report Generation Demo
def demo_report(image, mask=None):
    """Demonstrate BLIP report generation."""
    print("\n" + "="*60)
    print("Generating Diagnostic Report")
    print("="*60)
    
    report = reporter.generate_medical_report(image, mask=mask)
    formatted = reporter.format_report(report)
    print(formatted)
    
    return report

# %%
# Run Demo
DATA_DIR = project_root / 'data' / 'raw' / 'images'

if DATA_DIR.exists():
    samples = list(DATA_DIR.glob('*.png'))
    if samples:
        # Demo segmentation
        image, mask = demo_segmentation(
            samples[0],
            click_points=[(512, 512)]  # Center of image
        )
        
        # Demo report generation
        report = demo_report(image, mask)
    else:
        print(f"No images found in {DATA_DIR}")
else:
    print(f"Data directory not found: {DATA_DIR}")
    print("Please download the NIH Chest X-Ray dataset first.")

# %%
print("\n" + "="*60)
print("Demo Complete!")
print("="*60)
print("\nTo use the full application, run:")
print("  python app/app.py")

