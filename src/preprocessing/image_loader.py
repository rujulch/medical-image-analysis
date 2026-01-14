"""
Image loading utilities for medical images
Supports standard formats (PNG, JPG) and DICOM
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Tuple, Optional
import cv2

# Try to import pydicom for DICOM support
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    print("PyDICOM not available. DICOM support disabled.")


def load_image(
    image_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None,
    grayscale: bool = False
) -> np.ndarray:
    """
    Load an image from path and optionally resize.
    
    Args:
        image_path: Path to the image file
        target_size: Optional (width, height) to resize to
        grayscale: If True, convert to grayscale
        
    Returns:
        numpy array of the image (H, W, C) or (H, W) if grayscale
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check if DICOM
    if image_path.suffix.lower() == '.dcm':
        return load_dicom(image_path, target_size)
    
    # Load standard image formats
    image = Image.open(image_path)
    
    # Convert to RGB or grayscale
    if grayscale:
        image = image.convert('L')
    else:
        image = image.convert('RGB')
    
    # Resize if specified
    if target_size is not None:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    return np.array(image)


def load_dicom(
    dicom_path: Union[str, Path],
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Load a DICOM file and convert to standard image format.
    
    Args:
        dicom_path: Path to DICOM file
        target_size: Optional (width, height) to resize to
        
    Returns:
        numpy array of the image (H, W, 3) as RGB
    """
    if not DICOM_AVAILABLE:
        raise ImportError("PyDICOM is required for DICOM support. Install with: pip install pydicom")
    
    dicom_path = Path(dicom_path)
    ds = pydicom.dcmread(dicom_path)
    
    # Get pixel array
    pixel_array = ds.pixel_array.astype(np.float32)
    
    # Apply rescale if available
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        pixel_array = pixel_array * ds.RescaleSlope + ds.RescaleIntercept
    
    # Normalize to 0-255
    pixel_array = normalize_to_uint8(pixel_array)
    
    # Convert grayscale to RGB
    if len(pixel_array.shape) == 2:
        pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
    
    # Resize if specified
    if target_size is not None:
        pixel_array = cv2.resize(pixel_array, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return pixel_array


def load_xray(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (1024, 1024),
    apply_clahe: bool = True
) -> np.ndarray:
    """
    Load and preprocess a chest X-ray image for analysis.
    Optimized for SAM input requirements.
    
    Args:
        image_path: Path to X-ray image
        target_size: Size for SAM (default 1024x1024)
        apply_clahe: Apply contrast enhancement
        
    Returns:
        Preprocessed image as numpy array (H, W, 3)
    """
    # Load image
    image = load_image(image_path, grayscale=True)
    
    # Apply CLAHE for better contrast (common in medical imaging)
    if apply_clahe:
        image = apply_clahe_enhancement(image)
    
    # Convert to RGB (SAM expects 3 channels)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize for SAM
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return image


def apply_clahe_enhancement(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Commonly used in medical imaging to enhance local contrast.
    
    Args:
        image: Grayscale image (H, W)
        clip_limit: Threshold for contrast limiting
        
    Returns:
        Enhanced image
    """
    if image.dtype != np.uint8:
        image = normalize_to_uint8(image)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    
    return enhanced


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to uint8 range [0, 255].
    
    Args:
        image: Input image array
        
    Returns:
        Normalized uint8 image
    """
    # Handle edge case of constant image
    if image.max() == image.min():
        return np.zeros_like(image, dtype=np.uint8)
    
    # Normalize to 0-255
    image_normalized = (image - image.min()) / (image.max() - image.min())
    image_uint8 = (image_normalized * 255).astype(np.uint8)
    
    return image_uint8


def load_from_array(
    image_array: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Process an image from numpy array (e.g., from Gradio upload).
    
    Args:
        image_array: Input image array
        target_size: Optional resize dimensions
        
    Returns:
        Processed image array
    """
    # Ensure RGB
    if len(image_array.shape) == 2:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Resize if needed
    if target_size is not None:
        image_array = cv2.resize(image_array, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return image_array

