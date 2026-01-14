"""
Image transformation utilities for medical image preprocessing
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def normalize_xray(
    image: np.ndarray,
    method: str = "minmax"
) -> np.ndarray:
    """
    Normalize X-ray image for model input.
    
    Args:
        image: Input image array
        method: Normalization method ("minmax", "zscore", "imagenet")
        
    Returns:
        Normalized image array
    """
    image = image.astype(np.float32)
    
    if method == "minmax":
        # Scale to [0, 1]
        if image.max() > image.min():
            image = (image - image.min()) / (image.max() - image.min())
        else:
            image = np.zeros_like(image)
            
    elif method == "zscore":
        # Zero mean, unit variance
        mean = np.mean(image)
        std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        else:
            image = image - mean
            
    elif method == "imagenet":
        # ImageNet normalization (for pretrained models)
        # Assumes image is in [0, 255]
        if image.max() > 1:
            image = image / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        if len(image.shape) == 3:
            image = (image - mean) / std
            
    return image


def apply_windowing(
    image: np.ndarray,
    window_center: float = 127,
    window_width: float = 255
) -> np.ndarray:
    """
    Apply window/level adjustment (common in radiology).
    This simulates the windowing that radiologists use to view X-rays.
    
    Args:
        image: Input image (grayscale or RGB)
        window_center: Center of the window (brightness)
        window_width: Width of the window (contrast)
        
    Returns:
        Windowed image
    """
    # Calculate window boundaries
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    
    # Apply windowing
    windowed = np.clip(image, lower, upper)
    
    # Rescale to 0-255
    if upper > lower:
        windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
    else:
        windowed = np.zeros_like(image, dtype=np.uint8)
    
    return windowed


def resize_for_sam(
    image: np.ndarray,
    target_size: int = 1024
) -> Tuple[np.ndarray, dict]:
    """
    Resize image for SAM model input while preserving aspect ratio info.
    SAM expects 1024x1024 input.
    
    Args:
        image: Input image (H, W, C) or (H, W)
        target_size: Target dimension (default 1024 for SAM)
        
    Returns:
        Tuple of (resized_image, resize_info dict for coordinate mapping)
    """
    original_h, original_w = image.shape[:2]
    
    # Calculate scale to fit in target_size while maintaining aspect ratio
    scale = target_size / max(original_h, original_w)
    new_h = int(original_h * scale)
    new_w = int(original_w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # Create padded image (center the resized image)
    if len(image.shape) == 3:
        padded = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((target_size, target_size), dtype=image.dtype)
    
    # Calculate padding offsets
    pad_h = (target_size - new_h) // 2
    pad_w = (target_size - new_w) // 2
    
    # Place resized image in center
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    # Store resize info for coordinate transformation
    resize_info = {
        "original_size": (original_h, original_w),
        "resized_size": (new_h, new_w),
        "target_size": target_size,
        "scale": scale,
        "pad_h": pad_h,
        "pad_w": pad_w,
    }
    
    return padded, resize_info


def transform_coordinates(
    x: int,
    y: int,
    resize_info: dict,
    to_original: bool = False
) -> Tuple[int, int]:
    """
    Transform coordinates between original and resized image space.
    
    Args:
        x, y: Input coordinates
        resize_info: Dictionary from resize_for_sam
        to_original: If True, transform from SAM space to original.
                    If False, transform from original to SAM space.
                    
    Returns:
        Transformed (x, y) coordinates
    """
    if to_original:
        # From SAM (1024x1024) to original
        x_adj = x - resize_info["pad_w"]
        y_adj = y - resize_info["pad_h"]
        x_orig = int(x_adj / resize_info["scale"])
        y_orig = int(y_adj / resize_info["scale"])
        return x_orig, y_orig
    else:
        # From original to SAM (1024x1024)
        x_sam = int(x * resize_info["scale"]) + resize_info["pad_w"]
        y_sam = int(y * resize_info["scale"]) + resize_info["pad_h"]
        return x_sam, y_sam


def enhance_contrast(
    image: np.ndarray,
    alpha: float = 1.5,
    beta: float = 0
) -> np.ndarray:
    """
    Simple contrast and brightness adjustment.
    
    Args:
        image: Input image
        alpha: Contrast control (1.0-3.0)
        beta: Brightness control (0-100)
        
    Returns:
        Adjusted image
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted


def invert_xray(image: np.ndarray) -> np.ndarray:
    """
    Invert X-ray image (some X-rays are inverted).
    
    Args:
        image: Input X-ray image
        
    Returns:
        Inverted image
    """
    if image.dtype == np.uint8:
        return 255 - image
    elif image.max() <= 1.0:
        return 1.0 - image
    else:
        return image.max() - image


def prepare_for_display(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512)
) -> np.ndarray:
    """
    Prepare image for UI display.
    
    Args:
        image: Input image
        target_size: Display size (width, height)
        
    Returns:
        Display-ready image (uint8, RGB)
    """
    # Ensure uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Ensure RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize for display
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    
    return image

