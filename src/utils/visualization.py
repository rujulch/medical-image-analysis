"""
Visualization utilities for medical image analysis.
Creates overlays, comparisons, and publication-ready figures.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional, Union
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (14, 165, 233),  # Sky blue
    alpha: float = 0.4,
    edge_color: Tuple[int, int, int] = (6, 182, 212),  # Teal
    edge_thickness: int = 2
) -> np.ndarray:
    """
    Overlay a segmentation mask on an image with colored fill and edges.
    
    Args:
        image: Input image (H, W, 3)
        mask: Binary mask (H, W)
        color: RGB fill color
        alpha: Fill transparency
        edge_color: RGB edge color
        edge_thickness: Edge line thickness
        
    Returns:
        Image with mask overlay (H, W, 3)
    """
    # Ensure correct types
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Create overlay
    overlay = image.copy()
    
    # Apply colored fill
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        overlay[mask_bool] * (1 - alpha) + 
        np.array(color) * alpha
    ).astype(np.uint8)
    
    # Draw edge contours
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, edge_color, edge_thickness)
    
    return overlay


def create_comparison_view(
    original: np.ndarray,
    segmented: np.ndarray,
    padding: int = 10,
    background_color: Tuple[int, int, int] = (30, 41, 59)  # Slate-800
) -> np.ndarray:
    """
    Create side-by-side comparison of original and segmented images.
    
    Args:
        original: Original image (H, W, 3)
        segmented: Segmented/processed image (H, W, 3)
        padding: Padding between images
        background_color: Background color RGB
        
    Returns:
        Combined comparison image
    """
    # Ensure same size
    h, w = original.shape[:2]
    if segmented.shape[:2] != (h, w):
        segmented = cv2.resize(segmented, (w, h))
    
    # Ensure correct type
    for img in [original, segmented]:
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    
    # Create canvas
    canvas_width = w * 2 + padding * 3
    canvas_height = h + padding * 2
    canvas = np.full((canvas_height, canvas_width, 3), background_color, dtype=np.uint8)
    
    # Place images
    canvas[padding:padding+h, padding:padding+w] = original
    canvas[padding:padding+h, padding*2+w:padding*2+w*2] = segmented
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Original", (padding, padding - 5), font, 0.5, (248, 250, 252), 1)
    cv2.putText(canvas, "AI Segmentation", (padding*2 + w, padding - 5), font, 0.5, (248, 250, 252), 1)
    
    return canvas


def plot_segmentation(
    image: np.ndarray,
    mask: np.ndarray,
    point_coords: Optional[List[Tuple[int, int]]] = None,
    point_labels: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create a matplotlib figure showing segmentation result.
    
    Args:
        image: Original image
        mask: Segmentation mask
        point_coords: Click point coordinates
        point_labels: Point labels (1=foreground, 0=background)
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('#0F172A')  # Dark background
    
    # Original image
    axes[0].imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    axes[0].set_title('Original', color='white', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Plot points if provided
    if point_coords:
        for i, (x, y) in enumerate(point_coords):
            color = 'lime' if point_labels is None or point_labels[i] == 1 else 'red'
            axes[0].scatter(x, y, c=color, s=100, marker='*', edgecolors='white')
    
    # Mask
    axes[1].imshow(mask, cmap='Blues')
    axes[1].set_title('Segmentation Mask', color='white', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    overlay = overlay_mask(image.copy(), mask)
    axes[1].imshow(overlay)
    axes[1].set_title('Mask', color='white', fontsize=12, fontweight='bold')
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', color='white', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    for ax in axes:
        ax.set_facecolor('#1E293B')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, facecolor='#0F172A', bbox_inches='tight', dpi=150)
    
    return fig


def draw_point_marker(
    image: np.ndarray,
    point: Tuple[int, int],
    is_positive: bool = True,
    radius: int = 8
) -> np.ndarray:
    """
    Draw a point marker on the image.
    
    Args:
        image: Input image
        point: (x, y) coordinates
        is_positive: If True, green (foreground). If False, red (background)
        radius: Marker radius
        
    Returns:
        Image with marker
    """
    image = image.copy()
    color = (34, 197, 94) if is_positive else (239, 68, 68)  # Green or red
    
    # Draw circle with border
    cv2.circle(image, point, radius, (255, 255, 255), -1)  # White fill
    cv2.circle(image, point, radius - 2, color, -1)  # Colored center
    
    return image


def create_heatmap_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Create a heatmap overlay (useful for attention visualization).
    
    Args:
        image: Original image
        heatmap: Attention/activation heatmap (H, W)
        colormap: OpenCV colormap
        alpha: Overlay transparency
        
    Returns:
        Image with heatmap overlay
    """
    # Ensure correct format
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize heatmap if needed
    if heatmap.shape[:2] != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Normalize heatmap
    heatmap_normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, colormap)
    
    # Blend
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlay


def add_text_annotation(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.6,
    color: Tuple[int, int, int] = (248, 250, 252),
    background: bool = True
) -> np.ndarray:
    """
    Add text annotation to image with optional background.
    
    Args:
        image: Input image
        text: Text to add
        position: (x, y) position
        font_scale: Font size scale
        color: Text color RGB
        background: If True, add dark background behind text
        
    Returns:
        Annotated image
    """
    image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    
    if background:
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        x, y = position
        cv2.rectangle(
            image,
            (x - 2, y - text_h - 2),
            (x + text_w + 2, y + baseline + 2),
            (30, 41, 59),  # Dark background
            -1
        )
    
    # Draw text
    cv2.putText(image, text, position, font, font_scale, color, thickness)
    
    return image


def create_multi_mask_overlay(
    image: np.ndarray,
    masks: List[np.ndarray],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Overlay multiple masks with different colors.
    
    Args:
        image: Input image
        masks: List of binary masks
        colors: Optional list of RGB colors
        alpha: Transparency
        
    Returns:
        Image with all mask overlays
    """
    # Default color palette
    default_colors = [
        (14, 165, 233),   # Sky blue
        (34, 197, 94),    # Green
        (249, 115, 22),   # Orange
        (168, 85, 247),   # Purple
        (236, 72, 153),   # Pink
        (234, 179, 8),    # Yellow
    ]
    
    if colors is None:
        colors = default_colors
    
    result = image.copy()
    
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        result = overlay_mask(result, mask, color=color, alpha=alpha)
    
    return result

