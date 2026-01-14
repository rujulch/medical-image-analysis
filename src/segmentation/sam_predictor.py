"""
SAM (Segment Anything Model) integration for medical image segmentation.
Provides interactive point-based and automatic segmentation.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Union
import urllib.request
import os

# Import SAM components
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: segment_anything not installed. Run: pip install git+https://github.com/facebookresearch/segment-anything.git")

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import SAM_CONFIG, WEIGHTS_DIR, DEVICE


class SAMSegmenter:
    """
    SAM-based segmentation for medical images.
    Supports both interactive (click-based) and automatic segmentation.
    """
    
    def __init__(
        self,
        model_type: str = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize SAM segmenter.
        
        Args:
            model_type: SAM model variant ("vit_h", "vit_l", "vit_b")
            checkpoint_path: Path to model weights
            device: Torch device (cuda/cpu)
        """
        if not SAM_AVAILABLE:
            raise ImportError("segment_anything package is required. Install it first.")
        
        self.model_type = model_type or SAM_CONFIG["model_type"]
        self.device = device or DEVICE
        
        # Setup checkpoint path
        if checkpoint_path is None:
            checkpoint_path = WEIGHTS_DIR / SAM_CONFIG["checkpoint_name"]
        self.checkpoint_path = Path(checkpoint_path)
        
        # Download weights if not present
        self._ensure_weights()
        
        # Load model
        print(f"Loading SAM model ({self.model_type}) on {self.device}...")
        self.sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint_path))
        self.sam.to(self.device)
        self.sam.eval()
        
        # Initialize predictor for interactive segmentation
        self.predictor = SamPredictor(self.sam)
        
        # Current image embedding (cached for multiple predictions)
        self._current_image = None
        
        print("SAM model loaded successfully!")
    
    def _ensure_weights(self):
        """Download SAM weights if not present."""
        if self.checkpoint_path.exists():
            print(f"SAM weights found: {self.checkpoint_path}")
            return
        
        print(f"Downloading SAM weights...")
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        url = SAM_CONFIG["checkpoint_url"]
        
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rDownloading: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, self.checkpoint_path, show_progress)
        print("\nDownload complete!")
    
    def set_image(self, image: np.ndarray) -> None:
        """
        Set the image for segmentation. Computes image embedding.
        
        Args:
            image: Input image (H, W, 3) in RGB format, uint8
        """
        # Ensure correct format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 2:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Set image in predictor (computes embedding)
        self.predictor.set_image(image)
        self._current_image = image
        print("Image embedding computed.")
    
    def segment_point(
        self,
        point_coords: List[Tuple[int, int]],
        point_labels: Optional[List[int]] = None,
        multimask_output: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment based on point prompts (click locations).
        
        Args:
            point_coords: List of (x, y) coordinates for prompt points
            point_labels: List of labels (1 for foreground, 0 for background)
                         If None, all points are treated as foreground
            multimask_output: If True, returns 3 mask options
            
        Returns:
            masks: Array of masks (N, H, W) where N is 1 or 3
            scores: Confidence scores for each mask
            logits: Raw logits for each mask
        """
        if self._current_image is None:
            raise ValueError("No image set. Call set_image() first.")
        
        # Prepare point coordinates
        point_coords = np.array(point_coords)
        
        # Default: all foreground points
        if point_labels is None:
            point_labels = np.ones(len(point_coords), dtype=np.int32)
        else:
            point_labels = np.array(point_labels, dtype=np.int32)
        
        # Get predictions
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask_output
        )
        
        return masks, scores, logits
    
    def segment_box(
        self,
        box: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment based on bounding box prompt.
        
        Args:
            box: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            masks, scores, logits
        """
        if self._current_image is None:
            raise ValueError("No image set. Call set_image() first.")
        
        box = np.array(box)
        
        masks, scores, logits = self.predictor.predict(
            box=box,
            multimask_output=False
        )
        
        return masks, scores, logits
    
    def get_best_mask(
        self,
        point_coords: List[Tuple[int, int]],
        point_labels: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Get the best mask from point prompts.
        
        Args:
            point_coords: List of (x, y) coordinates
            point_labels: List of labels (1=foreground, 0=background)
            
        Returns:
            best_mask: Binary mask (H, W)
            score: Confidence score
        """
        masks, scores, _ = self.segment_point(
            point_coords, point_labels, multimask_output=True
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        return best_mask, float(best_score)
    
    def segment_automatic(
        self,
        image: np.ndarray,
        points_per_side: int = 16,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100
    ) -> List[dict]:
        """
        Automatic mask generation for the entire image.
        Useful for finding all potential regions of interest.
        
        Args:
            image: Input image (H, W, 3)
            points_per_side: Grid density for automatic sampling
            pred_iou_thresh: Predicted IoU threshold
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask area
            
        Returns:
            List of mask dictionaries with keys:
                - segmentation: Binary mask
                - area: Mask area in pixels
                - bbox: Bounding box [x, y, w, h]
                - predicted_iou: Predicted IoU score
                - stability_score: Stability score
        """
        # Create automatic mask generator
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )
        
        # Ensure correct format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Generate masks
        masks = mask_generator.generate(image)
        
        # Sort by area (largest first)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        
        return masks
    
    def get_mask_overlay(
        self,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 164, 229),  # Medical blue
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Create a colored overlay for the mask.
        
        Args:
            mask: Binary mask (H, W)
            color: RGB color tuple
            alpha: Overlay transparency
            
        Returns:
            Colored mask overlay (H, W, 4) with alpha channel
        """
        h, w = mask.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Set color where mask is True
        overlay[mask > 0] = [color[0], color[1], color[2], int(alpha * 255)]
        
        return overlay
    
    def clear_image(self):
        """Clear the current image embedding."""
        self._current_image = None
        self.predictor.reset_image()


def create_segmenter(
    model_type: str = "vit_b",
    device: Optional[str] = None
) -> SAMSegmenter:
    """
    Factory function to create SAM segmenter.
    
    Args:
        model_type: Model size ("vit_b" recommended for 6GB GPU)
        device: "cuda" or "cpu"
        
    Returns:
        Initialized SAMSegmenter
    """
    if device:
        dev = torch.device(device)
    else:
        dev = None
    
    return SAMSegmenter(model_type=model_type, device=dev)

