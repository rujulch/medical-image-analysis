"""
Evaluation metrics for medical image segmentation.
"""

import numpy as np
from typing import Tuple, Optional


def calculate_dice_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Dice coefficient (F1 score for segmentation).
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient (0-1, higher is better)
    """
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)
    
    intersection = np.sum(pred_flat & gt_flat)
    union_sum = np.sum(pred_flat) + np.sum(gt_flat)
    
    dice = (2.0 * intersection + smooth) / (union_sum + smooth)
    
    return float(dice)


def calculate_iou(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    smooth: float = 1e-6
) -> float:
    """
    Calculate Intersection over Union (Jaccard Index).
    IoU = |A ∩ B| / |A ∪ B|
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        smooth: Smoothing factor
        
    Returns:
        IoU score (0-1, higher is better)
    """
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)
    
    intersection = np.sum(pred_flat & gt_flat)
    union = np.sum(pred_flat | gt_flat)
    
    iou = (intersection + smooth) / (union + smooth)
    
    return float(iou)


def calculate_precision_recall(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate precision and recall for segmentation.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Tuple of (precision, recall)
    """
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)
    
    true_positive = np.sum(pred_flat & gt_flat)
    false_positive = np.sum(pred_flat & ~gt_flat)
    false_negative = np.sum(~pred_flat & gt_flat)
    
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    
    return float(precision), float(recall)


def calculate_specificity(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> float:
    """
    Calculate specificity (true negative rate).
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Specificity score
    """
    pred_flat = pred_mask.flatten().astype(bool)
    gt_flat = gt_mask.flatten().astype(bool)
    
    true_negative = np.sum(~pred_flat & ~gt_flat)
    false_positive = np.sum(pred_flat & ~gt_flat)
    
    specificity = true_negative / (true_negative + false_positive + 1e-6)
    
    return float(specificity)


def calculate_all_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray
) -> dict:
    """
    Calculate all segmentation metrics.
    
    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask
        
    Returns:
        Dictionary with all metrics
    """
    dice = calculate_dice_score(pred_mask, gt_mask)
    iou = calculate_iou(pred_mask, gt_mask)
    precision, recall = calculate_precision_recall(pred_mask, gt_mask)
    specificity = calculate_specificity(pred_mask, gt_mask)
    
    # Calculate mask statistics
    pred_area = np.sum(pred_mask > 0)
    gt_area = np.sum(gt_mask > 0)
    
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": dice,  # Dice is equivalent to F1
        "predicted_area_pixels": int(pred_area),
        "ground_truth_area_pixels": int(gt_area),
    }


def format_metrics_report(metrics: dict) -> str:
    """
    Format metrics dictionary as readable report.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted string
    """
    lines = [
        "=" * 40,
        "SEGMENTATION EVALUATION METRICS",
        "=" * 40,
        f"Dice Coefficient:  {metrics['dice']:.4f}",
        f"IoU (Jaccard):     {metrics['iou']:.4f}",
        f"Precision:         {metrics['precision']:.4f}",
        f"Recall:            {metrics['recall']:.4f}",
        f"Specificity:       {metrics['specificity']:.4f}",
        "-" * 40,
        f"Predicted Area:    {metrics['predicted_area_pixels']:,} pixels",
        f"Ground Truth Area: {metrics['ground_truth_area_pixels']:,} pixels",
        "=" * 40,
    ]
    
    return "\n".join(lines)


def calculate_mask_stats(mask: np.ndarray) -> dict:
    """
    Calculate statistics about a segmentation mask.
    
    Args:
        mask: Binary mask
        
    Returns:
        Dictionary with mask statistics
    """
    # Find connected components
    import cv2
    
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8)
    
    # Exclude background (label 0)
    num_regions = num_labels - 1
    
    if num_regions == 0:
        return {
            "num_regions": 0,
            "total_area": 0,
            "largest_area": 0,
            "centroid": None,
        }
    
    # Get areas (excluding background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    # Get centroid of largest region
    largest_idx = np.argmax(areas) + 1  # +1 because we excluded background
    main_centroid = tuple(centroids[largest_idx].astype(int))
    
    return {
        "num_regions": num_regions,
        "total_area": int(np.sum(areas)),
        "largest_area": int(np.max(areas)),
        "mean_area": float(np.mean(areas)),
        "centroid": main_centroid,
        "bounding_boxes": [
            {
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "width": int(stats[i, cv2.CC_STAT_WIDTH]),
                "height": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "area": int(stats[i, cv2.CC_STAT_AREA]),
            }
            for i in range(1, num_labels)
        ]
    }

