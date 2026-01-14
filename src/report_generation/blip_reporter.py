"""
BLIP Vision-Language Model for generating diagnostic reports.
Takes segmented regions and generates clinical descriptions.
"""

import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Optional, Union, List, Tuple
import cv2

# Import transformers for BLIP
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False
    print("Warning: transformers not installed. Run: pip install transformers")

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from config import BLIP_CONFIG, DEVICE, NIH_DATASET_CONFIG


class ReportGenerator:
    """
    BLIP-based report generator for medical images.
    Generates natural language descriptions of X-ray findings.
    """
    
    def __init__(
        self,
        model_name: str = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize BLIP report generator.
        
        Args:
            model_name: HuggingFace model name
            device: Torch device (cuda/cpu)
        """
        if not BLIP_AVAILABLE:
            raise ImportError("transformers package is required. Install it first.")
        
        self.model_name = model_name or BLIP_CONFIG["model_name"]
        self.device = device or DEVICE
        
        print(f"Loading BLIP model ({self.model_name}) on {self.device}...")
        
        # Load processor and model
        # Use safetensors format to avoid torch.load security restrictions
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            self.model_name,
            use_safetensors=True  # Required for PyTorch < 2.6 security compliance
        )
        self.model.to(self.device)
        self.model.eval()
        
        print("BLIP model loaded successfully!")
        
        # Medical terminology templates
        self._setup_medical_prompts()
    
    def _setup_medical_prompts(self):
        """Setup medical-specific prompts for better clinical descriptions."""
        self.prompts = {
            "general": "This chest X-ray shows",
            "finding": "The radiological finding in this image is",
            "region": "The highlighted region shows",
            "diagnostic": "Based on this X-ray, the diagnostic impression is",
            "description": "A detailed description of this medical image:",
        }
        
        # Medical terminology for post-processing
        self.medical_terms = {
            "dark area": "hypodense region",
            "white area": "hyperdense opacity",
            "spot": "focal lesion",
            "shadow": "opacity",
            "lung": "pulmonary parenchyma",
            "heart": "cardiac silhouette",
            "bone": "osseous structure",
        }
    
    def generate_caption(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: Optional[str] = None,
        max_length: int = None,
        num_beams: int = None
    ) -> str:
        """
        Generate a caption/description for the image.
        
        Args:
            image: Input image (numpy array or PIL Image)
            prompt: Text prompt to guide generation
            max_length: Maximum output length
            num_beams: Beam search width
            
        Returns:
            Generated caption string
        """
        max_length = max_length or BLIP_CONFIG["max_length"]
        num_beams = num_beams or BLIP_CONFIG["num_beams"]
        
        # Convert numpy to PIL
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            image = Image.fromarray(image)
        
        # Ensure RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Process with prompt
        if prompt:
            inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True
            )
        
        # Decode
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption
    
    def generate_medical_report(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        finding_type: Optional[str] = None
    ) -> dict:
        """
        Generate a structured medical report for the image.
        
        When a mask is provided, the report will ONLY analyze the segmented region,
        not the full image. This ensures excluded areas are not described.
        
        Args:
            image: Input X-ray image (H, W, 3)
            mask: Optional segmentation mask (H, W) - if provided, only this region is analyzed
            finding_type: Optional finding type hint
            
        Returns:
            Dictionary with report sections
        """
        report = {
            "general_impression": "",
            "findings": "",
            "segmented_region": "",
            "recommendations": "",
        }
        
        # Determine which image to analyze
        # If mask is provided, ONLY analyze the masked region
        if mask is not None:
            roi_image = self._extract_roi(image, mask)
            if roi_image is not None:
                # All analysis is done on the ROI only
                analysis_image = roi_image
                
                # Generate findings for the segmented region only
                report["general_impression"] = self.generate_caption(
                    analysis_image, 
                    prompt="This region of the chest X-ray shows"
                )
                
                report["findings"] = self.generate_caption(
                    analysis_image,
                    prompt=self.prompts["finding"]
                )
                
                report["segmented_region"] = self.generate_caption(
                    analysis_image,
                    prompt=self.prompts["region"]
                )
            else:
                # Mask is empty, analyze full image
                report["general_impression"] = self.generate_caption(
                    image, 
                    prompt=self.prompts["general"]
                )
                report["findings"] = self.generate_caption(
                    image,
                    prompt=self.prompts["finding"]
                )
        else:
            # No mask provided - analyze full image
            report["general_impression"] = self.generate_caption(
                image, 
                prompt=self.prompts["general"]
            )
            
            report["findings"] = self.generate_caption(
                image,
                prompt=self.prompts["finding"]
            )
        
        # Post-process to add medical terminology
        report = self._enhance_medical_terminology(report)
        
        # Add recommendations based on findings
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    def _extract_roi(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        padding: int = 20
    ) -> Optional[np.ndarray]:
        """
        Extract the region of interest based on mask.
        
        Args:
            image: Full image
            mask: Binary mask
            padding: Padding around bounding box
            
        Returns:
            Cropped region or None if mask is empty
        """
        # Find bounding box of mask
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return None
        
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # Add padding
        h, w = image.shape[:2]
        y_min = max(0, y_min - padding)
        y_max = min(h, y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(w, x_max + padding)
        
        # Crop
        roi = image[y_min:y_max, x_min:x_max]
        
        return roi
    
    def _enhance_medical_terminology(self, report: dict) -> dict:
        """Replace common terms with medical terminology."""
        for key, text in report.items():
            if isinstance(text, str):
                for common, medical in self.medical_terms.items():
                    text = text.replace(common, medical)
                report[key] = text
        return report
    
    def _generate_recommendations(self, report: dict) -> str:
        """Generate recommendations based on findings."""
        findings = report.get("findings", "").lower()
        
        recommendations = []
        
        # Check for concerning findings
        concerning_terms = ["mass", "nodule", "opacity", "lesion", "tumor", "consolidation"]
        for term in concerning_terms:
            if term in findings:
                recommendations.append(
                    f"Further evaluation recommended due to {term} finding."
                )
                break
        
        if not recommendations:
            recommendations.append("No concerning findings requiring immediate follow-up.")
        
        recommendations.append("Clinical correlation recommended.")
        
        return " ".join(recommendations)
    
    def format_report(self, report: dict) -> str:
        """
        Format report dictionary as readable text.
        
        Args:
            report: Report dictionary from generate_medical_report
            
        Returns:
            Formatted report string
        """
        sections = [
            ("GENERAL IMPRESSION", report.get("general_impression", "N/A")),
            ("FINDINGS", report.get("findings", "N/A")),
            ("SEGMENTED REGION ANALYSIS", report.get("segmented_region", "N/A")),
            ("RECOMMENDATIONS", report.get("recommendations", "N/A")),
        ]
        
        formatted = []
        formatted.append("=" * 60)
        formatted.append("RADIOLOGY REPORT - AI ASSISTED ANALYSIS")
        formatted.append("=" * 60)
        formatted.append("")
        
        for title, content in sections:
            if content and content != "N/A":
                formatted.append(f"{title}:")
                formatted.append("-" * 40)
                formatted.append(content)
                formatted.append("")
        
        formatted.append("=" * 60)
        formatted.append("DISCLAIMER: AI-generated analysis for research purposes only.")
        formatted.append("Not intended for clinical diagnosis.")
        formatted.append("=" * 60)
        
        return "\n".join(formatted)
    
    def generate_quick_description(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate a quick one-line description.
        
        Args:
            image: Input image
            mask: Optional segmentation mask
            
        Returns:
            Short description string
        """
        if mask is not None:
            roi = self._extract_roi(image, mask)
            if roi is not None:
                return self.generate_caption(roi, prompt="This shows")
        
        return self.generate_caption(image, prompt="This chest X-ray shows")


def create_reporter(device: Optional[str] = None) -> ReportGenerator:
    """
    Factory function to create report generator.
    
    Args:
        device: "cuda" or "cpu"
        
    Returns:
        Initialized ReportGenerator
    """
    if device:
        dev = torch.device(device)
    else:
        dev = None
    
    return ReportGenerator(device=dev)

