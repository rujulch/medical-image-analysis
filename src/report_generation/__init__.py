"""
Vision-Language model module for diagnostic report generation
Uses BLIP to generate clinical descriptions from segmented regions
"""

from .blip_reporter import ReportGenerator

__all__ = ["ReportGenerator"]

