"""
Medical Image Analysis Platform - Main Application
An AI-powered diagnostic tool for chest X-ray analysis using SAM and BLIP.
"""

import gradio as gr
import numpy as np
from pathlib import Path
import sys
import cv2
from PIL import Image
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import UI_CONFIG, IMAGE_CONFIG
from src.preprocessing.image_loader import load_from_array, load_xray
from src.preprocessing.transforms import resize_for_sam, prepare_for_display, apply_windowing
from src.utils.visualization import overlay_mask, create_comparison_view, draw_point_marker
from src.utils.metrics import calculate_mask_stats, format_metrics_report

# Global model instances (lazy loaded)
_segmenter = None
_reporter = None


def get_segmenter():
    """Lazy load SAM segmenter."""
    global _segmenter
    if _segmenter is None:
        from src.segmentation.sam_predictor import SAMSegmenter
        print("Initializing SAM segmenter...")
        _segmenter = SAMSegmenter()
    return _segmenter


def get_reporter():
    """Lazy load BLIP reporter."""
    global _reporter
    if _reporter is None:
        from src.report_generation.blip_reporter import ReportGenerator
        print("Initializing BLIP report generator...")
        _reporter = ReportGenerator()
    return _reporter


# State management
class AnalysisState:
    """Manages the current analysis state."""
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.resize_info = None
        self.current_mask = None
        self.bounding_box = None
        self.mask_score = 0.0
        self.comparison_image = None
        # Rectangle selection state
        self.rect_start = None  # First click (x, y)
        self.rect_end = None    # Second click (x, y)
        self.selection_step = 0  # 0 = waiting for first click, 1 = waiting for second click
        
    def reset(self):
        self.original_image = None
        self.processed_image = None
        self.resize_info = None
        self.current_mask = None
        self.bounding_box = None
        self.mask_score = 0.0
        self.comparison_image = None
        self.rect_start = None
        self.rect_end = None
        self.selection_step = 0
    
    def reset_selection(self):
        """Reset only the selection, keep the image."""
        self.current_mask = None
        self.bounding_box = None
        self.mask_score = 0.0
        self.comparison_image = None
        self.rect_start = None
        self.rect_end = None
        self.selection_step = 0


state = AnalysisState()


def draw_selection_rectangle(image, start, end, is_preview=False):
    """
    Draw a Windows screenshot-style selection rectangle on the image.
    
    Args:
        image: Input image (numpy array)
        start: (x, y) of first corner
        end: (x, y) of second corner (or current mouse position for preview)
        is_preview: If True, draw dashed preview; if False, draw solid selection
    """
    img = image.copy()
    
    x1, y1 = start
    x2, y2 = end
    
    # Ensure proper ordering
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)
    
    # Draw semi-transparent overlay outside the selection (darken non-selected areas)
    overlay = img.copy()
    
    # Create a mask for the selection area
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y_min:y_max, x_min:x_max] = 255
    
    # Darken areas outside selection
    darkened = (img * 0.5).astype(np.uint8)
    img = np.where(mask[:, :, np.newaxis] == 255, img, darkened)
    
    # Draw rectangle border - Windows style (cyan/teal color)
    border_color = (14, 165, 233)  # Sky blue
    thickness = 2
    
    if is_preview:
        # Draw dashed rectangle for preview
        dash_length = 10
        gap_length = 5
        
        # Top edge
        for x in range(x_min, x_max, dash_length + gap_length):
            cv2.line(img, (x, y_min), (min(x + dash_length, x_max), y_min), border_color, thickness)
        # Bottom edge
        for x in range(x_min, x_max, dash_length + gap_length):
            cv2.line(img, (x, y_max), (min(x + dash_length, x_max), y_max), border_color, thickness)
        # Left edge
        for y in range(y_min, y_max, dash_length + gap_length):
            cv2.line(img, (x_min, y), (x_min, min(y + dash_length, y_max)), border_color, thickness)
        # Right edge
        for y in range(y_min, y_max, dash_length + gap_length):
            cv2.line(img, (x_max, y), (x_max, min(y + dash_length, y_max)), border_color, thickness)
    else:
        # Draw solid rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), border_color, thickness)
    
    # Draw corner handles
    handle_size = 6
    handle_color = (255, 255, 255)  # White handles
    corners = [
        (x_min, y_min), (x_max, y_min),
        (x_min, y_max), (x_max, y_max)
    ]
    for cx, cy in corners:
        cv2.rectangle(img, (cx - handle_size//2, cy - handle_size//2), 
                     (cx + handle_size//2, cy + handle_size//2), handle_color, -1)
        cv2.rectangle(img, (cx - handle_size//2, cy - handle_size//2), 
                     (cx + handle_size//2, cy + handle_size//2), border_color, 1)
    
    return img


def process_uploaded_image(image, apply_enhancement=True):
    """
    Process an uploaded image for analysis.
    """
    if image is None:
        return None, None, "Please upload an image.", "Click on the image to start selecting a region."
    
    state.reset()
    
    try:
        # Process image
        processed = load_from_array(image)
        
        # Apply CLAHE if grayscale-ish (X-ray)
        if apply_enhancement:
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        
        # Resize for SAM
        sam_ready, resize_info = resize_for_sam(processed, target_size=1024)
        
        # Store in state
        state.original_image = image.copy()
        state.processed_image = sam_ready
        state.resize_info = resize_info
        
        # Set image in segmenter
        segmenter = get_segmenter()
        segmenter.set_image(sam_ready)
        
        # Prepare display version
        display = prepare_for_display(sam_ready, target_size=(512, 512))
        
        return display, display.copy(), "Image loaded successfully.", "Step 1: Click to set the FIRST corner of your selection rectangle."
        
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}", "Error loading image."


def handle_segmentation_click(image, evt: gr.SelectData):
    """
    Handle clicks for rectangle selection.
    First click = start corner, Second click = end corner -> trigger segmentation
    """
    if state.processed_image is None:
        return image, "Please upload an image first."
    
    try:
        # Get click coordinates (display is 512x512, SAM needs 1024x1024)
        x_display = int(evt.index[0])
        y_display = int(evt.index[1])
        
        # Scale to SAM coordinates
        x_sam = int(x_display * 2)
        y_sam = int(y_display * 2)
        
        if state.selection_step == 0:
            # First click - set start corner
            state.rect_start = (x_display, y_display)
            state.selection_step = 1
            
            # Draw a marker at the first click point
            display = prepare_for_display(state.processed_image, target_size=(512, 512))
            cv2.circle(display, (x_display, y_display), 8, (14, 165, 233), -1)
            cv2.circle(display, (x_display, y_display), 8, (255, 255, 255), 2)
            
            return display, "Step 2: Click to set the SECOND corner of your selection rectangle."
        
        else:
            # Second click - set end corner and perform segmentation
            state.rect_end = (x_display, y_display)
            state.selection_step = 0
            
            # Calculate bounding box in SAM coordinates
            x1_sam = int(state.rect_start[0] * 2)
            y1_sam = int(state.rect_start[1] * 2)
            x2_sam = int(state.rect_end[0] * 2)
            y2_sam = int(state.rect_end[1] * 2)
            
            # Ensure proper ordering
            box = (
                min(x1_sam, x2_sam),
                min(y1_sam, y2_sam),
                max(x1_sam, x2_sam),
                max(y1_sam, y2_sam)
            )
            
            # Get segmentation using box prompt
            segmenter = get_segmenter()
            masks, scores, _ = segmenter.segment_box(box)
            
            # Get the best mask
            mask = masks[0]
            score = scores[0]
            
            state.current_mask = mask
            state.mask_score = score
            state.bounding_box = box
            
            # Create overlay with segmentation
            overlay = overlay_mask(state.processed_image.copy(), mask)
            
            # Draw the selection rectangle on top
            display = prepare_for_display(overlay, target_size=(512, 512))
            display = draw_selection_rectangle(
                display, 
                state.rect_start, 
                state.rect_end,
                is_preview=False
            )
            
            # Create and store comparison view
            original_display = prepare_for_display(state.processed_image, target_size=(400, 400))
            overlay_small = prepare_for_display(overlay, target_size=(400, 400))
            state.comparison_image = create_comparison_view(original_display, overlay_small, padding=5)
            
            # Calculate mask stats
            mask_stats = calculate_mask_stats(mask)
            
            info = f"""Segmentation Complete!
Score: {score:.3f}
Regions Found: {mask_stats['num_regions']}
Total Area: {mask_stats['total_area']:,} pixels

Click again to make a new selection."""
            
            return display, info
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return image, f"Error during segmentation: {str(e)}"


def clear_segmentation():
    """Clear current segmentation and start fresh."""
    state.reset_selection()
    
    if state.processed_image is not None:
        display = prepare_for_display(state.processed_image, target_size=(512, 512))
        return display, "Selection cleared. Click to set the FIRST corner of a new selection."
    return None, "No image loaded."


def generate_report():
    """Generate diagnostic report for current segmentation."""
    if state.processed_image is None:
        return "Please upload an image first."
    
    if state.current_mask is None:
        return "Please select and segment a region first by clicking two corners on the image."
    
    try:
        reporter = get_reporter()
        
        # Generate report - pass mask so it only analyzes the segmented region
        report_dict = reporter.generate_medical_report(
            state.processed_image,
            mask=state.current_mask
        )
        
        # Format report
        formatted = reporter.format_report(report_dict)
        
        return formatted
        
    except Exception as e:
        return f"Error generating report: {str(e)}"


def export_results():
    """Export segmented image, comparison, and report."""
    if state.current_mask is None:
        return None, None, None, "No segmentation to export. Please select a region first."
    
    try:
        # Create overlay image
        overlay = overlay_mask(state.processed_image.copy(), state.current_mask)
        
        # Convert to PIL for saving
        overlay_pil = Image.fromarray(overlay)
        
        # Get comparison image
        comparison = state.comparison_image
        
        # Generate report
        reporter = get_reporter()
        report = reporter.format_report(
            reporter.generate_medical_report(state.processed_image, state.current_mask)
        )
        
        return overlay_pil, comparison, report, "Export ready! Right-click images to save."
        
    except Exception as e:
        return None, None, None, f"Error exporting: {str(e)}"


def adjust_window_level(image, window_center, window_width):
    """Adjust window/level for X-ray viewing."""
    if state.original_image is None:
        return image, image
    
    try:
        # Convert to grayscale for windowing
        gray = cv2.cvtColor(state.original_image, cv2.COLOR_RGB2GRAY)
        
        # Apply windowing
        windowed = apply_windowing(gray, window_center, window_width)
        
        # Convert back to RGB
        rgb = cv2.cvtColor(windowed, cv2.COLOR_GRAY2RGB)
        
        # Resize for SAM
        sam_ready, _ = resize_for_sam(rgb, target_size=1024)
        
        # Update state
        state.processed_image = sam_ready
        state.reset_selection()
        
        # Re-set image in segmenter
        segmenter = get_segmenter()
        segmenter.set_image(sam_ready)
        
        display = prepare_for_display(sam_ready, target_size=(512, 512))
        
        return display, display.copy()
        
    except Exception as e:
        return image, image


# Custom CSS for medical theme
CUSTOM_CSS = """
/* Medical Dark Theme */
:root {
    --primary: #0EA5E9;
    --primary-dark: #0284C7;
    --secondary: #06B6D4;
    --background: #0F172A;
    --surface: #1E293B;
    --surface-light: #334155;
    --text: #F8FAFC;
    --text-muted: #94A3B8;
    --success: #22C55E;
    --warning: #F59E0B;
    --error: #EF4444;
}

.gradio-container {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%) !important;
    font-family: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.main-title {
    text-align: center;
    background: linear-gradient(90deg, #0EA5E9, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
}

.subtitle {
    text-align: center;
    color: #94A3B8 !important;
    font-size: 1.1rem !important;
    margin-bottom: 2rem !important;
}

/* Image containers - same height */
.image-panel img {
    height: 450px !important;
    object-fit: contain !important;
}

/* Panel styling */
.panel {
    background: rgba(30, 41, 59, 0.8) !important;
    border: 1px solid rgba(51, 65, 85, 0.5) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    backdrop-filter: blur(10px);
}

/* Image containers */
.image-container {
    background: #0F172A !important;
    border: 2px solid #334155 !important;
    border-radius: 8px !important;
    overflow: hidden;
}

.image-container:hover {
    border-color: #0EA5E9 !important;
    box-shadow: 0 0 20px rgba(14, 165, 233, 0.2);
}

/* Buttons */
.primary-btn {
    background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 20px rgba(14, 165, 233, 0.4) !important;
}

.secondary-btn {
    background: #334155 !important;
    border: 1px solid #475569 !important;
    color: #F8FAFC !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
}

.danger-btn {
    background: rgba(239, 68, 68, 0.2) !important;
    border: 1px solid #EF4444 !important;
    color: #EF4444 !important;
}

/* Sliders */
input[type="range"] {
    accent-color: #0EA5E9 !important;
}

/* Text areas */
textarea, .prose {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    color: #F8FAFC !important;
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    border-radius: 8px !important;
}

/* Labels */
label {
    color: #94A3B8 !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Info cards */
.info-card {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.1) 0%, rgba(6, 182, 212, 0.1) 100%);
    border-left: 3px solid #0EA5E9;
    padding: 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1rem 0;
}

/* Accordion */
.accordion {
    background: #1E293B !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #64748B;
    font-size: 0.875rem;
    margin-top: 2rem;
    padding: 1rem;
    border-top: 1px solid #334155;
}

/* Hide default gradio footer */
footer {
    display: none !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #1E293B;
}

::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #64748B;
}

/* Selection instruction styling */
.selection-instruction {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.15) 0%, rgba(6, 182, 212, 0.15) 100%);
    border: 1px solid rgba(14, 165, 233, 0.3);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: #F8FAFC;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}
"""


def create_app():
    """Create and configure the Gradio application."""
    
    with gr.Blocks(
        title="Medical Image Analysis Platform",
        theme=gr.themes.Base(
            primary_hue="sky",
            secondary_hue="cyan",
            neutral_hue="slate",
            font=["Inter", "system-ui", "sans-serif"],
        ),
        css=CUSTOM_CSS
    ) as app:
        
        # Header
        gr.HTML("""
            <div style="text-align: center; padding: 2rem 0 1rem 0;">
                <h1 class="main-title">Medical Image Analysis Platform</h1>
                <p class="subtitle">AI-Powered Chest X-Ray Segmentation and Diagnostic Report Generation</p>
            </div>
        """)
        
        with gr.Row():
            # Left Panel - Image Upload
            with gr.Column(scale=1):
                gr.HTML('<h3 style="color: #F8FAFC; margin-bottom: 1rem;">Upload X-Ray</h3>')
                
                input_image = gr.Image(
                    label="Upload X-Ray Image",
                    type="numpy",
                    height=450,
                    sources=["upload", "clipboard"],
                    elem_classes=["image-container", "image-panel"]
                )
                
                with gr.Row():
                    enhance_checkbox = gr.Checkbox(
                        label="Apply Contrast Enhancement",
                        value=True,
                        info="Recommended for X-ray images"
                    )
                
                status_text = gr.Textbox(
                    label="Status",
                    value="Upload an image to begin analysis.",
                    interactive=False,
                    lines=2
                )
            
            # Center Panel - Segmentation View
            with gr.Column(scale=1):
                gr.HTML('<h3 style="color: #F8FAFC; margin-bottom: 1rem;">Interactive Segmentation</h3>')
                
                # Display-only image for segmentation (no upload sources)
                segmentation_image = gr.Image(
                    label="Click to Select Region",
                    type="numpy",
                    height=450,
                    sources=[],  # No upload - display only
                    interactive=True,  # Allow clicks
                    elem_classes=["image-container", "image-panel"]
                )
                
                # Instructions and info BELOW the image
                gr.HTML("""
                    <div class="selection-instruction">
                        <strong>How to select:</strong> Click once to set the first corner, click again to set the second corner. 
                        The AI will segment the region within your rectangle.
                    </div>
                """)
                
                mask_info = gr.Textbox(
                    label="Segmentation Info",
                    value="Upload an image, then click to select a region.",
                    interactive=False,
                    lines=4
                )
                
                with gr.Row():
                    clear_btn = gr.Button(
                        "Clear Selection",
                        elem_classes=["secondary-btn"]
                    )
                    generate_btn = gr.Button(
                        "Generate Report",
                        elem_classes=["primary-btn"]
                    )
            
            # Right Panel - Results
            with gr.Column(scale=1):
                gr.HTML('<h3 style="color: #F8FAFC; margin-bottom: 1rem;">Analysis Results</h3>')
                
                with gr.Tabs():
                    with gr.TabItem("Report"):
                        report_text = gr.Textbox(
                            label="Diagnostic Report",
                            placeholder="Select a region and click 'Generate Report' to create AI-assisted diagnostic description...",
                            lines=18,
                            interactive=False
                        )
                    
                    with gr.TabItem("Export"):
                        export_btn = gr.Button(
                            "Prepare Export",
                            elem_classes=["primary-btn"]
                        )
                        
                        gr.HTML('<p style="color: #94A3B8; font-size: 0.875rem; margin: 0.5rem 0;">Segmented Image:</p>')
                        export_image = gr.Image(
                            label="Segmented Image (Right-click to save)",
                            height=180
                        )
                        
                        gr.HTML('<p style="color: #94A3B8; font-size: 0.875rem; margin: 0.5rem 0;">Comparison View:</p>')
                        export_comparison = gr.Image(
                            label="Comparison (Right-click to save)",
                            height=130
                        )
                        
                        export_report = gr.Textbox(
                            label="Report Text",
                            lines=4
                        )
                        export_status = gr.Textbox(
                            label="Export Status",
                            interactive=False
                        )
        
        # Window/Level controls
        with gr.Accordion("Window/Level Adjustment", open=False):
            gr.HTML('<p style="color: #94A3B8; font-size: 0.875rem;">Adjust brightness and contrast like a DICOM viewer</p>')
            with gr.Row():
                window_center = gr.Slider(
                    minimum=0, maximum=255, value=127,
                    label="Window Center (Brightness)"
                )
                window_width = gr.Slider(
                    minimum=1, maximum=255, value=255,
                    label="Window Width (Contrast)"
                )
        
        # Instructions
        with gr.Accordion("How to Use", open=False):
            gr.HTML("""
                <div style="color: #94A3B8; padding: 1rem;">
                    <h4 style="color: #F8FAFC;">Quick Start Guide</h4>
                    <ol style="line-height: 1.8;">
                        <li><strong>Upload</strong> - Drag and drop or click to upload a chest X-ray image</li>
                        <li><strong>Select Region</strong> - Click on the segmentation panel to set the first corner of your rectangle</li>
                        <li><strong>Complete Selection</strong> - Click again to set the second corner - AI will automatically segment</li>
                        <li><strong>Generate Report</strong> - Click "Generate Report" for AI-assisted diagnostic description</li>
                        <li><strong>Export</strong> - Go to Export tab to download images and report</li>
                    </ol>
                    <p style="margin-top: 1rem; padding: 1rem; background: rgba(239, 68, 68, 0.1); border-left: 3px solid #EF4444; border-radius: 0 8px 8px 0;">
                        <strong>Disclaimer:</strong> This tool is for research and educational purposes only. 
                        AI-generated reports should not be used for clinical diagnosis.
                    </p>
                </div>
            """)
        
        # Footer
        gr.HTML("""
            <div class="footer">
                <p>Medical Image Analysis Platform | Powered by SAM + BLIP | For Research Use Only</p>
            </div>
        """)
        
        # Event Handlers
        input_image.upload(
            fn=process_uploaded_image,
            inputs=[input_image, enhance_checkbox],
            outputs=[input_image, segmentation_image, status_text, mask_info]
        )
        
        # Handle clicks on segmentation image for rectangle selection
        segmentation_image.select(
            fn=handle_segmentation_click,
            inputs=[segmentation_image],
            outputs=[segmentation_image, mask_info]
        )
        
        clear_btn.click(
            fn=clear_segmentation,
            outputs=[segmentation_image, mask_info]
        )
        
        generate_btn.click(
            fn=generate_report,
            outputs=[report_text]
        )
        
        export_btn.click(
            fn=export_results,
            outputs=[export_image, export_comparison, export_report, export_status]
        )
        
        # Window/Level adjustment
        window_center.change(
            fn=adjust_window_level,
            inputs=[input_image, window_center, window_width],
            outputs=[input_image, segmentation_image]
        )
        window_width.change(
            fn=adjust_window_level,
            inputs=[input_image, window_center, window_width],
            outputs=[input_image, segmentation_image]
        )
    
    return app


if __name__ == "__main__":
    print("=" * 60)
    print("Medical Image Analysis Platform")
    print("=" * 60)
    print("Initializing application...")
    print()
    
    app = create_app()
    
    print()
    print("Starting server...")
    print("=" * 60)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
