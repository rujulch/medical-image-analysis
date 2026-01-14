# AI-Assisted Medical Image Analysis Platform

An intelligent diagnostic tool for chest X-ray analysis that combines Meta's Segment Anything Model (SAM) with vision-language models (BLIP) to perform precise pathological segmentation and automated diagnostic report generation.

## Project Overview

This platform addresses critical needs in AI-assisted radiology by providing:

- **Interactive Segmentation**: Rectangle-based region selection with SAM-powered intelligent segmentation
- **Automated Report Generation**: Natural language diagnostic descriptions using BLIP vision-language model
- **Medical Image Preprocessing**: CLAHE enhancement, window/level adjustment, and DICOM-style controls
- **Professional Interface**: Dark-themed medical imaging interface with real-time visualization

## Technical Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        User Interface (Gradio)                         │
│  ┌──────────────┐  ┌──────────────────────┐  ┌───────────────────────┐ │
│  │   Upload     │  │ Interactive          │  │   Analysis Results   │ │
│  │   Panel      │  │ Segmentation Panel   │  │   (Report/Export)    │ │
│  └──────────────┘  └──────────────────────┘  └───────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │ SAM Segmentation  │           │ BLIP Report Gen   │
        │ (segment_anything)│           │ (transformers)    │
        └───────────────────┘           └───────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │     Preprocessing Pipeline    │
                    │  (CLAHE, Windowing, Resize)   │
                    └───────────────────────────────┘
```

## Key Features

### 1. SAM-Based Segmentation
- Utilizes Meta's Segment Anything Model (vit_b variant optimized for 6GB GPU)
- Box prompt-based segmentation for precise region selection
- Automatic detection of most prominent structure within selection

### 2. Vision-Language Report Generation
- BLIP model generates clinical-style descriptions of segmented regions
- Reports focus exclusively on selected regions (not full image)
- Medical terminology enhancement for professional output

### 3. Medical Image Processing
- CLAHE (Contrast Limited Adaptive Histogram Equalization) for X-ray enhancement
- Window/Level controls simulating DICOM viewer functionality
- Support for standard image formats (PNG, JPG) and DICOM

### 4. Evaluation Metrics
- Dice Coefficient for segmentation accuracy
- IoU (Intersection over Union) scoring
- Mask statistics (area, regions, centroids)

## Technology Stack

| Component | Technology |
|-----------|------------|
| Deep Learning | PyTorch 2.5+ with CUDA support |
| Segmentation | Segment Anything Model (SAM) |
| Vision-Language | BLIP (Salesforce) via HuggingFace |
| Image Processing | OpenCV, scikit-image, PIL |
| Web Interface | Gradio 6.x |
| Data Analysis | NumPy, Pandas, Matplotlib |

## Dataset

This project uses the **NIH Chest X-Ray Dataset**:
- 112,120 frontal-view chest X-ray images
- 30,805 unique patients
- 14 disease labels including Pneumonia, Mass, Nodule, Cardiomegaly
- Source: NIH Clinical Center via Kaggle

## Installation

### Prerequisites
- Python 3.10 or 3.11 (Python 3.14 not supported)
- NVIDIA GPU with 6GB+ VRAM (recommended)
- CUDA 12.1 compatible drivers

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/medical-image-analysis.git
cd medical-image-analysis

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate

# Activate environment (Linux/Mac)
source venv/bin/activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# Run setup to download model weights
python setup.py --all
```

### Dataset Setup

1. Download the NIH Chest X-Ray dataset from [Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)
2. Place images in `data/raw/images/`
3. Place `Data_Entry_2017.csv` in `data/raw/`

## Usage

### Running the Application

```bash
python app/app.py
```

Access the web interface at `http://localhost:7860`

### Workflow

1. **Upload**: Drag and drop a chest X-ray image
2. **Select Region**: Click once to set first corner, click again to set second corner
3. **View Segmentation**: AI automatically segments the region within the rectangle
4. **Generate Report**: Click "Generate Report" for diagnostic description
5. **Export**: Download segmented images and reports

## Project Structure

```
medical-image-analysis/
├── app/
│   └── app.py                  # Main Gradio application
├── src/
│   ├── preprocessing/
│   │   ├── image_loader.py     # Image loading utilities
│   │   └── transforms.py       # Image transformations
│   ├── segmentation/
│   │   └── sam_predictor.py    # SAM model integration
│   ├── report_generation/
│   │   └── blip_reporter.py    # BLIP report generator
│   └── utils/
│       ├── visualization.py    # Overlay and comparison tools
│       └── metrics.py          # Evaluation metrics
├── notebooks/
│   ├── dataset_exploration.py  # Dataset analysis
│   └── model_demo.py           # Model demonstration
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Processed outputs
├── models/
│   └── weights/                # Model weights (auto-downloaded)
├── config.py                   # Configuration settings
├── requirements.txt            # Package dependencies
├── requirements.lock           # Locked versions
├── setup.py                    # Setup and verification
├── Dockerfile                  # Docker configuration
└── docker-compose.yml          # Docker Compose setup
```

## Configuration

Key settings can be modified in `config.py`:

- `SAM_CONFIG`: Model type and checkpoint settings
- `BLIP_CONFIG`: Report generation parameters
- `IMAGE_CONFIG`: Image sizes and supported formats
- `UI_CONFIG`: Interface colors and theming

## Performance

### Hardware Requirements

| Configuration | GPU Memory | Processing Time |
|--------------|------------|-----------------|
| Minimum | 6GB VRAM | ~20-30 sec/image |
| Recommended | 8GB+ VRAM | ~10-15 sec/image |
| CPU Only | N/A | ~60-90 sec/image |

### Model Specifications

- **SAM (vit_b)**: 375 MB weights, optimized for memory efficiency
- **BLIP (base)**: 990 MB weights, balanced accuracy/speed

## Docker Deployment

```bash
# Build and run
docker-compose up --build

# With GPU support
docker-compose --profile gpu up --build
```

## Limitations

- Reports are AI-generated and intended for research purposes only
- Not validated for clinical diagnosis
- Performance depends on image quality and pathology visibility
- Limited to chest X-ray images (not trained on CT/MRI)

## Future Work

- Fine-tuning SAM on medical imaging datasets (MedSAM integration)
- Multi-modal fusion with additional clinical data
- Support for 3D volumetric imaging (CT/MRI)
- Integration with PACS systems

## References

1. Kirillov, A., et al. "Segment Anything." arXiv:2304.02643 (2023)
2. Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training." ICML (2022)
3. Wang, X., et al. "ChestX-ray8: Hospital-scale Chest X-ray Database." CVPR (2017)

## License

This project is for educational and research purposes only. The NIH Chest X-Ray dataset is publicly available for non-commercial research.

## Acknowledgments

- Meta AI for Segment Anything Model
- Salesforce Research for BLIP
- NIH Clinical Center for the Chest X-Ray dataset
- HuggingFace for model hosting and transformers library
