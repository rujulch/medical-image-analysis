"""
Setup script for Medical Image Analysis Platform.
Run this script to verify installation and download required models.
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """Install required packages."""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def download_sam_weights():
    """Download SAM model weights."""
    print("\nDownloading SAM model weights...")
    
    weights_dir = Path("models/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = weights_dir / "sam_vit_b_01ec64.pth"
    
    if checkpoint_path.exists():
        print(f"SAM weights already exist: {checkpoint_path}")
        return
    
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    import urllib.request
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = '=' * filled + '-' * (bar_length - filled)
        print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)
    
    print(f"Downloading from {url}")
    print("This may take a few minutes (~375 MB)...")
    
    urllib.request.urlretrieve(url, checkpoint_path, show_progress)
    print(f"\nDownload complete: {checkpoint_path}")


def verify_installation():
    """Verify all components are working."""
    print("\n" + "=" * 50)
    print("Verifying installation...")
    print("=" * 50)
    
    errors = []
    
    # Check PyTorch
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"[OK] PyTorch {torch.__version__}")
        if cuda_available:
            print(f"     GPU: {torch.cuda.get_device_name(0)}")
            print(f"     Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("     WARNING: CUDA not available, using CPU (slower)")
    except ImportError as e:
        errors.append(f"PyTorch: {e}")
        print(f"[FAIL] PyTorch: {e}")
    
    # Check SAM
    try:
        from segment_anything import sam_model_registry
        print("[OK] Segment Anything Model (SAM)")
    except ImportError as e:
        errors.append(f"SAM: {e}")
        print(f"[FAIL] SAM: {e}")
        print("     Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
    
    # Check Transformers (BLIP)
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        print("[OK] HuggingFace Transformers (BLIP)")
    except ImportError as e:
        errors.append(f"Transformers: {e}")
        print(f"[FAIL] Transformers: {e}")
    
    # Check Gradio
    try:
        import gradio
        print(f"[OK] Gradio {gradio.__version__}")
    except ImportError as e:
        errors.append(f"Gradio: {e}")
        print(f"[FAIL] Gradio: {e}")
    
    # Check OpenCV
    try:
        import cv2
        print(f"[OK] OpenCV {cv2.__version__}")
    except ImportError as e:
        errors.append(f"OpenCV: {e}")
        print(f"[FAIL] OpenCV: {e}")
    
    # Check SAM weights
    weights_path = Path("models/weights/sam_vit_b_01ec64.pth")
    if weights_path.exists():
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"[OK] SAM weights ({size_mb:.1f} MB)")
    else:
        print("[WARN] SAM weights not downloaded")
        print("     Run: python setup.py --download-weights")
    
    print("\n" + "=" * 50)
    if errors:
        print(f"Setup completed with {len(errors)} error(s)")
        print("Fix the errors above and run setup again.")
    else:
        print("Setup completed successfully!")
        print("\nTo start the application, run:")
        print("  python app/app.py")
    print("=" * 50)
    
    return len(errors) == 0


def main():
    """Main setup function."""
    print("=" * 50)
    print("Medical Image Analysis Platform - Setup")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true", help="Install requirements")
    parser.add_argument("--download-weights", action="store_true", help="Download SAM weights")
    parser.add_argument("--verify", action="store_true", help="Verify installation")
    parser.add_argument("--all", action="store_true", help="Run all setup steps")
    args = parser.parse_args()
    
    # Default to --all if no args provided
    if not any([args.install, args.download_weights, args.verify, args.all]):
        args.all = True
    
    if args.all or args.install:
        install_requirements()
    
    if args.all or args.download_weights:
        download_sam_weights()
    
    if args.all or args.verify:
        success = verify_installation()
        return 0 if success else 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

