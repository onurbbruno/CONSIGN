"""
Usage:
    python run_pipeline.py --category animals
    python run_pipeline.py --category vehicles
    python run_pipeline.py --category animals --skip-download
    python run_pipeline.py --category vehicles --skip-preprocess --device cuda
"""
import argparse

from download import ensure_all
from preprocess import preprocess_images
from inference import run_inference


def main():
    parser = argparse.ArgumentParser(
        description="COCO DeepLab inference pipeline for conformal calibration."
    )
    parser.add_argument(
        "--category", type=str, required=True, choices=["animals", "vehicles"],
        help="Category to process: 'animals' or 'vehicles'",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download step (assume data/models already present)",
    )
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip preprocessing step (assume processed images already exist)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="PyTorch device, e.g. 'cuda' or 'cpu' (default: auto-detect)",
    )
    args = parser.parse_args()

    # Step 1: Download
    if not args.skip_download:
        print("\n" + "=" * 60)
        print("STEP 1: Downloading data and models")
        print("=" * 60)
        ensure_all()
    else:
        print("Skipping download step.")

    # Step 2: Preprocess
    if not args.skip_preprocess:
        print("\n" + "=" * 60)
        print("STEP 2: Preprocessing images")
        print("=" * 60)
        preprocess_images(args.category)
    else:
        print("Skipping preprocessing step.")

    # Step 3: Inference
    print("\n" + "=" * 60)
    print("STEP 3: Running inference")
    print("=" * 60)

    import torch
    device = torch.device(args.device) if args.device else None
    out_dir = run_inference(args.category, device=device)

    print("\n" + "=" * 60)
    print(f"Pipeline complete. Output in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
