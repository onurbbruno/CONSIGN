import subprocess
import sys
import argparse

SCRIPTS = [
    "./COCO_test/run_pipeline.py",
    "./compare_SVD_PW.py",
    "./visualize_pred.py",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Run CP pipeline.")
    parser.add_argument("--category", type=str, required=True, choices=["animals", "vehicles"])
    parser.add_argument("--skip-download", action="store_true", help="download data")
    parser.add_argument("--skip-preprocess", action="store_true", help="preprocess data")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return parser.parse_args()

def run_script(index, total, script, extra_args=[]):
    print(f"\n{'='*50}")
    print(f"{'='*50}")
    print(f"[{index}/{total}] Running: {script}")
    print('='*50)
    print(f"{'='*50}")

    result = subprocess.run(
        [sys.executable, script, *extra_args],
        capture_output=False  # streams output directly to terminal
    )

    if index==1:
        print(f"\n Generation of softmax completed successfully.")
    elif index==2:
        print(f"\n Conformal prediction completed successfully.")

if __name__ == "__main__":
    args = parse_args()

    scriptCOCO_args = ["--category", args.category]
    if args.skip_download:
        scriptCOCO_args.append("--skip-download")
    if args.skip_preprocess:
        scriptCOCO_args.append("--skip-preprocess")
    if args.device:
        scriptCOCO_args.extend(["--device", args.device])

    run_script(1, 3, SCRIPTS[0], scriptCOCO_args)
    run_script(2, 3, SCRIPTS[1])
    run_script(3, 3, SCRIPTS[2])

    print(f"\nTest completed successfully.")
