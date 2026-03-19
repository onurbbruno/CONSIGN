import os
import subprocess
import zipfile
import urllib.request

from config import (
    DATA_DIR, COCO_IMAGE_DIR, COCO_ANN_FILE,
    COCO_IMAGES_URL, COCO_ANNOTATIONS_URL,
    VAINF_REPO_DIR, VAINF_REPO_URL,
    WEIGHTS_DIR, WEIGHT_URLS, MODEL_NAMES, weight_path,
)


def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"  Already exists: {dest_path}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"  Saved to {dest_path}")


def download_and_extract_zip(url, extract_to, zip_name):
    zip_path = os.path.join(extract_to, zip_name)
    if not os.path.exists(zip_path):
        download_file(url, zip_path)
    print(f"  Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"  Extracted to {extract_to}")


def ensure_coco_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.isdir(COCO_IMAGE_DIR) or len(os.listdir(COCO_IMAGE_DIR)) < 100:
        print("[1/2] Downloading COCO 2017 val images...")
        download_and_extract_zip(COCO_IMAGES_URL, DATA_DIR, "val2017.zip")
    else:
        print("[1/2] COCO val images already present.")

    if not os.path.isfile(COCO_ANN_FILE):
        print("[2/2] Downloading COCO 2017 annotations...")
        download_and_extract_zip(COCO_ANNOTATIONS_URL, DATA_DIR, "annotations_trainval2017.zip")
    else:
        print("[2/2] COCO annotations already present.")


def ensure_vainf_repo():
    if os.path.isdir(VAINF_REPO_DIR):
        print("VainF repo already cloned.")
        return
    os.makedirs(os.path.dirname(VAINF_REPO_DIR), exist_ok=True)
    print(f"Cloning {VAINF_REPO_URL} ...")
    subprocess.run(["git", "clone", VAINF_REPO_URL, VAINF_REPO_DIR], check=True)
    print("VainF repo cloned.")


def ensure_weights():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for name in MODEL_NAMES:
        dest = weight_path(name)
        if os.path.isfile(dest):
            print(f"  Weight already present: {name}")
            continue
        url = WEIGHT_URLS[name]
        print(f"  Downloading weights for {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  Saved: {dest}")


def ensure_all():
    print("=" * 60)
    print("Step 1: COCO 2017 val dataset")
    print("=" * 60)
    ensure_coco_data()

    print("\n" + "=" * 60)
    print("Step 2: VainF DeepLabV3Plus-Pytorch repository")
    print("=" * 60)
    ensure_vainf_repo()

    print("\n" + "=" * 60)
    print("Step 3: Pretrained VOC weights")
    print("=" * 60)
    ensure_weights()

    print("\nAll downloads complete.")


if __name__ == "__main__":
    ensure_all()
