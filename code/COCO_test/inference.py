import os
import sys
import numpy as np
import pickle
import torch
from PIL import Image
from torchvision import transforms

from config import (
    VAINF_REPO_DIR, OUTPUT_DIR,
    MODEL_NAMES, weight_path,
    CATEGORIES, COCO_TO_VOC,
    NUM_CLASSES_VOC, NUM_CLASSES_CUSTOM,
    CENTER_CROP, IMAGENET_MEAN, IMAGENET_STD,
    DATA_DIR,
)


def _add_vainf_to_path():
    if VAINF_REPO_DIR not in sys.path:
        sys.path.insert(0, VAINF_REPO_DIR)


def load_models(device):
    """Load all 6 DeepLab models with pretrained VOC weights."""
    _add_vainf_to_path()
    from network import modeling

    models = []
    for name in MODEL_NAMES:
        print(f"  Loading {name} ...")
        model = modeling.__dict__[name](num_classes=NUM_CLASSES_VOC, output_stride=16)
        checkpoint = torch.load(weight_path(name), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        models.append(model)
    print(f"  Loaded {len(models)} models on {device}.")
    return models


def _build_coco_to_custom_map(category_key):
    """Build COCO_id -> custom_id mapping (going through VOC as intermediate)."""
    voc_to_custom = CATEGORIES[category_key]["voc_to_custom"]
    mapping = {}
    for coco_id, voc_id in COCO_TO_VOC.items():
        mapping[coco_id] = voc_to_custom.get(voc_id, 0)
    return mapping


def _extract_custom_softmax(full_softmax, voc_ids):
    """Extract 8-class softmax (bg + 7 targets) from 21-class model output.

    Background channel = 1 - sum(target channels).
    """
    result = np.zeros((NUM_CLASSES_CUSTOM, full_softmax.shape[1], full_softmax.shape[2]),
                      dtype=np.float32)
    target_probs = full_softmax[voc_ids]  # (7, H, W)
    result[1:] = target_probs
    result[0] = 1.0 - np.sum(target_probs, axis=0)
    return np.clip(result, 0.0, 1.0)


def _remap_mask(mask_coco, coco_to_custom):
    """Remap COCO category IDs in a mask to custom class indices (0-7)."""
    result = np.zeros_like(mask_coco, dtype=np.int64)
    for coco_id, custom_id in coco_to_custom.items():
        result[mask_coco == coco_id] = custom_id
    return result


def run_inference(category_key, device=None):
    """Run 6 models on all preprocessed images, save softmax/labels/images as pickles.

    Output shapes:
        smx.pkl:    list of (6, 8, 256, 256) arrays
        labels.pkl: list of (256, 256) arrays
        images.pkl: list of (256, 256, 3) arrays
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    voc_ids = CATEGORIES[category_key]["voc_ids"]
    coco_to_custom = _build_coco_to_custom_map(category_key)

    processed_dir = os.path.join(DATA_DIR, f"processed_{category_key}")
    image_dir = os.path.join(processed_dir, "images")
    mask_dir = os.path.join(processed_dir, "masks")

    image_filenames = os.listdir(image_dir)
    mask_filenames = os.listdir(mask_dir)

    mask_lookup = {
        os.path.splitext(f)[0]: f
        for f in os.listdir(mask_dir)
    }

    assert len(image_filenames) == len(mask_filenames), (
        f"Mismatch: {len(image_filenames)} images vs {len(mask_filenames)} masks"
    )
    print(f"Running inference on {len(image_filenames)} images ({category_key})...")

    models = load_models(device)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    c = CENTER_CROP
    offset = (512 - c) // 2  # 128

    softmax_all = []
    labels_all = []
    images_all = []

    for i, img_fn in enumerate(image_filenames):
        stem = os.path.splitext(img_fn)[0]
        mask_fn = mask_lookup.get(stem)

        input_image = Image.open(os.path.join(image_dir, img_fn)).convert("RGB")
        input_tensor = preprocess(input_image).unsqueeze(0)  # (1, 3, 512, 512)

        # Store normalized center-cropped image
        img_np = input_tensor[0, :, offset:offset+c, offset:offset+c].numpy()
        images_all.append(np.transpose(img_np, (1, 2, 0)))

        # Load mask, remap COCO->custom, center crop
        mask_coco = np.array(Image.open(os.path.join(mask_dir, mask_fn)), dtype=np.int64)
        mask_custom = _remap_mask(mask_coco, coco_to_custom)
        labels_all.append(mask_custom[offset:offset+c, offset:offset+c])

        # Run all 6 models
        model_outputs = []
        input_batch = input_tensor.to(device)

        for model in models:
            with torch.no_grad():
                output = model(input_batch)  # (1, 21, H, W) - VainF returns plain tensor
                softmax_full = torch.softmax(output[0], dim=0).cpu().numpy()  # (21, 512, 512)

            custom_smx = _extract_custom_softmax(softmax_full, voc_ids)
            model_outputs.append(custom_smx[:, offset:offset+c, offset:offset+c])

        softmax_all.append(np.array(model_outputs))  # (6, 8, 256, 256)

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processed {i + 1}/{len(image_filenames)} images")

    # Save outputs
    out_dir = os.path.join(OUTPUT_DIR, 'COCO_'+category_key)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, f"smx_COCO_{category_key}.pkl"), "wb") as f:
        pickle.dump(softmax_all, f)
    with open(os.path.join(out_dir, f"labels_COCO_{category_key}.pkl"), "wb") as f:
        pickle.dump(labels_all, f)
    with open(os.path.join(out_dir, "imgs.pkl"), "wb") as f:
        pickle.dump(images_all, f)

    print(f"\nSaved {len(softmax_all)} samples to {out_dir}/")
    return out_dir
