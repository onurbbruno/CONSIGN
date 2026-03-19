import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image
import skimage.io as io

from config import COCO_IMAGE_DIR, COCO_ANN_FILE, DATA_DIR, CATEGORIES, CROP_SIZE


def crop_or_pad(img_array, size, pad_value=0):
    """Center crop or pad an image/mask to the desired size."""
    h, w = img_array.shape[:2]
    ch, cw = size
    top = max((h - ch) // 2, 0)
    left = max((w - cw) // 2, 0)
    cropped = img_array[top:top+ch, left:left+cw]

    pad_h = max(ch - cropped.shape[0], 0)
    pad_w = max(cw - cropped.shape[1], 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if img_array.ndim == 3:
        padded = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        constant_values=pad_value)
    else:
        padded = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)),
                        constant_values=pad_value)
    return padded


def create_multilabel_mask(annotations, height, width):
    """Create a multi-label mask from COCO annotations.
    Pixel values are COCO category IDs."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        cat_id = ann['category_id']
        rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
        ann_mask = maskUtils.decode(rle)
        if ann_mask.ndim == 3:
            ann_mask = np.any(ann_mask, axis=2).astype(np.uint8)
        mask[ann_mask == 1] = cat_id
    return mask


def get_filtered_image_ids(coco, category_key):
    """Return COCO image IDs where ALL annotations belong to the target classes."""
    cat_config = CATEGORIES[category_key]
    target_cat_ids = set(coco.getCatIds(catNms=cat_config["coco_names"]))

    img_ids_candidates = set()
    for cat_id in target_cat_ids:
        img_ids_candidates.update(coco.getImgIds(catIds=[cat_id]))

    filtered_ids = []
    for img_id in img_ids_candidates:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_cat_ids = set(ann['category_id'] for ann in anns)
        if img_cat_ids.issubset(target_cat_ids):
            filtered_ids.append(img_id)

    return sorted(filtered_ids)


def preprocess_images(category_key):
    """Filter, crop/pad, and save images and masks for the given category.

    Returns (output_image_dir, output_mask_dir, num_processed).
    """
    output_dir = os.path.join(DATA_DIR, f"processed_{category_key}")
    output_image_dir = os.path.join(output_dir, "images")
    output_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    coco = COCO(COCO_ANN_FILE)
    img_ids = get_filtered_image_ids(coco, category_key)
    print(f"Found {len(img_ids)} images for category '{category_key}'.")

    processed = 0
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        filename = img_info['file_name']
        img_path = os.path.join(COCO_IMAGE_DIR, filename)
        image = io.imread(img_path)
        height, width = img_info['height'], img_info['width']

        ann_ids = coco.getAnnIds(imgIds=img_id)
        annotations = coco.loadAnns(ann_ids)
        if not annotations:
            continue

        multi_label_mask = create_multilabel_mask(annotations, height, width)
        processed_image = crop_or_pad(image, CROP_SIZE, pad_value=0)
        processed_mask = crop_or_pad(multi_label_mask, CROP_SIZE, pad_value=0)

        Image.fromarray(processed_image).save(os.path.join(output_image_dir, filename))
        mask_filename = os.path.splitext(filename)[0] + '.png'
        Image.fromarray(processed_mask).save(os.path.join(output_mask_dir, mask_filename))
        processed += 1

    print(f"Processed and saved {processed} images and masks to {output_dir}")
    return output_image_dir, output_mask_dir, processed
