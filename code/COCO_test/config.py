import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_DIR = os.path.join(MODELS_DIR, "weights")
OUTPUT_DIR = os.path.join(BASE_DIR, "../softmax")

COCO_IMAGE_DIR = os.path.join(DATA_DIR, "val2017")
COCO_ANN_FILE = os.path.join(DATA_DIR, "annotations", "instances_val2017.json")

VAINF_REPO_DIR = os.path.join(MODELS_DIR, "DeepLabV3Plus-Pytorch")

# Download URLs
COCO_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAINF_REPO_URL = "https://github.com/VainF/DeepLabV3Plus-Pytorch.git"

# 6 DeepLab models from VainF repo
MODEL_NAMES = [
    "deeplabv3plus_resnet101",
    "deeplabv3plus_resnet50",
    "deeplabv3plus_mobilenet",
    "deeplabv3_resnet101",
    "deeplabv3_resnet50",
    "deeplabv3_mobilenet",
]

# Dropbox direct-download URLs for VOC pretrained weights
WEIGHT_URLS = {
    "deeplabv3plus_resnet101": "https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=1",
    "deeplabv3plus_resnet50":  "https://www.dropbox.com/s/dgxyd3jkyz24voa/best_deeplabv3plus_resnet50_voc_os16.pth?dl=1",
    "deeplabv3plus_mobilenet": "https://www.dropbox.com/s/0idrhwz6opaj7q4/best_deeplabv3plus_mobilenet_voc_os16.pth?dl=1",
    "deeplabv3_resnet101":     "https://www.dropbox.com/s/vtenndnsrnh4068/best_deeplabv3_resnet101_voc_os16.pth?dl=1",
    "deeplabv3_resnet50":      "https://www.dropbox.com/s/3eag5ojccwiexkq/best_deeplabv3_resnet50_voc_os16.pth?dl=1",
    "deeplabv3_mobilenet":     "https://www.dropbox.com/s/uhksxwfcim3nkpo/best_deeplabv3_mobilenet_voc_os16.pth?dl=1",
}


def weight_path(model_name):
    return os.path.join(WEIGHTS_DIR, f"best_{model_name}_voc_os16.pth")


# COCO category ID -> Pascal VOC class index
COCO_TO_VOC = {
    5: 1,    # airplane -> aeroplane
    2: 2,    # bicycle
    16: 3,   # bird
    9: 4,    # boat
    44: 5,   # bottle
    6: 6,    # bus
    3: 7,    # car
    17: 8,   # cat
    62: 9,   # chair
    21: 10,  # cow
    67: 11,  # dining table
    18: 12,  # dog
    19: 13,  # horse
    4: 14,   # motorcycle -> motorbike
    1: 15,   # person
    64: 16,  # potted plant
    20: 17,  # sheep
    63: 18,  # couch -> sofa
    7: 19,   # train
    72: 20,  # tv -> tvmonitor
}

# Category configurations: animals vs vehicles
# voc_to_custom maps VOC class index -> custom index (1-7). Unlisted VOC classes -> 0 (background).
# voc_ids lists the VOC indices whose softmax channels to extract from the 21-class model output.
CATEGORIES = {
    "animals": {
        "class_names": ["background", "bird", "cat", "cow", "dog", "horse", "person", "sheep"],
        "coco_names": ["cat", "dog", "sheep", "bird", "horse", "cow", "person"],
        "voc_ids": [3, 8, 10, 12, 13, 15, 17],
        "voc_to_custom": {3: 1, 8: 2, 10: 3, 12: 4, 13: 5, 15: 6, 17: 7},
    },
    "vehicles": {
        "class_names": ["background", "airplane", "bicycle", "bus", "car", "motorcycle", "person", "train"],
        "coco_names": ["airplane", "bicycle", "bus", "car", "motorcycle", "person", "train"],
        "voc_ids": [1, 2, 6, 7, 14, 15, 19],
        "voc_to_custom": {1: 1, 2: 2, 6: 3, 7: 4, 14: 5, 15: 6, 19: 7},
    },
}

NUM_CLASSES_CUSTOM = 8   # background + 7 foreground
NUM_CLASSES_VOC = 21
CROP_SIZE = (512, 512)
CENTER_CROP = 256
OUTPUT_STRIDE = 16

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
