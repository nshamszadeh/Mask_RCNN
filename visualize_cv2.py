import cv2
import numpy as np
import os
import sys
from mrcnn import utils
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
import coco

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
config.display()

model = modellib.MaskRCNN(
    mode="inference", model_dir=MODEL_DIR, config=config
)
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = [
    'BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
    'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

#class_names = ['person', 'car', 'bus', 'truck', 'traffic light', 'stop sign', 'backpack', 'umbrella', 'handbag', 'suitcase']


def get_mask(image, mask):
    # colors come in rbg values, to make things black or white we apply 0 or 1 to each rgb value in a loop
    for n in range(3):
        masked_part = 255.0 * np.ones(image[:, :, n].shape)
        black_part =  image[:, :, n] * 0.0
        image[:, :, n] = np.where(
            mask == 1,
            masked_part,
            black_part
        )
    return image

def display_mask(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    if n_instances == 0:
        print('NO OBJECTS DETECTED')
        image *= 0
    elif not n_instances: 
        print('NO INSTANCES TO DISPLAY (MASK)')
        image *= 0
    else:
        print(n_instances, " objects detected.")
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
        mask = masks[:, :, 0]
        # loop through each detected object and add (logical or) their individual masks to the main image mask
        for i in range(1, n_instances):
            mask = np.logical_or(mask, masks[:, :, i])
            print(names[ids[i]]) # prints each detected object to the console
        image = get_mask(image, mask)
    return image