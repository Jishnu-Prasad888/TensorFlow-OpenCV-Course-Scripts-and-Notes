import os
import numpy as np
import cv2
import zipfile
import requests
import glob as glob
from urllib.request import urlretrieve
import tensorflow_hub as hub
import matplotlib
import matplotlib.pyplot as plt
import warnings
import logging
import absl

# Filter absl warnings
warnings.filterwarnings("ignore", module="absl")
logging.captureWarnings(True)
absl_logger = logging.getLogger("absl")
absl_logger.setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets....", end="")

    urlretrieve(url, save_path)

    try:
        with zipfile.ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\nInvalid file.", e)


URL = r"https://www.dropbox.com/s/h7l1lmhvga6miyo/object_detection_images.zip?dl=1"
asset_zip_path = os.path.join(os.getcwd(), "object_detection_images.zip")

if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

image_paths = sorted(glob.glob('object_detection_images/*.png'))
for idx in range(len(image_paths)):
    print(image_paths[idx])


def load_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.expand_dims(image, axis=0)
    return image


images = []
num_images_to_load = min(4, len(image_paths))
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 15))
ax = ax.flat

for idx in range(num_images_to_load):
    image = load_image(image_paths[idx])
    images.append(image)
    ax[idx].imshow(np.squeeze(image))
    ax[idx].axis('off')

for j in range(num_images_to_load, 4):
    ax[j].axis('off')

class_index = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus',
    7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant',
    13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat',
    18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear',
    24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag',
    32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard',
    37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove',
    41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle',
    46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
    51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
    56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
    61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed',
    67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse',
    75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book',
    85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier',
    90: 'toothbrush'
}

R = np.array(np.arange(96, 256, 32))
G = np.roll(R, 1)
B = np.roll(R, 2)
COLOR_IDS = np.array(np.meshgrid(R, G, B)).T.reshape(-1, 3)

EfficientDet = {
    'EfficientDet D0 512x512': 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
    'EfficientDet D1 640x640': 'https://tfhub.dev/tensorflow/efficientdet/d1/1',
    'EfficientDet D2 768x768': 'https://tfhub.dev/tensorflow/efficientdet/d2/1',
    'EfficientDet D3 896x896': 'https://tfhub.dev/tensorflow/efficientdet/d3/1',
    'EfficientDet D4 1024x1024': 'https://tfhub.dev/tensorflow/efficientdet/d4/1',
    'EfficientDet D5 1280x1280': 'https://tfhub.dev/tensorflow/efficientdet/d5/1',
    'EfficientDet D6 1280x1280': 'https://tfhub.dev/tensorflow/efficientdet/d6/1',
    'EfficientDet D7 1536x1536': 'https://tfhub.dev/tensorflow/efficientdet/d7/1'
}

model_url = EfficientDet['EfficientDet D4 1024x1024']
print("Loading model:", model_url)
od_model = hub.load(model_url)
print("\n\nModel Loaded!")

results = od_model(images[0])
results = {key: value.numpy() for key, value in results.items()}

for key in results:
    print(key)

print('Num Raw Detections:', len(results['raw_detection_scores'][0]))
print('Num Detections:', int(results['num_detections'][0]))

num_dets = int(results['num_detections'][0])
print('\nDetection Scores:\n', results['detection_scores'][0][:num_dets])
print('\nDetection Classes:\n', results['detection_classes'][0][:num_dets])
print('\nDetection Boxes:\n', results['detection_boxes'][0][:num_dets])


def process_detection(image, results, min_det_thresh=0.3):
    scores = results['detection_scores'][0]
    boxes = results['detection_boxes'][0]
    classes = results['detection_classes'][0].astype(int)

    det_indices = np.where(scores >= min_det_thresh)[0]
    scores_thresh = scores[det_indices]
    boxes_thresh = boxes[det_indices]
    classes_thresh = classes[det_indices]

    img_bbox = image.copy()
    im_height, im_width = image.shape[:2]
    font_scale = 0.6
    box_thickness = 2

    for box, class_id, score in zip(boxes_thresh, classes_thresh, scores_thresh):
        ymin, xmin, ymax, xmax = box
        top = int(ymin * im_height)
        left = int(xmin * im_width)
        bottom = int(ymax * im_height)
        right = int(xmax * im_width)

        class_name = class_index.get(class_id, "N/A")
        color = tuple(COLOR_IDS[class_id % len(COLOR_IDS)].tolist())

        img_bbox = cv2.rectangle(img_bbox, (left, top), (right, bottom), color, thickness=box_thickness)

        display_txt = f'{class_name}: {100 * score:.2f}%'
        ((text_width, text_height), _) = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        shift_down = int(2 * (1.3 * text_height)) if top < text_height else 0

        cv2.rectangle(
            img_bbox,
            (left - 1, top - box_thickness - int(1.3 * text_height) + shift_down),
            (left - 1 + int(1.1 * text_width), top),
            color,
            thickness=-1
        )
        cv2.putText(
            img_bbox,
            display_txt,
            (left + int(0.05 * text_width), top - int(0.2 * text_height) + int(shift_down / 2)),
            cv2.FONT_HERSHEY_COMPLEX,
            font_scale,
            (0, 0, 0),
            1
        )

    return img_bbox


def run_inference(images, model):
    results_list = []
    for img in images:
        result = model(img)
        result = {key: value.numpy() for key, value in result.items()}
        results_list.append(result)
    return results_list


results_list = run_inference(images, od_model)

for idx in range(len(results_list)):
    image = np.squeeze(images[idx])
    image_bbox = process_detection(image, results_list[idx], min_det_thresh=0.31)
    plt.figure(figsize=[20, 10 * len(images)])
    plt.subplot(len(images), 1, idx + 1)
    plt.imshow(image_bbox)
    plt.axis('off')
    
plt.show()
