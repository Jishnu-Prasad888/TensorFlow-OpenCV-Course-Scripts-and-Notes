
# Introduction to Object Detection using TensorFlow Hub


- we will use the model [EfficientDet/d4](https://tfhub.dev/tensorflow/efficientdet/d4/1), which is from a family of models known as `EfficientDet`. 
- The pre-trained models from this family available on TensorFlow Hub were all trained on the [COCO 2017 dataset](https://cocodataset.org/#home). The different models in the family, ranging from D0 to D7, vary in terms of complexity and input image dimensions. D0, the most compact model, accepts input sizes of 512x512 pixels and provides the quickest inference speed. 
- At the other end of the spectrum, we have D7, which requires an input size of 1536x1536 and takes considerably longer to perform inference. Several other object detection models can be found [here](https://tfhub.dev/s?module-type=image-object-detection) as well.


```python
import os
import numpy as np
import cv2
import zipfile
import requests
import glob as glob

import tensorflow_hub as hub
import matplotlib
import matplotlib.pyplot as plt
import warnings
import logging
import absl

# Filter absl warnings
warnings.filterwarnings("ignore", module="absl")

# Capture all warnings in the logging system
logging.captureWarnings(True)

# Set the absl logger level to 'error' to suppress warnings
absl_logger = logging.getLogger("absl")
absl_logger.setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assests....", end="")

    # Downloading zip file using urllib package.
    urlretrieve(url, save_path)

    try:
        # Extracting zip file using the zipfile package.
        with ZipFile(save_path) as z:
            # Extract ZIP file contents in the same directory.
            z.extractall(os.path.split(save_path)[0])

        print("Done")

    except Exception as e:
        print("\nInvalid file.", e)

URL = r"https://www.dropbox.com/s/h7l1lmhvga6miyo/object_detection_images.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "object_detection_images.zip")
# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

def load_image(path):
    image = cv2.imread(path)
    # Convert image in BGR format to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Add a batch dimension which is required by the model.
    image = np.expand_dims(image, axis=0)    
    return image

images= []

fig , ax = plt.subplots(nrows = 2 , ncols = 2 , figsize = (20,15))

idx = 0
for axis in ax.flat:
	image = load_image(image_paths[idx])
	images.append(image)
	idx += 1

class_index =  \
{
         1: 'person',
         2: 'bicycle',
         3: 'car',
         4: 'motorcycle',
         5: 'airplane',
         6: 'bus',
         7: 'train',
         8: 'truck',
         9: 'boat',
         10: 'traffic light',
         11: 'fire hydrant',
         13: 'stop sign',
         14: 'parking meter',
         15: 'bench',
         16: 'bird',
         17: 'cat',
         18: 'dog',
         19: 'horse',
         20: 'sheep',
         21: 'cow',
         22: 'elephant',
         23: 'bear',
         24: 'zebra',
         25: 'giraffe',
         27: 'backpack',
         28: 'umbrella',
         31: 'handbag',
         32: 'tie',
         33: 'suitcase',
         34: 'frisbee',
         35: 'skis',
         36: 'snowboard',
         37: 'sports ball',
         38: 'kite',
         39: 'baseball bat',
         40: 'baseball glove',
         41: 'skateboard',
         42: 'surfboard',
         43: 'tennis racket',
         44: 'bottle',
         46: 'wine glass',
         47: 'cup',
         48: 'fork',
         49: 'knife',
         50: 'spoon',
         51: 'bowl',
         52: 'banana',
         53: 'apple',
         54: 'sandwich',
         55: 'orange',
         56: 'broccoli',
         57: 'carrot',
         58: 'hot dog',
         59: 'pizza',
         60: 'donut',
         61: 'cake',
         62: 'chair',
         63: 'couch',
         64: 'potted plant',
         65: 'bed',
         67: 'dining table',
         70: 'toilet',
         72: 'tv',
         73: 'laptop',
         74: 'mouse',
         75: 'remote',
         76: 'keyboard',
         77: 'cell phone',
         78: 'microwave',
         79: 'oven',
         80: 'toaster',
         81: 'sink',
         82: 'refrigerator',
         84: 'book',
         85: 'clock',
         86: 'vase',
         87: 'scissors',
         88: 'teddy bear',
         89: 'hair drier',
         90: 'toothbrush'
}

R = np.array(np.arange(96, 256, 32))
G = np.roll(R, 1)
B = np.roll(R, 2)

COLOR_IDS = np.array(np.meshgrid(R, G, B)).T.reshape(-1, 3)
```


###### 1. `R = np.array(np.arange(96, 256, 32))`
- This creates an array of red component values starting from **96** up to **256** (exclusive) in steps of **32**:
    ```python
    R = [96, 128, 160, 192, 224]
    ```
###### 2. `G = np.roll(R, 1)`
- This "rolls" or **circularly shifts** the `R` array by **1 position to the right**, to get green component values:
    ```python
    G = [224, 96, 128, 160, 192]
    ```
###### 3. `B = np.roll(R, 2)`
- This shifts the same `R` array by **2 positions**, giving blue component values:
    ```python
    B = [160, 192, 224, 96, 128]
    ```
###### 4. `COLOR_IDS = np.array(np.meshgrid(R, G, B)).T.reshape(-1, 3)`

- `np.meshgrid(R, G, B)` creates a 3D grid of all combinations of R, G, and B values.
- `.T` transposes the resulting array to align dimensions properly.
- `.reshape(-1, 3)` flattens the grid into a **2D array** where each row is a unique RGB color triplet.
###### Result
`COLOR_IDS` ends up being an array of shape `(125, 3)` (since 5 R values × 5 G values × 5 B values = 125 combinations), each row representing an RGB color with the interrelated shifts applied to G and B.

```python
EfficientDet  = {'EfficientDet D0 512x512'   : 'https://tfhub.dev/tensorflow/efficientdet/d0/1',
                 'EfficientDet D1 640x640'   : 'https://tfhub.dev/tensorflow/efficientdet/d1/1',
                 'EfficientDet D2 768x768'   : 'https://tfhub.dev/tensorflow/efficientdet/d2/1',
                 'EfficientDet D3 896x896'   : 'https://tfhub.dev/tensorflow/efficientdet/d3/1',
                 'EfficientDet D4 1024x1024' : 'https://tfhub.dev/tensorflow/efficientdet/d4/1',
                 'EfficientDet D5 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d5/1',
                 'EfficientDet D6 1280x1280' : 'https://tfhub.dev/tensorflow/efficientdet/d6/1',
                 'EfficientDet D7 1536x1536' : 'https://tfhub.dev/tensorflow/efficientdet/d7/1'
                }

model_url = EfficientDet['EfficientDet D4 1024x1024' ]

print('loading model: ', model_url)
od_model = hub.load(model_url)

print('\nmodel loaded!')


# Call the model. # The model returns the detection results in the form of a dictionary.
results = od_model(images[0])

results = {key:value.numpy() for key, value in results.items()}
results = {key:value.numpy() for key, value in results.items()}

print('Num Raw Detections: ', (len(results['raw_detection_scores'][0])))
print('Num Detections:     ', (results['num_detections'][0]).astype(int))

# Print the Scores, Classes and Bounding Boxes for the detections.
num_dets = (results['num_detections'][0]).astype(int)

print('\nDetection Scores: \n\n', results['detection_scores'][0][0:num_dets])
print('\nDetection Classes: \n\n', results['detection_classes'][0][0:num_dets])
print('\nDetection Boxes: \n\n', results['detection_boxes'][0][0:num_dets])
```


### Post-Process and Display Detections


Here we show the logic for how to interpret the detection data for a single image. As we showed above, the model returned 16 detections, however, many detections have low confidence scores and we, therefore, need to filter these further by using a minimum detection threshold.

1. Retrieve the detections from the results dictionary
2. Apply a minimum detection threshold to filter the detections
3. For each thresholded detection, display the bounding box and a label indicating the detected class and the confidence of the detection.

