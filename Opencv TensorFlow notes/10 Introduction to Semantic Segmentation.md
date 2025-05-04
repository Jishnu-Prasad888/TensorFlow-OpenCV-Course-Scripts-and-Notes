# Introduction to Semantic Segmentation using TensorFlow Hub


- we will learn how to perform semantic image segmentation using pre-trained models available in **TensorFlow Hub**.
- TensorFlow Hub is a library and platform designed for sharing, discovering, and reusing pre-trained machine learning models. 
- The primary goal of TensorFlow Hub is to simplify the process of reusing existing models, thereby promoting collaboration, reducing redundant work, and accelerating research and development in machine learning. 
- Users can search for pre-trained models, called modules, that have been contributed by the community or provided by Google. These modules can be easily integrated into a user's own machine learning projects with just a few lines of code.
- Image segmentation is a fundamental computer vision task that involves partitioning an image into multiple segments or regions, where each segment corresponds to a specific object, area of interest, or background.
	- By simplifying the representation of an image, segmentation techniques make it easier to analyze and process images for various applications, such as object recognition, tracking, and scene understanding. 
	- The goal of image segmentation is to simplify the representation of an image and make it more meaningful for analysis or further processing. 
- In this example, we will use an image segmentation model [camvid-hrnetv2-w48](https://tfhub.dev/google/HRNet/camvid-hrnetv2-w48/1) that was trained on [CamVid (Cambridge-driving Labeled Video Database)](https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), which is a driving and scene understanding dataset containing images extracted from five video sequences taken during real-world driving scenarios. The dataset contains 32 classes. Several other image segmentation models can be found [here](https://tfhub.dev/s?module-type=image-segmentation) as well.

```python
import os
import numpy as np
import cv2
import glob as glob

import tensorflow as tf
import tensorflow_hub as hub

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from zipfile import ZipFile
from urllib.request import urlretrieve

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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

URL = r"https://www.dropbox.com/s/wad5js22fbeo1be/camvid_images.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "camvid_images.zip")

# Download if assest ZIP does not exists.
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
```

##### `warnings.filterwarnings("ignore", module="absl")`

- **Purpose**: Suppresses all warnings emitted by modules whose name matches `"absl"`.
    
- **Effect**: Any warnings that originate from the `absl` module (commonly used by TensorFlow and other Google libraries) will not be shown.
    

---

##### `logging.captureWarnings(True)`

- **Purpose**: Redirects Python `warnings.warn()` calls to the logging system.
    
- **Effect**: Instead of being printed directly to stderr, warnings will now go through the `logging` module. This allows for better control over how warnings are displayed or recorded.
    

---

##### `absl_logger = logging.getLogger("absl")`

##### `absl_logger.setLevel(logging.ERROR)`

- **Purpose**: Gets the logger specifically associated with the `absl` module and sets its logging level to `ERROR`.
    
- **Effect**: Suppresses all logs from `absl` that are below the `ERROR` level (e.g., `WARNING`, `INFO`, or `DEBUG`).
    

---

##### `os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"`

- **Purpose**: Tells TensorFlow to suppress lower-level C++ backend logs.
    
- **Value meanings**:
    
    - `"0"`: All logs shown (default)
        
    - `"1"`: Filter out INFO logs
        
    - `"2"`: Filter out INFO and WARNING logs
        
    - `"3"`: Filter out INFO, WARNING, and ERROR logs
        
- **Effect**: With `"2"`, TensorFlow backend will show only `ERROR` and `FATAL` messages.

---------

### 1.1 Display Sample Images

```python
image_paths = sorted(glob.glob("camvid_images" + os.sep + "*.png"))

for idx in range(len(image_paths)):
    print(image_paths[idx])

def load_image(path):
    image = cv2.imread(path)

    # Convert image in BGR format to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Add a batch dimension which is required by the model.
    image = np.expand_dims(image, axis=0) / 255.0

    return image

images = []
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

for idx, axis in enumerate(ax.flat):
    image = load_image(image_paths[idx])
    images.append(image)
    axis.imshow(image[0])
    axis.axis("off")
```

##### np.expand_dims(image, axis=0)

The function **`np.expand_dims(image, axis=0)`** is used to **add a new dimension** to a NumPy array — in this case, at the **0th axis** (the front).

###### ✅ What It Does

```python
np.expand_dims(image, axis=0)
```

- **Adds a new dimension at axis 0**, effectively turning:
    - A shape like `(height, width, channels)` → into `(1, height, width, channels)`
This is common when you need to **add a batch dimension** before passing the image to a model.

###### 🔍 Why It's Used

Many machine learning models (like TensorFlow or PyTorch models) expect input in the form of a **batch**, even if it's just a single image.

- If `image.shape == (224, 224, 3)` → a single image
- After `np.expand_dims(image, axis=0)` → `shape == (1, 224, 224, 3)` → a batch of one image

###### 📌 Equivalent Alternative

```python
image[None, ...]  # Same as np.expand_dims(image, axis=0)
```

---

### 1.2 Define a Dictionary that Maps Class IDs to Class Names and Class Colors


```python
class_index = \
    {
         0: [(64, 128, 64),  'Animal'],
         1: [(192, 0, 128),  'Archway'],
         2: [(0, 128, 192),  'Bicyclist'],
         3: [(0, 128, 64),   'Bridge'],
         4: [(128, 0, 0),    'Building'],
         5: [(64, 0, 128),   'Car'],
         6: [(64, 0, 192),   'Cart/Luggage/Pram'],
         7: [(192, 128, 64), 'Child'],
         8: [(192, 192, 128),'Column Pole'],
         9: [(64, 64, 128),  'Fence'],
        10: [(128, 0, 192),  'LaneMkgs Driv'],
        11: [(192, 0, 64),   'LaneMkgs NonDriv'],
        12: [(128, 128, 64), 'Misc Text'],
        13: [(192, 0, 192),  'Motorcycle/Scooter'],
        14: [(128, 64, 64),  'Other Moving'],
        15: [(64, 192, 128), 'Parking Block'],
        16: [(64, 64, 0),    'Pedestrian'],
        17: [(128, 64, 128), 'Road'],
        18: [(128, 128, 192),'Road Shoulder'],
        19: [(0, 0, 192),    'Sidewalk'],
        20: [(192, 128, 128),'Sign Symbol'],
        21: [(128, 128, 128),'Sky'],
        22: [(64, 128, 192), 'SUV/Pickup/Truck'],
        23: [(0, 0, 64),     'Traffic Cone'],
        24: [(0, 64, 64),    'Traffic Light'],
        25: [(192, 64, 128), 'Train'],
        26: [(128, 128, 0),  'Tree'],
        27: [(192, 128, 192),'Truck/Bus'],
        28: [(64, 0, 64),    'Tunnel'],
        29: [(192, 192, 0),  'Vegetation Misc'],
        30: [(0, 0, 0),      'Void'],
        31: [(64, 192, 0),   'Wall']  
    }
```


## 2 Model Inference using TensorFlow Hub

- TensorFlow Hub contains many different pre-trained [segmentation models](https://tfhub.dev/s?module-type=image-segmentation). Here we will use the High-Resolution Network (HRNet) segmentation model trained on CamVid (`camvid-hrnetv2-w48`).
- The model has been pre-trained on the Imagenet ILSVRC-2012 classification task and fine-tuned on CamVid.

### 2.1 Load the Model from TensorFlow Hub

```python
model_url = "https://tfhub.dev/google/HRNet/camvid-hrnetv2-w48/1"
print("loading model:", model_url)

seg_model = hub.load(model_url)
print("\nmodel loaded!")
```

### 2.2 Perform Inference

Before we formalize the code to process several images and post-process the results, let's first see how to perform inference on a single image and study the output from the model.

### 2.2.1 Call the Model's `precict()` Method

```python
# Make a prediction using the first image in the list of images.
pred_mask = seg_model.predict(images[0])

# The predicted mask has the following shape: [B, H, W, C].
print("Shape of predicted mask:", pred_mask.shape)

#output
# Shape of predicted mask: (1, 720, 960, 33)
```

### 2.2.2 Post-Process the Predicted Segmentation Mask

The predicted segmentation mask returned by the model contains a separate channel for each class. Each channel contains the probability that a given pixel from the input image is associated with the class for that channel. This data, therefore, requires some post-processing to obtain meaningful results. Several steps need to be performed to arrive at a final visual representation.

1. Remove the batch dimension and the background class.
2. Assign a class label to every pixel in the image based on the highest probability score across all channels.
3. The previous step results in a single-channel image that contains the class labels for each pixel. We, therefore, need to map those class IDs to RGB values so we can visualize the results as a color-coded segmentation map.

#### Remove Batch Dimension and Background Class

```python
# Convert tensor to numpy array.
pred_mask = pred_mask.numpy()

# The 1st label is the background class added by the model, but we can remove it for this dataset.
# removing the background class (class 0) and keeping only classes 1, 2, and 3 (the meaningful foreground classes).
pred_mask = pred_mask[:, :, :, 1:]

# We also need to remove the batch dimension.
pred_mask = np.squeeze(pred_mask)

# Print the shape to confirm: [H, W, C].
print("Shape of predicted mask after removal of batch dimension and background class:", pred_mask.shape)
```

##### Background Class

In semantic segmentation tasks, the **background class** typically refers to **all regions in the image that do not belong to any of the target classes**. It's essentially a "catch-all" class for everything that isn't explicitly labeled as something of interest (like a person, car, tree, etc.).

###### 🔹 Key Characteristics of the Background Class

- **Class ID**: It is usually assigned class ID `0`.
- **Purpose**: To help the model distinguish between foreground objects (your actual classes) and irrelevant parts of the image.
- **Channel**: In a multi-channel output from the model (like `[batch, height, width, num_classes]`), the first channel (`channel 0`) often corresponds to the background.
###  Example Scenario

Imagine you're segmenting road scenes with 3 classes:

- `0`: Background
- `1`: Road
- `2`: Car
- `3`: Pedestrian

If your model outputs shape `(1, 256, 256, 4)`:

- That means you have 1 image, 256×256 pixels, and 4 channels (1 per class).
- You typically **ignore the background channel (`0`) during post-processing**, focusing only on meaningful foreground classes.

###### ✅ Why Remove the Background Class?

During post-processing and visualization:

- You usually **discard the background channel** to reduce confusion and highlight only the important classes.
- Or you include it but **render it with a neutral color** (e.g., black or gray).

------

#### Assign Each Pixel a Class Label

- Here we assign every pixel in the image with a class ID based on the class with the highest probability. We can visualize this as a grayscale image. In the code cell below, we will display just the top portion of the image to highlight a few of the class assignments.

```python
# Assign each pixel in the image a class ID based on the channel that contains the
# highest probability score. This can be implemented using the `argmax` function.
pred_mask_class = np.argmax(pred_mask, axis=-1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.title("Input Image", fontsize=12)
plt.imshow(np.squeeze(images[0]))

plt.subplot(1, 2, 2)
plt.title("Segmentation Mask", fontsize=12)
plt.imshow(pred_mask_class, cmap="gray")
plt.gca().add_patch(Rectangle((450, 200), 200, 3, edgecolor="red", facecolor="none", lw=0.5))
```

###### ✅ `axis=-1` in `np.argmax(pred_mask, axis=-1)` means:
 Compute the `argmax` **along the last axis** of the array — regardless of how many dimensions the array has.

---
#### Convert the Single Channel Mask to a Color Representation

We will also need to make use of the function below that will convert a single channel mask to an RGB representation for visualization purposes. 

Each class ID in the single-channel mask will be converted to a different color according to the `class_index` dictionary mapping.

```python
# Function to convert a single channel mask representation to an RGB mask.
def class_to_rgb(mask_class, class_index):
    
    # Create RGB channels.
    r_map = np.zeros_like(mask_class).astype(np.uint8)
    g_map = np.zeros_like(mask_class).astype(np.uint8)
    b_map = np.zeros_like(mask_class).astype(np.uint8)

    # Populate RGB color channels based on the color assigned to each class.
    for class_id in range(len(class_index)):
        index = mask_class == class_id
        r_map[index] = class_index[class_id][0][0]
        g_map[index] = class_index[class_id][0][1]
        b_map[index] = class_index[class_id][0][2]

    seg_map_rgb = np.stack([r_map, g_map, b_map], axis=2)

    return seg_map_rgb
```


`mask_class == class_id` creates a mask identifying all pixels belonging to a given class.

`seg_map_rgb = np.stack([r_map,g_map , b_map] , axis= 2 )`
- `np.stack([...], axis=2)` builds a color image from separate channels.
- `r_map`, `g_map`, `b_map` are each 2D arrays (e.g., shape `[H, W]`) representing the red, green, and blue values of an image.
- `np.stack(..., axis=2)` stacks them **along a new axis at position 2**, forming a 3D array with shape `[H, W, 3]`.
- This effectively creates an image where each pixel has 3 values: `[R, G, B]`.

![](Pasted%20image%2020250503233513.png)

---
## 3 Formalize the Implementation

```python
# Function to overlay a segmentation map on top of an RGB image.
def image_overlay(image, seg_map_rgb):
    
    alpha = 1.0  # Transparency for the original image.
    beta  = 0.6  # Transparency for the segmentation map.
    gamma = 0.0  # Scalar added to each sum.

    image = (image * 255.0).astype(np.uint8)
    seg_map_rgb = cv2.cvtColor(seg_map_rgb, cv2.COLOR_RGB2BGR)

    image = cv2.addWeighted(image, alpha, seg_map_rgb, beta, gamma)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def run_inference(images, model):
    for img in images:
        # Forward pass through the model (convert the tensor output to a numpy array).
        pred_mask = model.predict(img).numpy()

        # Remove the background class added by the model.
        pred_mask = pred_mask[:, :, :, 1:]

        # Remove the batch dimension.
        pred_mask = np.squeeze(pred_mask)

        # `pred_mask` is a numpy array of shape [H, W, 32] where each channel contains the probability
        # scores associated with a given class. We still need to assign a single class to each pixel
        # which is accomplished using the argmax function across the last dimension to obtain the class labels.
        pred_mask_class = np.argmax(pred_mask, axis=-1)

        # Convert the predicted (class) segmentation map to a color segmentation map.
        pred_mask_rgb = class_to_rgb(pred_mask_class, class_index)

        fig = plt.figure(figsize=(20, 15))

        # Display the original image.
        ax1 = fig.add_subplot(1, 3, 1)
        ax1.imshow(img[0])
        ax1.title.set_text("Input Image")
        plt.axis("off")

        # Display the predicted color segmentation mask.
        ax2 = fig.add_subplot(1, 3, 2)
        ax2.set_title("Predicted Mask")
        ax2.imshow(pred_mask_rgb)
        plt.axis("off")

        # Display the predicted color segmentation mask overlayed on the original image.
        overlayed_image = image_overlay(img[0], pred_mask_rgb)
        ax4 = fig.add_subplot(1, 3, 3)
        ax4.set_title("Overlayed Image")
        ax4.imshow(overlayed_image)
        plt.axis("off")

        plt.show()

def plot_color_legend(class_index):
    # Extract colors and labels from class_index dictionary.
    color_array = np.array([[v[0][0], v[0][1], v[0][2]] for v in class_index.values()]).astype(np.uint8)
    class_labels = [val[1] for val in class_index.values()]

    fig, ax = plt.subplots(nrows=2, ncols=16, figsize=(20, 3))
    plt.subplots_adjust(wspace=0.5, hspace=0.01)

    # Display color legend.
    for i, axis in enumerate(ax.flat):
        axis.imshow(color_array[i][None, None, :])
        axis.set_title(class_labels[i], fontsize=8)
        axis.axis("off")


run_inference(images, seg_model)
```

To perform inference on several images, we define the function below, which accepts a list of images and a pre-trained model. This function also handles all of the post-processing required to compute the final segmentation mask as well as the overlay.

`image_overlay()` is a helper function to overlay an RGB mask on top of the original image to better appreciate how the predictions line up with the original image.

![](Pasted%20image%2020250503234133.png)

