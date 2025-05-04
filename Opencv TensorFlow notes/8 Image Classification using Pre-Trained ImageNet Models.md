
# Pre-Trained ImageNet Models in TensorFlow and Keras


## 1 Pre-Trained Models in Keras

To use any of the pre-trained models in Keras, there are four basic steps required:

1. Load a pre-trained model
2. Preprocess the input image(s) using a dedicated pre-processing function that is accessible in the model, `preprocess_input()`
3. Call the model's `predict()` method to generate predictions
4. De-code the predictions using a dedicated post-processing function that is accessible in the model, `decode_predictions()`

### 1.1 Instantiate the Model

- Here we will use the [ResNet50](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50) model to describe the approach. Here we call the built-in model `ResNet50()` to instantiate the ResNet50 pre-trained model. 
- Notice that the function has several optional arguments, which provide a lot of flexibility for using the model. However, the default settings allow you to use the model right out of the box to perform image classification from 1,000 classes in the ImageNet dataset.

```python
model_resnet50 = tf.keras.applications.resnet50.ResNet50(include_top=True,                                                       weights='imagenet', 
										input_tensor=None,                                                      input_shape=None, 
                                        pooling=None, 
                                        classes=1000,                                                           classifier_activation='softmax'               )
```


### 1.2 Preprocess the Inputs

- When these models were trained on the ImageNet dataset, the input images were preprocessed in a specific way. Besides resizing images to conform to the expected size of the network, the images are typically zero-centered and normalized. When using these models, it's important that your input images are pre-processed in the same way the training images were processed. For convenience, each model in Keras includes a dedicated pre-processing function [preprocess_input](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/preprocess_input). Here `x` represents a floating point `numpy.array` or a `tf.Tensor` containing the image data.

```python
tf.keras.applications.resnet50.preprocess_input(x, data_format=None)
```

### 1.3 Call the Model's `predict()` Method

- After pre-processing the input images, we can then pass them to the model's `predict()` method as shown below. 
- Because TensorFlow and Keras process image data in batches, we will need to add a batch dimension to the images, even if we process one image at a time. 
- As an example, ResNet50 expects color images with a shape of `[224,224,3]`, but we must add a batch dimension so that the image batch has a shape: `[B, H, W, C]`, even if we intend to process a single image at a time. We'll see how this is done further below.

```python
preds = model_resnet50.predict(image_batch)
```

- The predictions returned by the `predict()` method will contain the class probabilities for all 1,000 classes in a NumPy array.

### 1.4 Decode the Predictions

- Fortunately, there is a convenience function available for decoding the predictions returned by the model. You can use the model-specific function [resnet50.decode_predictions](https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/decode_predictions) or the [imagenet_utils](https://www.tensorflow.org/api_docs/python/tf/keras/applications/imagenet_utils/preprocess_input) version (both shown below), which will consolidate the top `k` predictions in descending order.

```python
decoded_preds = tf.keras.applications.resnet50.decode_predictions(
    preds, 
    top=5
)

decoded_preds = tf.keras.applications.imagenet_utils.decode_predictions(
    preds, 
    top=5
)
```

- These functions return a list of the top `k` (default = 5) predictions, along with the class IDs and class descriptions (names). This makes it easy to parse and report results.

## 2 Read and Display Sample Images

### 2.1 Download Assets and read them

The `download_and_unzip(...)` is used to download and extract the notebook assests.

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import glob as glob
import os

from zipfile import ZipFile
from urllib.request import urlretrieve

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
        
URL = r"https://www.dropbox.com/s/8srx6xdjt9me3do/TF-Keras-Bootcamp-NB07-assets.zip?dl=1"

asset_zip_path = os.path.join(os.getcwd(), "TF-Keras-Bootcamp-NB07-assets.zip")

# Download if assest ZIP does not exists. 
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

# Store all the image paths in a list.
image_paths = sorted(glob.glob("images" + os.sep + "*.png"))
print(image_paths)
```

## 3 Pre-Trained Model Setup

### 3.1 Load the Models

```python
model_vgg16        = tf.keras.applications.vgg16.VGG16()
model_resnet50     = tf.keras.applications.resnet50.ResNet50()
model_inception_v3 = tf.keras.applications.inception_v3.InceptionV3()
```

### 3.2 Create a Convenience Function for Batch Processing

For convenience, we will create a function that will automate the processing steps required for processing each image.

1. Read the images
2. Perform the required pre-processing for the images as required by the model
3. Add a batch dimension to the image tensor
4. Call the model's predict() method to make predictions
5. Decode the predictions to find the class name and confidence score for the top-k predictions
6. Display the results

```python
def process_images(model, image_paths, size, preprocess_input, display_top_k=False, top_k=2):
    
    plt.figure(figsize=(20,7))
    for idx, image_path in enumerate(image_paths):
    
        # Read the image using TensorFlow.
        tf_image = tf.io.read_file(image_path)

        # Decode the above `tf_image` from a Bytes string to a numeric Tensor.
        decoded_image = tf.image.decode_image(tf_image)

        # Resize the image to the spatial size required by the model.
        image_resized = tf.image.resize(decoded_image, size)

        # Add a batch dimension to the first axis (required). 
        image_batch = tf.expand_dims(image_resized, axis=0)

        # Pre-process the input image.
        image_batch = preprocess_input(image_batch)

        # Forward pass through the model to make predictions.
        preds = model.predict(image_batch)

        # Decode (and rank the top-k) predictions. 
        # Returns a list of tuples: (class ID, class description, probability)
        decoded_preds = tf.keras.applications.imagenet_utils.decode_predictions(
            preds=preds,
            top=5
        )
        
        if display_top_k == True:
            for jdx in range(top_k):
                print("Top {} predicted class:   Pr(Class={:20} [index={:4}]) = {:5.2f}".format(
                    jdx + 1, decoded_preds[0][jdx][1], jdx, decoded_preds[0][jdx][2] * 100))
    
        plt.subplot(2,4,idx+1)
        plt.imshow(decoded_image)
        plt.axis('off')
        label = decoded_preds[0][0][1]
        score = decoded_preds[0][0][2] * 100
        title = label + ' ' + str('{:.2f}%'.format(score))
        plt.title(title, fontsize=16)
```


## 4 Make Predictions using the Pre-Trained Models

### 4.1 VGG-16

```python
model = model_vgg16
size = (224, 224) 

preprocess_input = tf.keras.applications.vgg16.preprocess_input

process_images(model, image_paths, size, preprocess_input)
```

### 4.2 Resnet-50

```python
model = model_resnet50
size = (224, 224)

preprocess_input = tf.keras.applications.resnet50.preprocess_input

process_images(model, image_paths, size, preprocess_input)
```

### 4.3 Inception-V3

```python
model = model_inception_v3
size = (299, 299)

preprocess_input = tf.keras.applications.inception_v3.preprocess_input

process_images(model, image_paths, size, preprocess_input, display_top_k=True)
```


--------


**Question**

*model = tf.keras.Sequential()*  
*model.add(Conv2D(64, kernel_size=5, padding='valid', activation='relu'))*  
*model.add(Conv2D(32, kernel_size=3, padding='same', activation='relu'))*  
*model.add(MaxPooling2D(pool_size=(2, 2)))*

*What is the shape of the output tensor when an tensor of shape of (2, 15, 15, 3) is passed as input to the model above?*

**Answer :**

Let's walk through the model layer by layer to compute the output shape when an input tensor of shape **(2, 15, 15, 3)** is passed in.

---

###### 🔹 Input:
- Shape: **(2, 15, 15, 3)**

    - Batch size: 2
    - Height: 15
    - Width: 15
    - Channels: 3
###### 🔹 First Layer: `Conv2D(64, kernel_size=5, padding='valid', activation='relu')`

- **Padding = 'valid'** → No padding, so the output size shrinks.
- **Kernel size = 5x5**
- **Strides = 1** (default)

**Formula for output size (valid padding):**

$$\text{Output size} = \left\lfloor \frac{\text{Input size} - \text{Kernel size}}{\text{Stride}} \right\rfloor + 1

$$
- New shape: **(2, 11, 11, 64)**
    

###### 🔹 Second Layer: `Conv2D(32, kernel_size=3, padding='same', activation='relu')`

- **Padding = 'same'** → Output size stays the same
- **Kernel size = 3x3**
Output shape: **(2, 11, 11, 32)**

###### 🔹 Third Layer: `MaxPooling2D(pool_size=(2, 2))`

- **Pool size = 2x2**
- **Stride = 2** (default for 2x2)

Output size becomes half of height and width:

$$112=⌊5.5⌋=5\frac{11}{2} = \left\lfloor 5.5 \right\rfloor = 5211​=⌊5.5⌋=5$$

Output shape: **(2, 5, 5, 32)**


