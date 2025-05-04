# Unlock the Power of Fine-Tuning Pre-Trained Models in Tensorflow & Keras

- We will demonstrate Fine-Tuning in the context of image classification using the VGG-16 network, but the concepts we cover here are not specific to any particular model or task. Fine-Tuning is especially useful when you have a limited dataset and/or limited computational resources. However, before we get into the details of Fine-Tuning, we will first summarize several options for leveraging pre-trained models, which include:

		1. Using Pre-Trained models out of the box
		2. Training models from scratch
		3. Transfer Learning
		4. Fine-Tuning

- In the implementation below, we will use a modified version of the German Traffic Sign Recognition Benchmark (GTSRB), a well-known classification dataset that contains German traffic signs from 43 classes. The entire dataset (Train and Test) contains over 50,000 images. However, to demonstrate the power of Fine-Tuning on small datasets, we have carved out a small percentage of the original dataset to include just 40 samples per class (28 for training and 12 for validation). To evaluate how well the model generalizes, we will use the original test dataset, which contains 12,630 samples.


## 1 Overview of Pre-Trained Model Use Cases

### 1.1 Pre-Trained ImageNet Models

- For situations where your application contains specific classes that are not contained in ImageNet, you have three additional options. The Keras applications API conveniently provides access to many CNN classification models, which you can load into memory, customize, and train.

![](Pasted%20image%2020250503122510.png)

### Transfer Learning

 - Transfer Learning is a simple approach for re-purposing a pre-trained model to make predictions on a new dataset. 
 - The concept is simple. We use the model's pre-trained feature extractor (convolutional base) and re-train a new classifier to learn new weights for the new dataset. This is sometimes referred to as "freezing" the layers in the feature extractor, meaning that we load the pre-trained weights and do not attempt to modify them further during the training process. The theory is that the pre-trained ImageNet Feature Extractor has learned valuable features for detecting many different object types. We assume such features are general enough that we only need to re-train the classifier portion of the network.
 
![](Pasted%20image%2020250503122851.png)

- sometimes retraining the classifier isn't enough. This is where Fine-Tuning can be very beneficial.

### Fine Tuning

- Fine-Tuning represents a flexible alternative to Transfer Learning. It is very similar to Transfer Learning.
- Instead of locking down the feature extractor completely, we load the feature extractor with ImageNet weights and then freeze the first several layers of the feature extractor but allow the last few layers to be trained further. 
- The idea is that the first several layers in the feature extractor represent generic, low-level features (e.g., edges, corners, and arcs) that are fundamental building blocks necessary to support many classification tasks. Subsequent layers in the feature extractor build upon the lower-level features to learn more complex representations that are more closely related to the content of a particular dataset.

```python
import matplotlib.pyplot as plt

import zipfile
import requests
import glob as glob

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.utils import image_dataset_from_directory

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from dataclasses import dataclass

from zipfile import ZipFile
from urllib.request import urlretrieve
```

## 2 Download and Extract the Dataset

```python
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

URL = r"https://www.dropbox.com/s/uzgh5g2bnz40o13/dataset_traffic_signs_40_samples_per_class.zip?dl=1"

dataset_path   = os.path.join(os.getcwd(), "dataset_traffic_signs_40_samples_per_class")
asset_zip_path = os.path.join(os.getcwd(), "dataset_traffic_signs_40_samples_per_class.zip")

# Download if assest ZIP does not exists. 
if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)
```

## 3 Dataset and Training Configuration

```python
@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 43
    IMG_HEIGHT:  int = 224
    IMG_WIDTH:   int = 224
    CHANNELS:    int = 3
        
    DATA_ROOT_TRAIN:  str = os.path.join(dataset_path, "Train")  
    DATA_ROOT_VALID:  str = os.path.join(dataset_path, "Valid")
    DATA_ROOT_TEST:   str = os.path.join(dataset_path, "Test")
    DATA_TEST_GT:     str = os.path.join(dataset_path, "Test.csv")    
        

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE:       int   = 32
    EPOCHS:           int   = 101
    LEARNING_RATE:    float = 0.0001
    DROPOUT:          float = 0.6
    LAYERS_FINE_TUNE: int   = 8        
```

## 4 Create Train and Validation Datasets

Here we use `image_dataset_from_directory()`, a very convenient utility in Keras for creating an image dataset. The expected file structure for a dataset is shown below, where the images for each class are contained in a separate class sub-folder.

main_directory/ 
    class_a/
        a_image_1.png
        a_image_2.png
    class_b/
        b_image_1.png
        b_image_2.png

The function is documented [here](https://keras.io/api/preprocessing/image/). It has only one required argument, which is the top-level folder for the dataset, but there are several optional arguments that can be used to configure the dataset. We highlight a few even though we are using the default values for some of them. Among them, we have the option for how to specify the label encoding. Here we decided to use integer encoding, which is the default (`label_mode='int'`), rather than one-hot label encoding. Either option can be used, but there is additional code further below in this implementation that assumes integer encoding.


```python
train_dataset = image_dataset_from_directory(directory=DatasetConfig.DATA_ROOT_TRAIN,
                                             batch_size=TrainingConfig.BATCH_SIZE,
                                             shuffle=True,
                                             seed=SEED_VALUE,
                                             label_mode='int', # Use integer encoding
                                             image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
                                            )

valid_dataset = image_dataset_from_directory(directory=DatasetConfig.DATA_ROOT_VALID,
                                             batch_size=TrainingConfig.BATCH_SIZE,
                                             shuffle=True,
                                             seed=SEED_VALUE,
                                             label_mode='int', # Use integer encoding
                                             image_size= (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH),
                                            )

print(train_dataset.class_names)
```


### 4.1 Display Sample Images from Training Dataset

```python
class_names = train_dataset.class_names

plt.figure(figsize=(18, 10))

# Assumes dataset batch_size is at least 32.
num_rows = 4
num_cols = 8

# Here we use the take() method to retrieve just the first batch of data from the training portion of the dataset.
for image_batch, labels_batch in train_dataset.take(1):
    # Plot each of the images in the batch and the associated ground truth labels.
    for i in range(num_rows * num_cols):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        truth_idx = labels_batch[i].numpy()
        plt.title(class_names[truth_idx])
        plt.axis("off")
```


## 5 Create the Test Dataset


It was easy to create the training and validation datasets using `image_dataset_from_directory()`. However, since the images and labels for the Test dataset are stored separately on the filesystem, we'll need to write some custom code to read and load this data. To create the test dataset, we will need to load the images and the labels in memory and combine them to create a (`tf.data.Dataset`) test dataset. Four steps are required:

1. Retrieve the class labels from the provided `csv` file and store them in memory as a Python list
2. Build a list of the image file paths in memory as a Python list
3. Combine the image paths and associated labels in a `tf.data.Dataset` object
4. Use the dataset object `map` method to the load and preprocess the images in the dataset

### 5.1 Load Test Ground Truth Labels

The ground truth labels for the test dataset are listed in a `csv` file (`Test.csv`)

We can easily extract the class IDs from this file using a Pandas dataframe. The following code cell reads the class IDs from this file and stores them in a Python list.

```python
import pandas as pd

input_file = DatasetConfig.DATA_TEST_GT

dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
cols = [6]
df = df[df.columns[cols]]
ground_truth_ids = df["ClassId"].values.tolist()
print("Total number of Test labels: ", len(ground_truth_ids))
print(ground_truth_ids[0:10])
print(train_dataset.class_names)
```

### 5.1.1 Map Ground Truth Class IDs to IDs in Train/Valid Datasets

To create a test dataset that is consistent with the label IDs in the training and validation datasets we need to build a dictionary mapping ground truth IDs in the test dataset to class name IDs. This will ensure that the ground truth labels in the test dataset are correctly interpreted by the model.

**_Note_**: This step is not normally required but is an artifact of the way the class names were inferred from the class folder names. An alternative solution would be to use the `class_names` argument in `image_dataset_from_directory()`, in which we explicitly specify the order of the class names to be in numerical order.

```python
# Convert train/valid class names to integers.
class_names_int = list(map(int, train_dataset.class_names))

# Create a dictionary mapping ground truth IDs to class name IDs.
gtid_2_cnidx = dict(zip(class_names_int, range(0, DatasetConfig.NUM_CLASSES)))

gtid_2_cnidx.items()

# Convert the ground truth Class IDs to IDs that correctly map to the same classes
# in the train and validation datasets.
label_ids = []
for idx in range(len(ground_truth_ids)):
    label_ids.append(gtid_2_cnidx[ground_truth_ids[idx]])

print("Original ground truth class IDs: ", ground_truth_ids[0:10])
print("New mapping required:            ", label_ids[0:10])
print("")
print("Train/Valid dataset class names: ", train_dataset.class_names)
```

### 5.2 Create File Paths to Test Images

```python
# Get all the path names to the Test images (will prune later)
image_paths = sorted(glob.glob(DatasetConfig.DATA_ROOT_TEST + os.sep + "*.png"))

print(len(image_paths))
print("")
# Print the first 5 image paths to confirm.
for idx in range(5):
    print(image_paths[idx])
```

### 5.3 Combine images and labels to create the Test dataset

`test_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_ids))`

### 5.4 Load and Preprocess the Images

```python
def preprocess_image(image):
    # Decode and resize image.
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH])
    return image

def load_and_preprocess_image(path):
    # Read image into memory as a byte string.
    image = tf.io.read_file(path)
    return preprocess_image(image)

# Apply the functions above to the test dataset.
test_dataset = test_dataset.map(load_and_preprocess_from_path_label)

# Set the batch size for the dataset.
test_dataset = test_dataset.batch(TrainingConfig.BATCH_SIZE)
```

### 5.5 Display Sample Images from the Test Dataset

```python
plt.figure(figsize=(18, 10))

# Assumes dataset batch_size is at least 32.
num_rows = 4
num_cols = 8

# Here we use the take() method to retrieve just the first batch of data from the test dataset.
for image_batch, labels_batch in test_dataset.take(1):
    
    # Plot each of the images in the batch and the associated ground truth labels.
    for i in range(num_rows * num_cols):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        truth_idx = labels_batch[i].numpy()
        plt.title(class_names[truth_idx])
        plt.axis("off")
```


## 6 Modeling VGG-16 (for Fine-Tuning)

The Keras API provides the following utility that is used to instantiate a VGG-16 model. The default settings are shown below for the pre-trained ImageNet model.

```python
tf.keras.applications.vgg16.VGG16(include_top=True, 
                                  weights='imagenet', 
                                  input_tensor=None,
                                  input_shape=None, 
                                  pooling=None, 
                                  classes=1000,
                                  classifier_activation='softmax',
                                 )
```

To configure the model for Fine-Tuning, we will load the model's convolutional base with weights that were learned from the ImageNet dataset. These weights serve as a starting point for Fine-Tuning the model for our dataset. Since we need to redefine the classifier, we will load the model without a classifier (`include_top=False`), so we can specify our own classifier for the dataset.

For more information on the VGG-16 model available in Keras, here is the documentation link: [Keras VGG-16 Model API](https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/VGG16).

### 6.1 Loading the VGG-16 Convolutional Base

We begin by creating a model of the VGG-16 convolutional base. We can do this by instantiating the model and setting `include_top = False`, which excludes the fully connected layers. We will also instantiate the model with weights that were learned by training the model on the ImageNet dataset `weights='imagenet'`.

```python
# Specify the model input shape.
input_shape = (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH, DatasetConfig.CHANNELS)

print('Loading model with ImageNet weights...')
vgg16_conv_base = tf.keras.applications.vgg16.VGG16(input_shape=input_shape,
                                                    include_top=False, # We will supply our own top.
                                                    weights='imagenet',
                                                   )
vgg16_conv_base.summary()
```

### 6.2 Freeze the Initial Layers in the Convolutional Base

Now that we have loaded the convolutional base, we need to lock down the initial layers so that only the last few laters (`TrainingConfig.LAYERS_FINE_TUNE = 8`) are trainable. There are two ways to specify which layers in the model are trainable (tunable).

1. We can start by making the entire convolutional base trainable by setting the trainable flag to True. Then loop over the initial layers and make them untrainable by setting the same (`trainable`) flag for each layer to False.
    
2. We can freeze the entire convolutional base by setting the trainable flag to False, and then loop over the last few layers and set the `trainable` flag to `True`.
    

We use the first approach in the code cell below. We start by setting the entire convolutional base as "trainable" by setting the `trainable`e attribute to `True`.


```python
# Set all layers in the convolutional base to Trainable (will FREEZE initial layers further below).
vgg16_conv_base.trainable = True

# Specify the number of layers to fine tune at the end of the convolutional base.
num_layers_fine_tune = TrainingConfig.LAYERS_FINE_TUNE
num_layers = len(vgg16_conv_base.layers)

# Freeze the initial layers in the convolutional base.
for model_layer in vgg16_conv_base.layers[: num_layers - num_layers_fine_tune]:
    print(f"FREEZING LAYER: {model_layer}")
    model_layer.trainable = False

print("\n")
print(f"Configured to fine tune the last {num_layers_fine_tune} convolutional layers...")
print("\n")

vgg16_conv_base.summary()
```


### 6.3 Add the Classifier to Complete the Model

Since we intend to train and use the model to classify traffic signs (43 classes), we will need to add our own classification layer. In this example, we have chosen to use just a single fully connected dense layer that contains 128 nodes, followed by a softmax output layer that contains 43 nodes for each of the 43 classes. The number of dense layers and the number of nodes per layer is a design choice, but the number of nodes in the output layer must match the number of classes in the dataset. Because we are working with a very small dataset (40 samples per class), the model will be prone to overfitting, so we have also added a dropout layer in the classifier. The entire model is then assembled as shown below.


```python
inputs = tf.keras.Input(shape=input_shape)

x = tf.keras.applications.vgg16.preprocess_input(inputs)

x = vgg16_conv_base(x)

# Flatten the output from the convolutional base.
x = layers.Flatten()(x)

# Add the classifier.
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(TrainingConfig.DROPOUT)(x)

# Output layer.
outputs = layers.Dense(DatasetConfig.NUM_CLASSES, activation="softmax")(x)

# The final model.
model_vgg16_finetune = keras.Model(inputs, outputs)

model_vgg16_finetune.summary()
```


### 6.4 Compile and Train the Model

Here we use `SparseCategoricalCrossentropy` since we are using integer-encoded labels. For one-hot encoded labels, the appropriate loss function would be `CategoricalCrossentropy`. Since we included a Softmax layer in the model output, we specify `from_logits=False`. This is the default setting, but it's good practice to be explicit. Alternatively, you can remove the softmax layer in the model and set `from_logits=True`, and the loss function will apply the softmax function internally. The results should be identical.

```python
# Use this for integer encoded labels.
model_vgg16_finetune.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=TrainingConfig.LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

# Train the Model.
training_results = model_vgg16_finetune.fit(train_dataset,
	                                            epochs=TrainingConfig.EPOCHS,
                                            validation_data=valid_dataset,
                                           )

```


### 6.4 Plot the Training Results

The convenience function below is used to plot both the training and validation loss and accuracy.


```python
def plot_results(metrics, ylabel=None, ylim=None, metric_name=None, color=None):
    fig, ax = plt.subplots(figsize=(15, 4))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]

    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xlim([0, TrainingConfig.EPOCHS - 1])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()
    plt.close()
```


```python
# Retrieve training results.
train_loss = training_results.history["loss"]
train_acc  = training_results.history["accuracy"]
valid_loss = training_results.history["val_loss"]
valid_acc  = training_results.history["val_accuracy"]

plot_results(
    [train_loss, valid_loss],
    ylabel="Loss",
    ylim=[0.0, 5.0],
    metric_name=["Training Loss", "Validation Loss"],
    color=["g", "b"],
)

plot_results(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.0, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"],
)
```

The training plots above clearly show a rapid decrease in loss and increase in accuracy, which can be attributed to the model being initialized with pre-trained weights. It's worth noting that despite the small size of the dataset and the variations in image quality, the validation accuracy is impressive, exceeding 95%. Techniques such as data augmentation could be employed to improve the accuracy even further. Overall, these results are encouraging and indicate that the model has learned to generalize well to unseen data.

![](Pasted%20image%2020250503222354.png)

![](Pasted%20image%2020250503222348.png)

### 6.4.1 Comparison: Fine-Tuning vs. Training from Scratch

![](Pasted%20image%2020250503222409.png)


## 7 Model Evaluation

### 7.1 Validation Dataset

`print(f"Model valid accuracy: {model_vgg16_finetune.evaluate(valid_dataset)[1]*100.:.3f}")`


### 7.2 Test Dataset

`print(f"Model test accuracy: {model_vgg16_finetune.evaluate(test_dataset)[1]*100.:.3f}")`


### 7.3 Display Sample Predictions

```python
def display_predictions(dataset, model, class_names):
    
    plt.figure(figsize=(20, 20))
    num_rows = 8
    num_cols = 8
    jdx = 0

    # Evaluate two batches.
    for image_batch, labels_batch in dataset.take(2):
        print(image_batch.shape)

        # Predictions for the current batch.
        predictions = model.predict(image_batch)

        # Loop over all the images in the current batch.
        for idx in range(len(labels_batch)):
            pred_idx = tf.argmax(predictions[idx]).numpy()
            truth_idx = labels_batch[idx].numpy()

            # Set the title color based on the prediction.
            if pred_idx == truth_idx:
                color = "g"
            else:
                color = "r"

            jdx += 1

            if jdx > num_rows * num_cols:
                # Break from the loops if the maximum number of images have been plotted
                break

            ax = plt.subplot(num_rows, num_cols, jdx)
            title = str(class_names[truth_idx]) + " : " + str(class_names[pred_idx])

            title_obj = plt.title(title)
            plt.setp(title_obj, color=color)
            plt.axis("off")
            plt.imshow(image_batch[idx].numpy().astype("uint8"))
    return

display_predictions(valid_dataset, model_vgg16_finetune, class_names)
display_predictions(test_dataset, model_vgg16_finetune, class_names)
```

![](Pasted%20image%2020250503222743.png)



