# Implementing a CNN in TensorFlow and Keras

```python
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
from dataclasses import dataclass

SEED_VALUE = 42

# Fix seed to make training deterministic.
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)
```

## 1 Load the CIFAR-10 Dataset

- The CIFAR-10 dataset is included in TensorFlow, so we can load the dataset using the `load_data()` function as shown in the code cell below and confirm the number of samples and the shape of the data.

```python
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

## 2 Dataset Preprocessing

- Here, we normalize the image data to the range `[0,1]`. This is very common when working with image data which helps the model train more efficiently. We also convert the integer labels to one-hot encoded labels

```python
# Normalize images to the range [0, 1].
X_train = X_train.astype("float32") / 255
X_test  = X_test.astype("float32")  / 255

# Change the labels from integer to categorical data.
print('Original (integer) label for the first training sample: ', y_train[0])

# Convert labels to one-hot encoding.
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

print('After conversion to categorical one-hot encoded labels: ', y_train[0])
```


## 3 Dataset and Training Configuration Parameters

- Before we describe the model implementation and training, we’re going to apply a little more structure to our training process by using the `dataclasses` module in python to create simple `DatasetConfig` and `TrainingConfig` classes to organize several data and training configuration parameters. This allows us to create data structures for configuration parameters, as shown below. The benefit of doing this is that we have a single place to go to make any desired changes.

```python
@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES:  int = 10
    IMG_HEIGHT:   int = 32
    IMG_WIDTH:    int = 32
    NUM_CHANNELS: int = 3
    
@dataclass(frozen=True)
class TrainingConfig:
    EPOCHS:        int = 31
    BATCH_SIZE:    int = 256
    LEARNING_RATE: float = 0.001 
```


## 4 CNN Model Implementation in Keras

### 4.1 Model Structure

- The model contains three convolutional blocks followed by a fully connected layer and an output layer. For reference, we’ve included the number of channels at key points in the architecture. We have also indicated the spatial size of the activation maps at the end of each convolutional block. This is a good visual to refer back to when studying the code below.
- For convenience, we’re going to define the model in a function. Notice that the function has one optional argument: the input shape for the model. We first start by instantiating the model by calling the `sequential()` method. This allows us to build a model sequentially by adding one layer at a time. Noticed that we define three convolutional blocks and that their structure is very similar.

### 4.2 Define the Convolutional Blocks

- Let’s start with the very first convolutional layer in the first convolutional block. To define a convolutional layer in Keras, we call the `Conv2D()` function, which takes several input arguments. 
- First, we defined the layer to have 32 filters. The kernel size for each filter is 3 (which is interpreted as 3x3). We use a padding option called `same`, which will pad the input tensor so that the output of the convolution operation has the same spatial size as the input. This is not required, but it’s commonly used.
- if you don’t explicitly specify this padding option, then the default behavior has no padding, and therefore, the spatial size of output from the convolutional layer will be slightly smaller than the input size.
- We use a `ReLU` activation function in all the layers in the Network except for the output layer.
- For the very first convolutional layer, we need to specify the shape of the input, but for all subsequent layers, this is not necessary since the shape of the input is automatically computed based on the shape of the output from previous layers, so we have two convolutional layers with 32 filters each, and then we follow that with a max pooling layer that has a window size of (2x2), so the output shape from this first convolution block is (16x16 x32). 
- Next, we have the second convolutional block, which is nearly identical to the first, with the exception that we have 64 filters in each convolutional layer instead of 32, and then finally, the third convolutional block is an exact copy of the second convolutional block.
- The number of filters in each convolutional layer is something that you will need to experiment with. A larger number of filters allows the model to have a greater learning capacity, but this also needs to be balanced with the amount of data available to train the model. Adding too many filters (or layers) can lead to overfitting, one of the most common issues encountered when training models.

```python
def cnn_model(input_shape=(32, 32, 3)):
    
    model = Sequential()
    
    #------------------------------------
    # Conv Block 1: 32 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #------------------------------------
    # Conv Block 2: 64 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #------------------------------------
    # Conv Block 3: 64 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #------------------------------------
    # Flatten the convolutional features.
    #------------------------------------
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    
    return model
```


### 4.3 Create the Model

- We can now create an instance of the model by calling the function above and use the `summary()` method to display the model summary to the console.

```python
# Create the model.
model = cnn_model()
model.summary()
```

### 4.4 Compile the Model

- The next step is to compile the model, where we specify the optimizer type and loss function and any additional metrics we would like recorded during training. 
- Here we specify `RMSProp` as the optimizer type for gradient descent, and we use a cross-entropy loss function which is the standard loss function for classification problems.
- We specifically use `categorical_crossentropy` since our labels are one-hot encoded. 
- Finally, we specify `accuracy` as an additional metric to record during training. 
- The value of the loss function is always recorded by default, but if you want accuracy, you need to specify it.

```python
model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

### 4.5 Train the Model

- Since the dataset does not include a validation dataset, and since we did not previously split the training dataset to create a validation dataset, we will use the `validation_split` argument below so that 30% of the training dataset is automatically reserved for validation. 
- This approach reserves the last 30% of the dataset for training. This is a very convenient approach, but *if the training dataset has any specific ordering (say, ordered by classes), you will need to take steps to randomize the order before splitting.*

```python
history = model.fit(X_train,
                    y_train,
                    batch_size=TrainingConfig.BATCH_SIZE, 
                    epochs=TrainingConfig.EPOCHS, 
                    verbose=1, 
                    validation_split=.3,
                   )
```


### 4.6 Plot the Training Results

```python
def plot_results(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    fig, ax = plt.subplots(figsize=(15, 4))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics,]
        metric_name = [metric_name,]

    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
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

# Retrieve training results.
train_loss = history.history["loss"]
train_acc  = history.history["accuracy"]
valid_loss = history.history["val_loss"]
valid_acc  = history.history["val_accuracy"]

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

![](Pasted%20image%2020250503105810.png)

![](Pasted%20image%2020250503105816.png)

- The results from our baseline model reveal that the model is overfitting. Notice that the validation loss increases after about ten epochs of training while the training loss continues to decline. This means that the network learns how to model the training data well but does not generalize to unseen test data well. It is basically memorizing the data instead of learning it
- The accuracy plot shows a similar trend where the validation accuracy levels off after about ten epochs while the training accuracy continues to approach 100% as training progresses. This is a common problem when training neural networks and can occur for a number of reasons. One reason is that the model can fit the nuances of the training dataset, especially when the training dataset is small.

## 5 Adding `Dropout` to the Model

- To help mitigate this problem, we can employ one or more **regularization** strategies to help the model generalize better
- . Regularization techniques help to restrict the model's flexibility so that it doesn't overfit the training data. One approach is called **Dropout**, which is built into Keras. Dropout is implemented in Keras as a special layer type that randomly drops a percentage of neurons during the training process. 
- When dropout is used in convolutional layers, it is usually used **after** **the max pooling layer** and has the effect of eliminating a percentage of neurons in the feature maps. 

![](Pasted%20image%2020250503110052.png)

![](Pasted%20image%2020250503110103.png)

### 5.1 Define the Model (with Dropout)

```python
def cnn_model_dropout(input_shape=(32, 32, 3)):
    
    model = Sequential()
    
    #------------------------------------
    # Conv Block 1: 32 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #------------------------------------
    # Conv Block 2: 64 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #------------------------------------
    # Conv Block 3: 64 Filters, MaxPool.
    #------------------------------------
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    #------------------------------------
    # Flatten the convolutional features.
    #------------------------------------
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model

# Create the model.
model_dropout = cnn_model_dropout()
model_dropout.summary()

model_dropout.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model_dropout.fit(X_train,
                            y_train,
                            batch_size=TrainingConfig.BATCH_SIZE, 
                            epochs=TrainingConfig.EPOCHS, 
                            verbose=1, 
                            validation_split=.3,
                           )

# Retrieve training results.
train_loss = history.history["loss"]
train_acc  = history.history["accuracy"]
valid_loss = history.history["val_loss"]
valid_acc  = history.history["val_accuracy"]

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

![](Pasted%20image%2020250503110222.png)

![](Pasted%20image%2020250503110226.png)

- In the plots above, the training curves align very closely with the validation curves. Also, notice that we achieve a higher validation accuracy than the baseline model that did not contain dropout

## 6 Saving and Loading Models

### 6.1 Saving Models

- You can easily save a model using the `save()` method which will save the model to the file system in the 'SavedModel' format. 
- This method creates a folder on the file system. Within this folder, the model architecture and training configuration (including the optimizer, losses, and metrics) are stored in `saved_model.pb`.
- The `variables/` folder contains a standard training checkpoint file that includes the weights of the model.

```python
# Using the save() method, the model will be saved to the file system in the 'SavedModel' format.
model_dropout.save("model_dropout")
```

### 6.2 Loading Models

```python
from tensorflow.keras import models
reloaded_model_dropout = models.load_model('model_dropout')
```

## 7 Model Evaluation

### 7.1 Evaluate the Model on the Test Dataset

```python
test_loss, test_acc = reloaded_model_dropout.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc*100:.3f}")
```

### 7.2 Make Predictions on Sample Test Images

```python
def evaluate_model(dataset, model):
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    num_rows = 3
    num_cols = 6

    # Retrieve a number of images from the dataset.
    data_batch = dataset[0 : num_rows * num_cols]

    # Get predictions from model.
    predictions = model.predict(data_batch)

    plt.figure(figsize=(20, 8))
    num_matches = 0

    for idx in range(num_rows * num_cols):
        ax = plt.subplot(num_rows, num_cols, idx + 1)
        plt.axis("off")
        plt.imshow(data_batch[idx])

        pred_idx = tf.argmax(predictions[idx]).numpy()
        truth_idx = np.nonzero(y_test[idx])

        title = str(class_names[truth_idx[0][0]]) + " : " + str(class_names[pred_idx])
        title_obj = plt.title(title, fontdict={"fontsize": 13})

        if pred_idx == truth_idx:
            num_matches += 1
            plt.setp(title_obj, color="g")
        else:
            plt.setp(title_obj, color="r")

        acc = num_matches / (idx + 1)
    print("Prediction accuracy: ", int(100 * acc) / 100)

    return

evaluate_model(X_test, reloaded_model_dropout)
```

### 7.3 Confusion Matrix

- A confusion matrix is a very common **metric** that is used to summarize the results of a classification problem. 
- The information is presented in the form of a table or matrix where one axis represents the ground truth labels for each class, and the other axis represents the predicted labels from the network.
- The entries in the table represent the number of instances from an experiment (which are sometimes represented as percentages rather than counts). 
- Generating a confusion matrix in TensorFlow is accomplished by calling the `function tf.math.confusion_matrix()`, which takes two required arguments: the list of ground truth labels and the associated predicted labels.

```python
# Generate predictions for the test dataset.
predictions = reloaded_model_dropout.predict(X_test)

# For each sample image in the test dataset, select the class label with the highest probability.
predicted_labels = [np.argmax(i) for i in predictions]

# Convert one-hot encoded labels to integers.
y_test_integer_labels = tf.argmax(y_test, axis=1)

# Generate a confusion matrix for the test dataset.
cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

# Plot the confusion matrix as a heatmap.
plt.figure(figsize=[12, 6])
import seaborn as sn

sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 12})
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
```

![](Pasted%20image%2020250503110601.png)

- Two of the ten classes tend to be misclassified more than others: Dogs and Cat. More specifically, a large percentage of the time, the model confuses these two classes with each other. Let’s take a closer look. The ground truth label for a cat is 3, and the ground truth label for a dog is 5. Notice that when the input image is a cat (index 3), it is often most misclassified as a dog, with 176 misclassified samples. When the input image is a dog (index 5), the most misclassified examples are cats, with 117 samples. 

- Also, notice that the last row, which represents trucks, is most often confused with automobiles. So all of these observations make intuitive sense, given the similarity of the classes involved.


