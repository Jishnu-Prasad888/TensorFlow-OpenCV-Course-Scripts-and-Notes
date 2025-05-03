import os
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Conv2D , MaxPooling2D , Dropout , Flatten

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib.ticker import ( MultipleLocator , FormatStrFormatter)
from dataclasses import dataclass

SEED_VALUE = 42

random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

(X_train , y_train) , (X_test , y_test) = cifar10.load_data()


def display_sample_images_from_dataset():

    print(X_train.shape)
    print(X_test.shape)

    plt.figure(figsize=(18,8))

    num_rows = 4
    num_colmuns = 8

    # PLotting each image
    for i in range(num_rows*num_colmuns):
        ax = plt.subplot(num_rows,num_colmuns , i + 1)
        plt.imshow(X_train[i, :, :])
        plt.axis("off")
    
    plt.show()

X_train = X_train.astype("float32") / 255
X_test  = X_test.astype("float32")  / 255

# print('Original (integer) label for the first training sample: ', y_train[0])

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

# print('After conversion to categorical one-hot encoded labels: ', y_train[0])


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

def cnn_model(input_shape = (32,32,3)):
    
    model = Sequential()

    #------------------------------------
    # Conv Block 1: 32 Filters, MaxPool.
    #------------------------------------

    model.add(
        Conv2D(
            filters = 32,
            kernel_size = 3,
            padding  = 'same',
            activation = 'relu',
            input_shape = input_shape
        )
    )

    model.add(
        Conv2D(
            filters = 32,
            kernel_size = 3,
            padding  = 'same',
            activation = 'relu',
        )
    )

    model.add(
        MaxPooling2D(
            pool_size = (2,2)
        )
    )

    model.add(Dropout(0.25))

    #------------------------------------
    # Conv Block 2: 64 Filters, MaxPool.
    #------------------------------------
    
    model.add(
        Conv2D(
            filters = 64,
            kernel_size = 3,
            padding  = 'same',
            activation = 'relu',
        )
    )

    model.add(
        Conv2D(
            filters = 64,
            kernel_size = 3,
            padding  = 'same',
            activation = 'relu',
        )
    )

    model.add(
        MaxPooling2D(
            pool_size = (2,2)
        )
    )

    model.add(Dropout(0.25))
    
    #------------------------------------
    # Conv Block 3: 64 Filters, MaxPool.
    #------------------------------------

    
    model.add(
        Conv2D(
            filters = 64,
            kernel_size = 3,
            padding  = 'same',
            activation = 'relu',
        )
    )

    model.add(
        Conv2D(
            filters = 64,
            kernel_size = 3,
            padding  = 'same',
            activation = 'relu',
        )
    )

    model.add(
        MaxPooling2D(
            pool_size = (2,2)
        )
    )    
    model.add(Dropout(0.25))
    #------------------------------------
    # Flatten the convolutional features.
    #------------------------------------

    model.add(Flatten())
    model.add(
        Dense(
            512,
            activation = 'relu'
        )
    )

    model.add(Dropout(0.5))
    
    model.add(
        Dense(
            10,
            activation = 'softmax'
        )
    )

    return model

model = cnn_model()
model.summary()

model.compile(
    optimizer = "rmsprop",
    loss = "categorical_crossentropy",
    metrics = ["accuracy"]
)

history = model.fit(
                    X_train,
                    y_train,
                    batch_size = TrainingConfig.BATCH_SIZE,
                    epochs = TrainingConfig.EPOCHS,
                    verbose = 1,
                    validation_split = .3
                )


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


# train_loss = history.history["loss"]
# train_acc = history.history["accuracy"]
# valid_loss = history.history["val_loss"]
# valid_acc = history.history["val_accuracy"]



# plot_results(
#     [train_loss, valid_loss],
#     ylabel="Loss",
#     ylim=[0.0, 5.0],
#     metric_name=["Training Loss", "Validation Loss"],
#     color=["g", "b"],
# )

# plt.show()

# plot_results(
#     [train_acc, valid_acc],
#     ylabel="Accuracy",
#     ylim=[0.0, 1.0],
#     metric_name=["Training Accuracy", "Validation Accuracy"],
#     color=["g", "b"],
# )

# plt.show()

model.save("model_dropout")
