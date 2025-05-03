import os
import pathlib
import random
import numpy as np
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

import pandas as pd

from zipfile import ZipFile
from urllib.request import urlretrieve

SEED_VALUE = 41

random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

def download_and_unzip(url, save_path):
    print(f"Downloading and extracting assets .....", end="")
    urlretrieve(url, save_path)
    try:
        with ZipFile(save_path) as z:
            z.extractall(os.path.split(save_path)[0])
        print("Done")
    except Exception as e:
        print("\n Invalid file : ", e)

URL = r"https://www.dropbox.com/s/uzgh5g2bnz40o13/dataset_traffic_signs_40_samples_per_class.zip?dl=1"

dataset_path = os.path.join(os.getcwd(), "dataset_traffic_signs_40_samples_per_class")
asset_zip_path = os.path.join(os.getcwd(), "dataset_traffic_signs_40_samples_per_class.zip")

if not os.path.exists(asset_zip_path):
    download_and_unzip(URL, asset_zip_path)

@dataclass(frozen=True)
class DatasetConfig:
    NUM_CLASSES: int = 43
    IMG_HEIGHT: int = 224
    IMG_WIDTH: int = 224
    CHANNELS: int = 3
    DATA_ROOT_TRAIN: str = os.path.join(dataset_path, "Train")
    DATA_ROOT_VALID: str = os.path.join(dataset_path, "Valid")
    DATA_ROOT_TEST: str = os.path.join(dataset_path, "Test")
    DATA_TEST_GT: str = os.path.join(dataset_path, "Test.csv")

@dataclass(frozen=True)
class TrainingConfig:
    BATCH_SIZE: int = 32
    EPOCHS: int = 101
    LEARNING_RATE: float = 0.0001
    DROPOUT: float = 0.6
    LAYERS_FINE_TUNE: int = 8

train_dataset = image_dataset_from_directory(
    directory=DatasetConfig.DATA_ROOT_TRAIN,
    batch_size=TrainingConfig.BATCH_SIZE,
    shuffle=True,
    seed=SEED_VALUE,
    label_mode="int",
    image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH)
)

valid_dataset = image_dataset_from_directory(
    directory=DatasetConfig.DATA_ROOT_VALID,
    batch_size=TrainingConfig.BATCH_SIZE,
    shuffle=True,
    seed=SEED_VALUE,
    label_mode='int',
    image_size=(DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH)
)

print(train_dataset.class_names)
class_names = train_dataset.class_names

input_file = DatasetConfig.DATA_TEST_GT
dataset = pd.read_csv(input_file)
df = pd.DataFrame(dataset)
ground_truth_ids = df["ClassId"].values.tolist()
print("Total number of test labels : " + str(len(ground_truth_ids)))
print(ground_truth_ids[0:10])

class_names_int = list(map(int, train_dataset.class_names))
gtid_2_cnidx = dict(zip(class_names_int, range(0, DatasetConfig.NUM_CLASSES)))

label_ids = []
for idx in range(len(ground_truth_ids)):
    label_ids.append(gtid_2_cnidx[ground_truth_ids[idx]])

print("Original ground truth class IDs: ", ground_truth_ids[0:10])
print("New mapping required:            ", label_ids[0:10])
print("\nTrain/Valid dataset class names: ", train_dataset.class_names)

image_paths = sorted(glob.glob(DatasetConfig.DATA_ROOT_TEST + os.sep + "*.png"))
print(len(image_paths))
for idx in range(5):
    print(image_paths[idx])

test_dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_ids))

def preprocess_image(image):
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH])
    return image

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def load_and_preprocess_from_path_label(path, label):
    image = load_and_preprocess_image(path)
    return image, label

test_dataset = test_dataset.map(load_and_preprocess_from_path_label)
test_dataset = test_dataset.batch(TrainingConfig.BATCH_SIZE)

input_shape = (DatasetConfig.IMG_HEIGHT, DatasetConfig.IMG_WIDTH, DatasetConfig.CHANNELS)
print("Loading models with ImageNet weights .....")

vgg16_conv_base = tf.keras.applications.vgg16.VGG16(
    input_shape=input_shape,
    include_top=False,
    weights='imagenet'
)

vgg16_conv_base.summary()
vgg16_conv_base.trainable = True
num_layers_fine_tune = TrainingConfig.LAYERS_FINE_TUNE
num_layers = len(vgg16_conv_base.layers)

for model_layer in vgg16_conv_base.layers[:num_layers - num_layers_fine_tune]:
    print(f"Freezing Conv Base layer : {model_layer.name}")
    model_layer.trainable = False

print("\nConfigured to fine tune the last", num_layers_fine_tune, "convolutional layers ...\n")
vgg16_conv_base.summary()

inputs = tf.keras.Input(shape=input_shape)
x = tf.keras.applications.vgg16.preprocess_input(inputs)
x = vgg16_conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(TrainingConfig.DROPOUT)(x)
outputs = layers.Dense(DatasetConfig.NUM_CLASSES, activation="softmax")(x)

model_vgg_finetune = keras.Model(inputs, outputs)
model_vgg_finetune.summary()

model_vgg_finetune.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=TrainingConfig.LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

training_results = model_vgg_finetune.fit(
    train_dataset,
    epochs=TrainingConfig.EPOCHS,
    validation_data=valid_dataset
)

def plot_results(metrics, ylabel=None, ylim=None, metric_name=None, color=None):
    fig, ax = plt.subplots(figsize=(15, 4))
    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics, ]
        metric_name = [metric_name, ]
    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(ylabel)
    plt.xlim([0, TrainingConfig.EPOCHS - 1])
    plt.ylim(ylim)
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()
    plt.close()

train_loss = training_results.history["loss"]
train_acc = training_results.history["accuracy"]
valid_loss = training_results.history["val_loss"]
valid_acc = training_results.history["val_accuracy"]

plot_results(
    [train_loss, valid_loss],
    ylabel="Loss",
    ylim=[0.0, 5.0],
    metric_name=["Training Loss", "Validation Loss"],
    color=["g", "b"]
)

plt.show()

plot_results(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.0, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"]
)

plt.show()


def display_predictions(dataset, model , class_name):
    
    plt.figure(figsize=(20,20))
    num_rows = 8
    num_cols = 8
    jdx = 0

    for image_batch , labels_batch in dataset.take(2):
        print(image_batch.size)

        predictions = model.predict(image_batch)

        for idx in range(len(image_batch)):
            pred_idx = tf.argmax(predictions[idx]).numpy()
            truth_idx = labels_batch[idx].numpy()

            if pred_idx == truth_idx:
                color = "g"
            else :
                color = 'r'
            jdx += 1

            if jdx > num_rows * num_cols:
                break

            ax = plt.subplot(num_rows, num_cols , jdx)
            title = str(class_names[truth_idx] + " : " + str(class_names[predictions]))

            title_obj = plt.title(title)
            plt.setp(title_obj , color = color)
            plt.axis("off")
            plt.imshow(image_batch[idx].numpy().astype("uint8"))
    plt.show()
    return

display_predictions(valid_dataset , model_vgg_finetune , class_names)
display_predictions(test_dataset , model_vgg_finetune ,class_names)