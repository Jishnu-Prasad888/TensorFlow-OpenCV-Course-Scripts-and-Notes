
#  Image Classification using Feedforward Networks in Keras


- classify hand-written digits from the MNIST dataset using a feed-forward Multilayer Perceptron Network. MLPs are not the preferred way to process image data, but this serves as a good example to introduce some new concepts. The MNIST hand-written digit dataset is included in Tensorflow and can easily be imported and loaded, as we will see below. Using this dataset and a simple feed-forward network, we will demonstrate one approach for how to work with image data and build a network that classifies digits `[0,9]`.

## Dataset Preprocessing

###  Input Feature Transformation and Normalization

- Since we are now working with images as the input, we need to find some logical way to represent the image data as a set of features. A naive approach that actually works fairly well for this dataset is to just assume that the pixel intensities **are** the features.
- we also normalize the pixel intensities to be in the range `[0, 1]`. This is very common when working with image data which helps the model train more efficiently.

`X_train = X_train.reshape((X_train.shape[0], 28 * 28))`
This reshapes the training data from 2D image format (28×28 pixels) to a 1D flat vector of 784 elements (since 28 × 28 = 784).
`X_train.shape[0] `refers to the number of images (samples), so this keeps the sample count the same but flattens each image.
If `X_train` originally has shape (60000, 28, 28) (60,000 images, each 28x28), after this line it becomes:
`X_train.shape → (60000, 784)`

`X_train = X_train.astype("float32") / 255`
- Converts the data type to `float32` (which is required by most deep learning frameworks).
- Divides every pixel value by 255 to **normalize** pixel intensities from range `[0, 255]` to `[0, 1]`.
- Neural networks perform better when input values are scaled to a small range (like `[0, 1]` or `[-1, 1]`).

### 2.2 Label Encoding Options

- When working with categorical data, the target labels need to be represented as numerical values prior to processing the data through machine learning algorithms.
- Label encoding is the process of converting class labels from strings to numerical values. We have a couple of options for how to numerically encode the labels for each class. We can use ordinal **integer encoding**, where an integer is assigned to each class, or we can use a technique called **one-hot encoding**, which uses a separate binary vector to encode each class label. 
- Depending on the dataset, one approach might be preferred over the other, but for most situations, one-hot encoding is often used.

#### 2.2.1 Integer Label Encoding

As a concrete example, consider the dictionary mapping shown below for the Fashion MNIST dataset.

Label   Description
0       T-shirt/top
1       Trouser
2       Pullover
3       Dress
4       Coat
5       Sandal
6       Shirt
7       Sneaker
8       Bag
9       Ankle boot

The Fashion MNIST dataset itself contains the integer labels which we can verify this by loading the dataset and printing out some of the labels as shown in the output from the code cell below. This type of label encoding is called **Integer Encoding** because unique integers are used to encode the class (string) labels. 

However, when the class labels have no relationship to one another, it is often recommended that **One-Hot Encoding** is used instead

-------
### 2.2.2  Why One-Hot Encoding?

Suppose you have three classes:

- `Class A → 0`
- `Class B → 1` 
- `Class C → 2`
If you encode these as plain integers (`0`, `1`, `2`), you're **implicitly introducing ordinal relationships**:
- The model may interpret `Class C > Class B > Class A`, which is **not true** if the classes are categorical and unordered.
This can mislead many machine learning algorithms (especially linear models or neural networks), causing **biased learning**.

### One-Hot Encoding solves this:

Each class gets its **own binary feature**:

- `Class A → [1, 0, 0]`
- `Class B → [0, 1, 0]`
- `Class C → [0, 0, 1]`

#### ✔ No ordering or magnitude:

- There's no implied hierarchy.
- The distance between any two class encodings is the same.

So rather than representing the class labels as unique integers, we can represent each label as a binary vector using the `to_categorical()` function in Keras as a pre-processing step. In this case, each label is converted to a binary vector where the length of the vector is equal to the number of classes. All entries are set to zero except for the element that corresponds to the integer label.

```python
y_train_onehot = to_categorical(y_train_fashion[0:9])
print(y_train_onehot)
```

```Output
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
```

```python
# Convert integer labels to one-hot encoded vectors.
y_train = to_categorical(y_train)
y_valid = to_categorical(y_valid)
y_test  = to_categorical(y_test)
```

## 3 Model Architecture

### 3.1 Deep Neural Network Architecture

The network architecture shown below has multiple layers. An input layer, two hidden layers, and an output layer. There are several things to note about this architecture.

1. **Input Data**: The image input data is pre-processed (flattened) from a 2-Dimensional array `[28x28]` to 1-Dimensional vector of length `[784x1]` where the elements in this input vector are the normalized pixel intensities. The input to the network is sometimes referred to as the input "layer", but it's not technically a layer in the network because there are no trainable parameters associated with it.
2. **Hidden Layers**: We have two hidden layers that contain some number of neurons (that we need to specify). Each of the neurons in these layers has a non-linear activation function (e.g., ReLU, Sigmoid, etc...).
3. **Output Layer**: We now have ten neurons in the output layer to represent the ten different classes (digits: 0 to 9), instead of a single neuron as in the regression example.
4. **Dense Layers**: All the layers in the network are fully connected, meaning that each neuron in a given layer is fully connected (or dense) to each of the neurons in the previous layer. The **weights** associated with each layer are represented in bold to indicate that these are matrices that contain each of the weights for all the connections between adjacent layers in the network.
5. **Softmax Function**: The values from each of the neurons in the output layer are passed through a **softmax** function to produce a probability score for each of the ten digits in the dataset.
    
6. **Network Output**: The network output (y′), is a vector of length ten, that contains the probabilities of each output neuron. Predicting the class label simply requires passing (y′) through the `argmax` function to determine the index of the predicted label.
    
7. **Loss Function**: The loss function used is **Cross Entropy Loss**, which is generally the preferred loss function for classification problems. It is computed from the ground truth labels (y) and the output probabilities of the network (y′). Note that y and y′ are both vectors whose length is equal to the number of classes.

Backpropagation is used to compute the gradient of the loss with respect to the weights in the network. An optimizer (which implements gradient descent) is used to update the weights in the neural network


![](Pasted%20image%2020250502124217.png)

----------

#### activation function

An **activation function** in machine learning—especially in neural networks—is a mathematical function applied to the **output of each neuron** (or node) to introduce **non-linearity** into the model.

###### 🔍 Why is it needed?

Without activation functions, a neural network would just be a **linear model**, no matter how many layers it has. This means it could only learn linear relationships, which are not powerful enough for most real-world problems.

###### 🧠 What does it do?

- Takes the **weighted sum** of inputs into a neuron and transforms it.
    
- Helps the network learn **complex patterns** by introducing **non-linear decision boundaries**.
    

###### ✅ Common Activation Functions:

|Activation|Formula|Use Case & Notes|
|---|---|---|
|**ReLU**|`f(x) = max(0, x)`|Most common for hidden layers; fast and simple|
|**Sigmoid**|`f(x) = 1 / (1 + e^-x)`|Used in binary classification (output layer)|
|**Tanh**|`f(x) = (e^x - e^-x)/(e^x + e^-x)`|Outputs in [-1, 1]; zero-centered|
|**Softmax**|`f(xᵢ) = e^xᵢ / sum(e^xⱼ)`|Multi-class classification (output layer)|

###### 📌 In Summary:

An **activation function** controls the output of each neuron and allows the model to learn **non-linear relationships**, making neural networks powerful for tasks like image recognition, language processing, and more.

--------
##### Cross Entropy Loss

Great! If you're using **Cross Entropy Loss**, that typically means you're working on a **classification problem**, and here's how it works:



###### 🧠 **What is Cross Entropy Loss?**


$$
{Loss} = - \sum y_i \cdot \log(\hat{y}_i)
$$

Where:

- yi= true label (usually one-hot encoded)
- y^i = predicted probability for class ii

###### ✅ **When to use it:**

| Problem Type                              | Use this Loss Function          | With this Output Activation |
| ----------------------------------------- | ------------------------------- | --------------------------- |
| **Binary Classification**                 | `BinaryCrossentropy`            | Sigmoid                     |
| **Multi-Class Classification**            | `CategoricalCrossentropy`       | Softmax                     |
| **Multi-Class (with labels as integers)** | `SparseCategoricalCrossentropy` | Softmax                     |

###### 🔍 **Why it's effective:**

- Encourages the model to predict **high probability for the correct class**.
- Penalizes **confident wrong predictions** more heavily.
###### 🧪 Example in TensorFlow:

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',   # or 'sparse_categorical_crossentropy'
    metrics=['accuracy']
)
```

###### 🔸 1. `categorical_crossentropy`
- **Labels must be One-Hot encoded**
    - Example: If you have 3 classes and the true class is class 1 (index starts at 0):
        `y_true = [0, 1, 0]  # One-hot encoded for class 1`
- Used when you manually or automatically one-hot encode your target labels.

###### 🔸 2. `sparse_categorical_crossentropy`
- **Labels must be integers (class indices)**
    - Same example:
        `y_true = 1  # Integer index for class 1`
- You do **not** need to convert the labels to one-hot format — TensorFlow handles it internally.

---

## 4 Model Implementation


- Here we use Keras to define the model architecture, which has two dense layers (each with 128 neurons) and a single output layer with 10 neurons. Each neuron in the output layer corresponds to a class label from the dataset (0 through 9) where the output of each represents the probability that the input image corresponds to the class associated with that neuron. For example, if the output from the 5th neuron is .87, then that means the probability that the input image is a 4 is 87% (since the first class is 0, the 5th neuron represents the digit `4`).

- Notice that the first hidden layer has an input shape of [784,1] since the 28x28 image is flattened to a vector of length 784. The neurons in each of the hidden layers have activation functions called "ReLU" which stands for Rectified Linear Unit. The neurons in the output layer are then passed through a "softmax" function which transforms (normalizes) the raw output, which can be interpreted as a probability as described above.

### 4.1 Define the Model

```Python
# Instantiate the model.
model = tf.keras.Sequential()

# Build the model.
model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10,  activation="softmax"))

# Display the model summary.
model.summary()
```

### 4.2 Compile the Model

This step defines the optimizer and the loss function that will be used in the training loop. This is also where we can specify any additional metrics to track.

**Optimizer**: Here, we will use the RMSProp optimizer in Keras.

**Loss Function**: As mentioned above, the preferred loss function for classification problems is **Cross Entropy**. But depending on how the labels are encoded we’ll need to specify the proper form of the cross entropy loss function. If the labels are one-hot encoded, then you should specify the loss function as `categorical_crossentropy`, and if the labels are integer encoded, then you should use `sparse_categorical_crossentropy`. When performing binary classification, you should use `binary_crossentropy` as the loss function. Since we are using one-hot encoding in this example, we will specify the loss function as `categorical_crossentropy`.

**Metrics**: Finally, we also specify `accuracy` as an additional metric to record during training so that we can plot it after training is completed. The training loss and validation loss are automatically recorded, so there is no need to specify those.

```python
model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
```

----
### 4.3 Train the Model

```python
training_results = model.fit(X_train, 
                             y_train, 
                             epochs=21, 
                             batch_size=64, 
                             validation_data=(X_valid, y_valid));
```

----

### 4.4 Plot the Training Results

The function below is a convenience function to plot training and validation losses and training and validation accuracies. It has a single required argument which is a list of metrics to plot. 

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
    plt.xlim([0, 20])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)   
    plt.show()
    plt.close()
```

The loss and accuracy metrics can be accessed from the `history` object returned from the fit method. We access the metrics using predefined dictionary keys as shown below.


```python
# Retrieve training results.
train_loss = training_results.history["loss"]
train_acc  = training_results.history["accuracy"]
valid_loss = training_results.history["val_loss"]
valid_acc  = training_results.history["val_accuracy"]

plot_results(
    [train_loss, valid_loss],
    ylabel="Loss",
    ylim=[0.0, 0.5],
    metric_name=["Training Loss", "Validation Loss"],
    color=["g", "b"],
)

plot_results(
    [train_acc, valid_acc],
    ylabel="Accuracy",
    ylim=[0.9, 1.0],
    metric_name=["Training Accuracy", "Validation Accuracy"],
    color=["g", "b"],
)
```

-----

## 5 Model Evaluation

We can now predict the results for all the test images, as shown in the code below. Here, we call the `predict()` method to retrieve all the predictions, and then we select a specific index from the test set and print out the predicted scores for each class. You can experiment with the code below by setting the test index to various values and see how the highest score is usually associated with the correct value indicated by the ground truth.

```python
predictions = model.predict(X_test)
index = 0  # up to 9999
print("Ground truth for test digit: ", y_test[index])
print("\n")
print("Predictions for each class:\n")
for i in range(10):
    print("digit:", i, " probability: ", predictions[index][i])
```


```output
313/313 [==============================] - 1s 1ms/step
Ground truth for test digit:  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]


Predictions for each class:

digit: 0  probability:  9.819607e-24
digit: 1  probability:  2.4064698e-18
digit: 2  probability:  1.4520596e-13
digit: 3  probability:  2.4951994e-13
digit: 4  probability:  1.5394617e-26
digit: 5  probability:  9.713211e-23
digit: 6  probability:  4.6183826e-30
digit: 7  probability:  1.0
digit: 8  probability:  1.8647681e-26
digit: 9  probability:  1.4221963e-17

```

----

### 5.2 Confusion Matrix

A confusion matrix is a very common metric that is used to summarize the results of a classification problem. 

The information is presented in the form of a table or matrix where one axis represents the ground truth labels for each class, and the other axis represents the predicted labels from the network. 

The entries in the table represent the number of instances from an experiment (which are sometimes represented as percentages rather than counts). 

Generating a confusion matrix in TensorFlow is accomplished by calling the function `tf.math.confusion_matrix()`, which takes two required arguments which are the list of ground truth labels and the associated predicted labels.

```python
# Generate predictions for the test dataset.
predictions = model.predict(X_test)

# For each sample image in the test dataset, select the class label with the highest probability.
predicted_labels = [np.argmax(i) for i in predictions]
```

```python
# Convert one-hot encoded labels to integers.
y_test_integer_labels = tf.argmax(y_test, axis=1)

# Generate a confusion matrix for the test dataset.
cm = tf.math.confusion_matrix(labels=y_test_integer_labels, predictions=predicted_labels)

# Plot the confusion matrix as a heatmap.
plt.figure(figsize=[15, 8])
import seaborn as sn

sn.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 14})
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()
```

![](Pasted%20image%2020250502131132.png)


###### 🔍 How to Read It:

- **Rows** = Actual (true) labels
- **Columns** = Predicted labels
- Each cell [i,j] = Number of times **class i** was **predicted as class j**.

The diagonal cells (top-left to bottom-right) show the number of **correct predictions** for each class.
###### ✅ Insights:

1. **High Accuracy**:

    - Most predictions lie on the diagonal, meaning the model is classifying digits correctly most of the time.
    - E.g., digit **1** was predicted correctly **1,121** times.
2. **Misclassifications**:
    - Off-diagonal values represent errors.
    - E.g., actual digit **5** was misclassified as:
        - **3** → 12 times
        - **8** → 3 times
        - **6** → 3 times
3. **Classes with More Errors**:
    - **Digit 5**: Fewer correct predictions (866), relatively higher misclassifications.
    - **Digit 9**: Some confusion with **4**, **3**, and **7** (notably 17 predicted as 8).

###### 📊 Metrics You Can Derive:

From this matrix, you can calculate:

- **Accuracy** = (Sum of diagonal) / (Sum of all values)
- **Precision / Recall / F1-score** for each class
- **Class imbalance** (if some rows have much lower total counts)
