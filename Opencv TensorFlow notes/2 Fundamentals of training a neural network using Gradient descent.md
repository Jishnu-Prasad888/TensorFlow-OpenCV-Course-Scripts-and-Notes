
### Labeled Training Data and One-Hot Encoding

1) Labeled training data consists of images and their corresponding ground truth (categorical) labels. If a network is designed to classify objects from three classes (e.g., Cats, Dogs, Other), we will need training samples from all three classes. Typically thousands of samples from each class are required.
2) Datasets that contain categorical labels may represent the labels internally as strings ("Cat", "Dog, "Other") or as integers (0,1,2). However, prior to processing the dataset through a neural network, the labels must have a numerical representation. When the dataset contains integer labels (e.g., 0, 1, 2) to represent the classes, a class label file is provided that defines the mapping from class names to their integer representations in the dataset. This allows the integers to be mapped back to class names when needed.This type of label encoding is called **Integer Encoding** because unique integers are used to encode the class labels.
3)  when the class labels have no relationship to one another, it is recommended that **One-Hot Encoding** be used instead. One-hot encoding is a technique that represents categorical labels as binary vectors (containing only zeros and ones). In this example, we have three different classes (Cat, Dog, and Other), so we can represent each of the classes numerically with a vector of length three where one of the entries is a one, and the others are all zeros.The particular order is arbitrary, but it needs to be consistent throughout the dataset.

**Inputs**

| Cat | Dog | Other |
| --- | --- | ----- |
| 1   | 0   | 0     |
| 0   | 1   | 0     |
| 0   | 0   | 1     |
**Prediction as vector**

| 0.37 |
| ---- |
| 0.50 |
| 0.13 |

4) All neural networks use a loss function that quantifies the error between the predicted output and the ground truth for a given training sample.
5) One way to quantify the error between the network output and the expected result is to compute the Sum of Squared Errors (SSE), as shown below. This is also referred to as a loss. In the example below, we compute the error for a single training sample by computing the difference between the elements of the ground truth vector and the corresponding elements of the predicted output. Each term is then squared, and the total sum of all three represents the total error, which in this case, is `0.6638`.
![](Pasted%20image%2020250325015636.png)

6) When neural networks are trained in practice, many images are used to compute a loss before the network weights are updated. Therefore, the next equation is often used to compute the Mean Squared Error (MSE) for a number of training images. The MSE is just the mean of the SSE for all the images that were used. The number of images used to update the weights is referred to as the **batch size** (a batch size of 32 is typically a good default). The processing of a batch of images is referred to as an "iteration".
![](Pasted%20image%2020250325015715.png)

### Gradient Descent (Optimization)

1)  there is a principled way to tune the weights of a neural network called **gradient descent.** For simplicity, we’re going to illustrate the concept with just a single tunable parameter called WW, and we’re going to assume the loss function is convex and therefore shaped like a bowl, as shown in the figure.
![](Pasted%20image%2020250325121555.png)

2) The value of the loss function is shown on the vertical axis, and the value of our single trainable weight is shown on the horizontal axis. Let’s assume the current estimate of the weight is We1
3) Referring to the plot on the left, If we compute the slope of the loss function at the point corresponding to the current weight estimate, We1, we can see that the slope (gradient) is negative. In this situation, we would need to increase the weight to get closer to the optimum value indicated by Wo. So we would need to move in a direction opposite from the sign of the gradient.
4) On the other hand, if our current weight estimate, We1>Wo (as shown in the plot to the right), the gradient would be positive, and we would need to reduce the value of the current weight to get closer to the optimum value of Wo. Notice that in both cases, we still need to move in a direction opposite from the sign of the gradient.
5) Remember that the slope of a line is defined as the rise over the run and that when the weight is to the left of the optimum value, the slope of the function is negative, and when the weight is to the right of the optimum value, the slope of the function is positive. So it’s the sign of the gradient that’s important.
![](Pasted%20image%2020250325121934.png)

6) The learning rate is something that we need to specify prior to training and is not something that is learned by the network. Parameters like this are often called hyperparameters to distinguish them from trainable parameters (such as the network weights).
7) In practice, the loss function has many dimensions and is not typically convex but has many peaks and valleys. In the general case, the slope of the loss function is called the gradient and is a function of all the weights in the network. But the approach used to update the weights is conceptually the same as described here.


### Weight Update Sample Calculation

- One thing we haven’t talked about yet is how you actually compute the gradient of the loss function with respect to the weights in the network. Fortunately, this is handled by an algorithm called **backpropagation,** which is built into deep learning frameworks, such as TensorFlow, Keras, and PyTorch, so it’s not something you need to implement yourself.

### The Complete Training Loop

![](Pasted%20image%2020250325125240.png)

### Training plots

1) training a neural network is an iterative process that typically requires passing the entire training set through the Network multiple times.
2) Each time the entire training dataset is passed through the network, we refer to that as a **training epoch.** Training neural networks often require many training epochs until the point where the loss stops decreasing with additional training. As you can see in the first plot below, the rate at which the loss decreases tapers off as training progresses, indicating that the model is approaching its capacity to learn.
![](Pasted%20image%2020250325125425.png)

3) One important topic that we have not yet covered is **_data splitting._** This involves the concept of a validation dataset used to evaluate the quality of the trained model during the training process. This is an important and fundamental topic that will be covered in a subsequent post.

###  Performing Inference using a Trained Model

1) Once we have a trained Network, we can supply it with images of unknown content and use the network to make a prediction as to what class the image belongs to. This is reflected in the diagram below, and notice that at this point, we do not require any labeled data. All we need are images with unknown content that we wish to classify. Making predictions on unknown data is often referred to as using the network to perform inference.
2) 