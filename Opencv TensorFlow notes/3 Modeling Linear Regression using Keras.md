
 Since linear regression can be modeled as a neural network, it provides an excellent example to introduce the essential components of neural networks. Regression is a form of supervised learning which aims to model the relationship between one or more input variables (features) and a continuous (target) variable. We assume that the relationship between the input variables xx and the target variable yy can be expressed as a weighted sum of the inputs (i.e., the model is linear in the parameters). In short, linear regression aims to learn a function that maps one or more input features to a single numerical target value.


```python
SEED_VALUE = 42
```
- used to add a bit of randomness to the input to make the model better at recognizing images when a image similar to the training image but not the same on is given

### Load the California Housing Dataset

1) we will be working with the California Housing dataset. This dataset contains information collected by the U.S Census Service concerning housing in Boston MA. It has been used extensively throughout the literature to benchmark algorithms and is also suitable for demonstration purposes due to its small size. 
2) The dataset contains 14 unique attributes, among which is the median value (price in $K) of a home for a given suburb. We will use this dataset as an example of how to develop a model that allows us to predict the median price of a home based on a single attribute in the dataset (average number of rooms in a house).
3) Keras provides the `load_data()` function to load this dataset. Datasets are typically partitioned into `train`, and `test` components, and the `load_data()` function returns a **tuple** for each. Each tuple contains a 2-dimensional array of features (e.g., X_train) and a vector that contains the associated target values for each sample in the dataset (e.g., y_train). So, for example, the rows in `X_train` represent the various samples in the dataset and the columns represent the various features. In this notebook, we are only going to make use of the training data to demonstrate how to train a model. However, in practice, it is very important to use the test data to see how well the trained model performs on unseen data.

### Some other stuff to keep in mind

---------

```python
from sklearn.model_selection import train_test_split from sklearn.preprocessing import StandardScaler
```

**`from sklearn.model_selection import train_test_split`**
- This function is used to split your dataset into **training** and **testing** sets.
- It's essential for evaluating your model's performance.
- Typically, a dataset is divided into a larger training set (e.g., 80%) and a smaller testing set (e.g., 20%) using this function.
**Example:**
`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`
- `X` = Features
- `y` = Target labels
- `test_size=0.2` means 20% of the data is used for testing
- `random_state=42` ensures reproducibility


**`from sklearn.preprocessing import StandardScaler`**
- `StandardScaler` is used to **normalize** or **standardize** your data.
- It transforms the data so that it has a **mean of 0** and a **standard deviation of 1**.
- This is especially useful for algorithms that are sensitive to feature scaling, like **Logistic Regression** or **Support Vector Machines (SVM)**.
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
- - `.fit_transform()` is applied to the training data to compute the mean and standard deviation, then scale it.
- `.transform()` is applied to the test data using the same scaling parameters.

----------

 #### **StandardScaler (Normalization)**

- **What it does:** `StandardScaler` adjusts your data so that it has a **mean (average)** of **0** and a **standard deviation** of **1**.
- **Why it's useful:** Many machine learning algorithms perform better when the data is scaled. Algorithms like logistic regression, SVM, and k-means clustering are sensitive to the scale of data.

**How it works:**  
The formula used by `StandardScaler` is:

![](Pasted%20image%2020250326003948.png)

👉 After applying `StandardScaler`:

- The average (mean) of each feature becomes **0**
- The spread (standard deviation) becomes **1**

This makes the data **centered** and **standardized**, helping the model learn efficiently.

----------
##### **random_state=42 (Reproducibility)**

- **What it does:** `random_state=42` is a parameter used to control **randomness** in functions like `train_test_split`.
- **Why it's useful:** When you split your data or apply any algorithm with randomness, the results might change every time you run the code. Setting `random_state=42` ensures you get the **same split every time**, which makes your results **reproducible**.

**Why 42?**
- The number **42** is not special — it's just a popular choice. It became a common example in coding tutorials, inspired by _The Hitchhiker's Guide to the Galaxy_, where **42** is the "Answer to the Ultimate Question of Life, the Universe, and Everything."
- You can use any number (e.g., `random_state=7`, `random_state=123`). As long as you use the same number, you’ll get the same results.

---------

#### **`np.isnan(X_train).sum()`**

### **`np.isnan(X_train)`**

- This checks for **NaN** (**Not a Number**) values in the `X_train` array.
- It returns a **boolean array** of the same shape as `X_train`, where:
    - `True` means the value is **NaN**
    - `False` means the value is **not NaN**

```python
import numpy as np

X_train = np.array([[1, 2], [np.nan, 4], [5, np.nan]])
print(np.isnan(X_train))

''' 
output :
[[False False]
 [ True False]
 [False  True]]
'''
```

### **`.sum()`**

- After `np.isnan()` generates a boolean array, `.sum()` adds up all the `True` values, where `True` is treated as `1` and `False` as `0`.
- This gives the **total number of NaN values** in `X_train`.



-------

*Suppose we have a regression dataset containing 11 features. What is the total number of training parameters of the model in scenarios:*
*When we use 10 of the features with bias.*
 *When we use 5 features without bias.*

Total parameters=Number of features+Bias term (if included)

