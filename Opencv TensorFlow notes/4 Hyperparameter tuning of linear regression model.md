### **What is Hyperparameter Tuning?**

- **Hyperparameters** are settings that control the training process of a machine learning model, but they are not learned from the data.
- **Tuning** involves finding the optimal set of hyperparameters that minimizes the error and improves the model's performance.
### 📦 **Examples of Hyperparameters**

- Learning rate
- Batch size
- Number of epochs
- Number of neurons
- Activation functions
- Dropout rate
- Optimizer type
- Regularization strength (L1/L2)

---

## 🛠️ **Why is Hyperparameter Tuning Important?**

- A poor choice of hyperparameters can lead to **underfitting** or **overfitting**.
    
- Tuning helps improve **accuracy**, **reduce loss**, and optimize overall model performance.
    

---

## 🔎 **Methods for Hyperparameter Tuning**

### 1. **Grid Search**

- Tries **all possible combinations** of hyperparameters within a specified range.
- Computationally expensive but effective for smaller search spaces.
### 2. **Random Search**

- Selects hyperparameters **randomly** from the specified range.
- More efficient than grid search for large search spaces.

### 3. **Bayesian Optimization**

- Uses probability models to predict the best set of hyperparameters.
- More intelligent than random search as it learns from previous attempts.

### 4. **Hyperband**

- Efficiently allocates resources by evaluating multiple configurations quickly.
- Based on an adaptive resource allocation strategy.

