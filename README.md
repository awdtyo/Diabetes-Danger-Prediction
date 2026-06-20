# Diabetes Prediction with Deep Learning

## Project Overview
This project implements a deep learning model using TensorFlow/Keras to predict the onset of diabetes based on diagnostic measurements. The dataset used is the Pima Indians Diabetes Database, sourced from the UCI Machine Learning Repository.

## Dataset
The `diabetes.csv` dataset contains various medical predictor variables and one target variable, `Outcome` (0 for non-diabetic, 1 for diabetic).

**Features (X)**:
*   `Pregnancies`: Number of times pregnant
*   `Glucose`: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
*   `BloodPressure`: Diastolic blood pressure (mm Hg)
*   `SkinThickness`: Triceps skin fold thickness (mm)
*   `Insulin`: 2-Hour serum insulin (mu U/ml)
*   `BMI`: Body mass index (weight in kg/(height in m)^2)
*   `DiabetesPedigreeFunction`: Diabetes pedigree function
*   `Age`: Age (years)

**Target (y)**:
*   `Outcome`: Class variable (0 or 1)

## Methodology
1.  **Data Loading**: The dataset is loaded using Pandas.
2.  **Data Splitting**: The dataset is split into features (`X`) and target (`y`).
3.  **Train-Test Split**: The data is further divided into training and testing sets with a `test_size` of 20% and `random_state` for reproducibility.
4.  **Feature Scaling**: `StandardScaler` from `sklearn.preprocessing` is applied to normalize the features, which is crucial for neural networks to converge efficiently.
5.  **Model Architecture**: A sequential Keras model is constructed with three `Dense` layers:
    *   Input Layer: 64 neurons, ReLU activation, `input_shape` matching the number of features.
    *   Hidden Layer: 32 neurons, ReLU activation.
    *   Output Layer: 1 neuron, Sigmoid activation (for binary classification).
6.  **Model Compilation**: The model is compiled with the `Adam` optimizer, `binary_crossentropy` loss function, and `accuracy` as the evaluation metric.
7.  **Model Training**: The model is trained for 50 epochs, with validation performed on the held-out test set.
8.  **Model Evaluation**: The trained model's performance is evaluated on the test set.
9.  **Loss Visualization**: Training and validation loss over epochs are plotted to monitor for overfitting or underfitting.
10. **Model Saving**: The trained model is saved in HDF5 format (`model.h5`).

## Model Architecture
```python
model = Sequential([
    Dense(64, activation = 'relu', input_shape = (X_train.shape[1],)),
    Dense(32, activation = 'relu'),
    Dense(1, activation = 'sigmoid')
])
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
