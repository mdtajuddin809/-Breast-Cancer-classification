# 🧠 Breast Cancer Classification using ANN

This project focuses on building an Artificial Neural Network (ANN) model to classify breast cancer tumors as **Malignant** or **Benign**.  
A Streamlit web application is also developed to allow users to make real-time predictions.

---

## 📌 1. Problem Statement

Breast cancer is one of the most common cancers worldwide. Early detection is critical for effective treatment.

The objective of this project is to build a machine learning model that can predict whether a tumor is:
- **Malignant (Cancerous)**
- **Benign (Non-cancerous)**

based on medical diagnostic features.

---

## 📊 2. Dataset Overview

The dataset used in this project is the **Breast Cancer Wisconsin Dataset**.

It contains:
- 569 rows (samples)
- 30 numerical features (medical measurements)
- 1 target column (`diagnosis`)

### Target Variable:
- M → Malignant
- B → Benign

The features include measurements such as:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Symmetry
- Fractal dimension
(Each measured as mean, standard error, and worst value)

---

## 🔄 3. Data Preprocessing

### ✅ Dropping Unnecessary Columns
- `id` column removed (not useful for prediction)
- `Unnamed: 32` removed (empty column)

### ✅ Target Encoding
The target column was converted into numeric format:
- M → 1
- B → 0

### ✅ Train-Test Split
- 80% data used for training
- 20% data used for testing
- `random_state=42` used for reproducibility

### ✅ Feature Scaling
Feature scaling was applied using **StandardScaler** to normalize the data.

Why?
Neural networks perform better when input features are scaled.

---

## 🧠 4. Model Building (ANN Architecture)

The Artificial Neural Network was built using TensorFlow/Keras.

### Architecture:

- Input Layer → 30 neurons (one for each feature)
- Hidden Layer 1 → 16 neurons (ReLU activation)
- Hidden Layer 2 → 8 neurons (ReLU activation)
- Output Layer → 1 neuron (Sigmoid activation)

### Activation Functions:
- ReLU → For hidden layers
- Sigmoid → For binary classification output

### Loss Function:
- `binary_crossentropy`

### Optimizer:
- `Adam`

---

## 🚀 5. Model Training

The model was trained with:

- Epochs: 100
- Batch size: 32
- Validation data: Test set

### EarlyStopping:
EarlyStopping callback was used to:
- Monitor validation loss
- Prevent overfitting
- Stop training when performance stops improving

---

## 📈 6. Model Evaluation

After training:

- Predictions were made on test data
- Probabilities were converted into 0 or 1 using threshold 0.5
- Accuracy score was calculated

The model achieved high classification accuracy on test data.

---

## 💾 7. Model Saving

To use the model in the Streamlit application:

- Trained ANN model saved as:
