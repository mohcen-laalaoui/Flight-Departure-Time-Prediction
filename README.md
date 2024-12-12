# Flight Departure Time Prediction

This project aims to predict flight departure time categories using machine learning. The model classifies flights into categories based on features such as the airline, departure and arrival locations, and flight delays. The model leverages deep learning techniques and is built with PyTorch.

## Tools and Libraries Used

- **Python**: Programming language used for data processing, model building, and evaluation.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: Used for numerical operations and handling arrays.
- **Scikit-learn**:
  - **OneHotEncoder**: For encoding categorical features into numerical format.
  - **LabelEncoder**: For encoding the target variable into numerical labels.
  - **train_test_split**: For splitting the dataset into training and testing sets.
  - **ConfusionMatrix, ClassificationReport, and ROC Curve**: For evaluating the model’s performance.
- **Matplotlib & Seaborn**: For data visualization (loss, accuracy, confusion matrix, ROC curve).
- **PyTorch**: Deep learning framework used for model building, training, and evaluation.

## Problem Overview

The goal of this project is to predict the **departure time category** of a flight based on various input features. These features include:
- **Airline**: The airline operating the flight.
- **Departure Location**: The city from which the flight departs.
- **Arrival Location**: The city where the flight arrives.
- **Flight Delay**: Delay information for the flight.

This is a **multi-class classification problem**, where the task is to assign each flight to a specific departure time category (target variable).

## Dataset

The dataset contains information about different flights, including:
- Airline name
- Departure and arrival locations
- Flight delays
- Departure time (target variable)

The data is processed, cleaned, and encoded before being used to train a deep learning model.

## Workflow

### 1. Data Preprocessing

The following preprocessing steps were applied:
- **Handling Missing Data**: Missing values are dropped from the dataset.
- **Encoding Categorical Variables**: `OneHotEncoder` is used to encode categorical features like the airline, departure, and arrival locations into numerical format.
- **Label Encoding**: The target variable (departure time) is encoded using `LabelEncoder` for classification tasks.

### 2. Model Architecture

The deep learning model is built using **PyTorch**. The architecture includes:
- **Input Layer**: A layer that receives the input features (encoded variables).
- **Hidden Layers**: Two fully connected layers with ReLU activations to introduce non-linearity.
- **Output Layer**: A fully connected layer with a softmax activation function to output class probabilities for multi-class classification.

### 3. Training

The model is trained using:
- **Cross-Entropy Loss**: For multi-class classification.
- **Adam Optimizer**: For optimizing the model parameters.

The training process includes plotting training loss and accuracy curves over the epochs to monitor the model's learning progress.

### 4. Evaluation

After training, the model is evaluated on the test set:
- **Accuracy**: The overall classification accuracy of the model.
- **Confusion Matrix**: To visualize the performance across all classes.
- **Classification Report**: Precision, recall, and F1-score metrics for each class.
- **ROC Curve**: To assess the model's ability to distinguish between classes, with the corresponding AUC score.

### 5. Results

The model's evaluation is visualized with:
- **Training Loss and Accuracy Curves**
- **Confusion Matrix**
- **ROC Curve and AUC**

## Usage

To run the project locally, follow these steps:

### Requirements

- Python 3.x
- Install the necessary libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn torch
