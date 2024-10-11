# Heart Disease Prediction and Explainability Using Deep Learning

This project aims to predict heart disease using a neural network and provide insights into the model's decisions through feature attributions. We utilize the Captum library's DeepLift method to explain how various health features contribute to the model's predictions.

## Overview

Heart disease is a significant health concern worldwide. Early diagnosis and intervention can drastically improve patient outcomes. In this project, we develop a neural network model to classify patients into three categories of heart disease based on their health features. To enhance model interpretability, we employ Captum's DeepLift method to analyze the importance of each feature in the prediction process.

### Objectives

1. **Prediction**: Train a neural network to predict the presence and type of heart disease using health-related features.
2. **Explainability**: Use the DeepLift method from Captum to understand which features are most influential in the model's predictions for each class.
3. **Visualization**: Present the attributions through visualizations, helping stakeholders interpret the model's decisions.

## Dataset

The dataset used in this project is the **Heart Disease dataset**, which includes the following features:

- **age**: Age of the patient.
- **sex**: Gender of the patient (1 = male; 0 = female).
- **cp**: Chest pain type (0-3).
- **trestbps**: Resting blood pressure (in mm Hg).
- **chol**: Serum cholesterol in mg/dl.
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).
- **restecg**: Resting electrocardiographic results (0, 1, 2).
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise induced angina (1 = yes; 0 = no).
- **oldpeak**: ST depression induced by exercise relative to rest.
- **slope**: The slope of the peak exercise ST segment.
- **ca**: Number of major vessels (0-3) colored by fluoroscopy.
- **thal**: Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect).
- **target**: Class label (0, 1, 2) indicating the presence of heart disease.

## Neural Network Architecture

The neural network architecture consists of:

- **Input Layer**: Accepts features from the dataset.
- **Hidden Layers**: Two hidden layers with ReLU activation, containing 64 and 32 neurons, respectively.
- **Output Layer**: Produces three output values representing the classes of heart disease.

## Explainability with Captum's DeepLift

To gain insights into the model's predictions, we use Captum's **DeepLift** method. DeepLift computes feature attributions by comparing the activation of neurons for a given input against a reference (baseline) input. The attributions indicate how much each feature contributes positively or negatively to the final prediction.

### Key Steps

1. **Model Training**: The neural network is trained on the heart disease dataset.
2. **Attribution Calculation**: For each sample in the test dataset, we compute feature attributions using DeepLift, comparing the sample against a baseline input (e.g., zeroed feature values).
3. **Visualization**: The attributions for each class are averaged and plotted to visualize the importance of each feature.

## Requirements

To run the code, install the necessary packages. You can use the `requirements.txt` file to install them easily:

```bash
pip install -r requirements.txt
```

Run the Script: Execute the script to train the neural network and calculate feature attributions:
```bash
python heart_disease_captum.py
```
View Results: The results will include the loss during training and a bar chart showing the normalized average attribution of each feature for all target classes.
