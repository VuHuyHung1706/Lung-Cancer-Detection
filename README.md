# Lung Cancer Detection Project

## Overview
This repository contains two approaches for lung cancer detection:
1. **Traditional Machine Learning (ML)**: Using survey data with patient symptoms and risk factors
2. **Convolutional Neural Networks (CNN)**: Using chest X-ray images to detect pneumonia as a potential indicator of lung cancer risk

## Repository Structure


## Machine Learning Component
The `lung-cancer-detection.ipynb` notebook implements several machine learning models to predict lung cancer based on patient survey data:

- **Data**: Patient survey information with 15 features including:
  - Demographics: Gender, Age
  - Risk factors: Smoking, Alcohol consumption
  - Symptoms: Coughing, Chest pain, etc.
- **Models**:
  - Support Vector Machines (SVM)
  - Random Forest Classifier
  - Decision Tree Classifier
- **Feature Engineering**: SMOTE for handling class imbalance
- **Performance**: The Random Forest model achieved the highest accuracy (96.9%) on the test data

## Deep Learning Component
The CNN model analyzes chest X-ray images to detect pneumonia:

- **Model Architecture**: Custom CNN with:
  - Initial layers: 7×7 kernels with 3×3 max pooling
  - Middle layers: 5×5 kernels
  - Deeper layers: Multiple 3×3 convolutional blocks
  - Dense layers with dropout for regularization
- **Data Processing**:
  - Image resizing to 196×196 pixels
  - Grayscale conversion and normalization
  - Data augmentation (rotation, zoom, shifting)
- **Performance**: The model achieved over 90% accuracy on test data

## Key Findings
- Traditional ML models provide strong predictive performance based on patient survey data
- The CNN approach demonstrated high accuracy in identifying pneumonia from chest X-rays
- Combining both approaches could provide a comprehensive lung cancer risk assessment system

## Requirements
- Python 3.6+
- TensorFlow 2.x
- scikit-learn
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Seaborn
- imbalanced-learn

## Usage
The notebooks are self-contained and include all the necessary code to:
1. Load and preprocess the data
2. Train the models
3. Evaluate performance
4. Make predictions on new data

## Note
This repository contains only the code files. The training image datasets are not included due to their large size.
