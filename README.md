# EEG Biometric Authentication
This Python package provides tools for biometric authentication of EEG (Electroencephalography) data. It includes functionalities for data loading, model creation (specifically a Convolutional Neural Network), training, evaluation, and visualization.
## Features

- **Data Loading:** Load EEG data from CSV files.
- **Model Creation:** Define a Convolutional Neural Network (CNN) model for EEG data classification.
- **Training and Evaluation:** Train the CNN model and evaluate its performance.
- **Prediction:** Make predictions on new EEG data.
- **Visualization:** Visualize evaluation metrics such as accuracy and confusion matrix.

## Installation
You can install the EEG Biometric Authentication using pip:
```bash
pip install eeg_biometric_authentication_package
Usage
Here's a basic example of how to use the package:

python
Copy code
from eeg_analysis_package.data.loader import load_data
from eeg_analysis_package.models.cnn_model import create_cnn_model
from eeg_analysis_package.train_eval.trainer_evaluator import (
    train_model,
    evaluate_model,
    predict_labels,
    calculate_subject_accuracies,
    generate_confusion_matrix
)
from eeg_analysis_package.utils.visualization import plot_confusion_matrix

# Load data, create model, train model, evaluate model, etc. (as shown in the example script
