# Import necessary modules
from data.loader import load_data
from model.cnn_model import create_cnn_model
from train_eval.trainer_evaluator import (
    train_model,
    evaluate_model,
    predict_labels,
    calculate_subject_accuracies,
    generate_confusion_matrix
)
from utils.visualization import plot_confusion_matrix

# Define parameters
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
train_sessions = [1, 2, 3]
test_session = 4
max_time_steps = 23000

# Load data
X_train, y_train, X_test, y_test = load_data(subjects, train_sessions, test_session, max_time_steps)

# Adjust labels
y_train -= 1
y_test -= 1

# Create model
model = create_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=len(subjects))

# Train model
train_model(model, X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model
test_loss, test_acc = evaluate_model(model, X_test, y_test)
print('Test accuracy:', test_acc)

# Predict labels
y_pred = predict_labels(model, X_test)

# Calculate accuracies
accuracies = calculate_subject_accuracies(y_test, y_pred)

# Print accuracies
for i, subject in enumerate(subjects):
    print(f"Subject {subject}: {accuracies[i]*100:.2f}% accuracy")

# Generate confusion matrix
cm = generate_confusion_matrix(y_test, y_pred, subjects)

# Plot confusion matrix
plot_confusion_matrix(cm, subjects)
