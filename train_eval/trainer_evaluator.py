from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
def train_model(model, X_train, y_train, epochs, batch_size, validation_split):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history

def evaluate_model(model, X_test, y_test):
    test_loss, test_acc = model.evaluate(X_test, y_test)
    return test_loss, test_acc

def predict_labels(model, X_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    return y_pred

def calculate_subject_accuracies(y_test, y_pred):
    subjects = np.unique(y_test)
    accuracies = []
    for subject in subjects:
        mask = y_test == subject
        subject_true_labels = y_test[mask]
        subject_predicted_labels = y_pred[mask]
        subject_accuracy = accuracy_score(subject_true_labels, subject_predicted_labels)
        accuracies.append(subject_accuracy)
    return accuracies

def generate_confusion_matrix(y_test, y_pred, subjects):
    cm = confusion_matrix(y_test, y_pred)
    return cm
