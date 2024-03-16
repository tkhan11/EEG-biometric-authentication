import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, subjects):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=subjects, yticklabels=subjects)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
