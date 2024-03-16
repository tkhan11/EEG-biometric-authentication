import numpy as np
import pandas as pd

def load_data(subjects, train_sessions, test_session, max_time_steps):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for subject in subjects:
        for session in train_sessions:
            file_path = f"EEG_dataset/s{subject}_s{session}.csv"  # Update the file path accordingly
            df = pd.read_csv(file_path)
            data = df[['T7', 'F8', 'Cz', 'P4']].values
            if data.shape[0] < max_time_steps:  # Pad if too short
                pad_width = ((0, max_time_steps - data.shape[0]), (0, 0))
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
            elif data.shape[0] > max_time_steps:  # Truncate if too long
                data = data[:max_time_steps, :]
            X_train.append(data)
            y_train.append(subject)  # Assign unique label for each subject
        
        # Load test data
        if test_session is not None:
            file_path = f"EEG_dataset/s{subject}_s{test_session}.csv"  # Update the file path accordingly
            df = pd.read_csv(file_path)
            data = df[['T7', 'F8', 'Cz', 'P4']].values
            if data.shape[0] < max_time_steps:  # Pad if too short
                pad_width = ((0, max_time_steps - data.shape[0]), (0, 0))
                data = np.pad(data, pad_width, mode='constant', constant_values=0)
            elif data.shape[0] > max_time_steps:  # Truncate if too long
                data = data[:max_time_steps, :]
            X_test.append(data)
            y_test.append(subject)  # Assign unique label for each subject
    
    # Shuffle training data
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = np.array(X_train)[indices]
    y_train = np.array(y_train)[indices]

    # Shuffle test data
    indices = np.arange(len(X_test))
    np.random.shuffle(indices)
    X_test = np.array(X_test)[indices]
    y_test = np.array(y_test)[indices]

    return X_train, y_train, X_test, y_test
