import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_path = './data.pickle'
try:
    with open(data_path, 'rb') as f:
        data_dict = pickle.load(f)
except FileNotFoundError:
    print(f"Error: The file '{data_path}' was not found.")
    exit()

# Process data
try:
    # Ensure all data samples have the same length
    fixed_length = max(len(item) for item in data_dict['data'])

    def pad_or_truncate(sequence, target_length):
        if len(sequence) > target_length:
            return sequence[:target_length]
        else:
            return sequence + [0] * (target_length - len(sequence))

    data = np.array([pad_or_truncate(item, fixed_length) for item in data_dict['data']])
    labels = np.asarray(data_dict['labels'])
except Exception as e:
    print(f"Error processing data: {e}")
    exit()

# Split the data into training and testing sets
try:
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
except Exception as e:
    print(f"Error splitting data: {e}")
    exit()

# Train the model
try:
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
except Exception as e:
    print(f"Error training the model: {e}")
    exit()

# Evaluate the model
try:
    y_predict = model.predict(x_test)
    score = accuracy_score(y_predict, y_test)
    print(f'{score * 100:.2f}% of samples were classified correctly!')
except Exception as e:
    print(f"Error during evaluation: {e}")
    exit()

# Save the trained model
model_path = './model.p'
try:
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model}, f)
    print(f"Model saved successfully to '{model_path}'.")
except Exception as e:
    print(f"Error saving the model: {e}")
