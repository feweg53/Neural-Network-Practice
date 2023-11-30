import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import max_norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Example data loading (replace with data source)
# Function to load data from an Excel file
def load_data(excel_path):
    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Extract features and labels
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # The last column

    return X, y

# Path to the Excel file (update with the actual path)
excel_path = 'Cats_Dogs_Dataset.xlsx'

# Load features and labels
X, y = load_data(excel_path)

# For demonstration, let's create some synthetic data
np.random.seed(0)  # For reproducible results
X = np.random.rand(1000, 5)  # 1000 samples, 5 features each
y = np.random.randint(0, 2, 1000)  # Random binary labels (0 or 1)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Neural network structure
model = Sequential([
    Dense(5, input_shape=(5,), activation='relu'),
    Dense(10, activation='relu', kernel_constraint=max_norm(3)),
    Dense(10, activation='relu', kernel_constraint=max_norm(3)),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

'''
Predictions start
'''

# Function for preprocessing new data (similar to training data)
def preprocess_new_data(new_data, scaler):
    # Assuming new_data is a NumPy array with the same number of features as training data
    # Normalize the data using the same scaler used for training data
    new_data_normalized = scaler.transform(new_data)
    return new_data_normalized

# Example of preparing and predicting new data
# Replace this with actual new data
new_data_samples = np.array([
    [5, 30, 15, 5, 0, 1, 0, 0, 0],  # Sample new data point 1
    [8, 45, 20, 7, 0, 0, 1, 0, 0]   # Sample new data point 2
])

# Preprocess the new data
new_data_preprocessed = preprocess_new_data(new_data_samples, scaler)

# Make predictions
predictions = model.predict(new_data_preprocessed)

# Process and print predictions
# In this case, predictions are probabilities of being a dog (1)
# m=Might want to convert these to class labels (0 or 1)
predicted_classes = (predictions > 0.5).astype(int)
print("Predictions (Probabilities):", predictions)
print("Predicted Classes (0 for Cat, 1 for Dog):", predicted_classes)
