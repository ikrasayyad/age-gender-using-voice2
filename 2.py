# 2.py
from tensorflow.keras.models import Sequential  # Ensure this import works
from tensorflow.keras.layers import Dense  # Ensure this import works
from dataset import load_data  # Correct the import based on your structure

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))  # Example input_dim needs to be defined
    model.add(Dense(10, activation='softmax'))  # Example output layer
    return model

if __name__ == "__main__":
    load_data()  # Load data from dataset.py
    model = build_model()
    print("Model built successfully.")
