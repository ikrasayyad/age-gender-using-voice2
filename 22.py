import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = r'C:\Users\ikras\OneDrive\Desktop\V\household_income.csv'
print("Loading dataset...")
data = pd.read_csv(file_path)
print("Dataset loaded successfully!")

# Display the first few rows
print(data.head())

# Check for missing values
print("Checking for missing values...")
print(data.isnull().sum())

# Convert relevant columns to numeric
data['Estimate; Aggregate household income in the past 12 months (in 2015 Inflation-adjusted dollars)'] = pd.to_numeric(
    data['Estimate; Aggregate household income in the past 12 months (in 2015 Inflation-adjusted dollars)'], errors='coerce')
data['Margin of Error; Aggregate household income in the past 12 months (in 2015 Inflation-adjusted dollars)'] = pd.to_numeric(
    data['Margin of Error; Aggregate household income in the past 12 months (in 2015 Inflation-adjusted dollars)'], errors='coerce')

# Drop rows with NaN values
data.dropna(inplace=True)

# Prepare features and target variable
# Assuming you want to predict the estimate of household income
X = data[['Id']]  # Features (you can include more columns if needed)
y = data['Estimate; Aggregate household income in the past 12 months (in 2015 Inflation-adjusted dollars)']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple model (make sure this aligns with your task)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mean_squared_error')

print("Building model...")
# Train the model
print("Starting model training...")
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))  # Validation data added
print("Model training complete!")
