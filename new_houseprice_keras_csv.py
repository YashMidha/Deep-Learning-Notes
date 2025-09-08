import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('house_data.csv')

# Features and target
X = data[['bedrooms', 'sqft', 'location_score', 'age']]
y = data['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(20, input_dim=4, activation='relu'))  # First hidden layer
model.add(Dense(10, activation='relu'))               # Second hidden layer
model.add(Dense(1))                                    # Output layer

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(X_train, y_train, epochs=10, batch_size=1, validation_split=0.2)

loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae}")

# Plotting Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)

# Plotting MAE (Accuracy for regression)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE', marker='o')
plt.plot(history.history['val_mae'], label='Validation MAE', marker='o')
plt.title('Model MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

import numpy as np

# Sample input (same order as training features)
sample_house = np.array([[3, 1600, 8, 12]])

# Apply the same scaling used during training
sample_scaled = scaler.transform(sample_house)

# Predict using the trained model
predicted_price = model.predict(sample_scaled)

print(f"Predicted House Price: ₹{predicted_price[0][0]:,.2f}")
