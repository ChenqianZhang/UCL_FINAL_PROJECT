import tensorflow as tf
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
model = tf.keras.models.load_model('DNN-CSI.h5')

# Load the data from .mat file
position_data = scipy.io.loadmat('POSITION.mat')['POSITION']
csi_info_data = scipy.io.loadmat('CSI_INFO.mat')['CSI_INFO']

# Preparing the input and output data
X_data = position_data.T  # Assuming POSITION matrix is already in the desired format
Y_data = csi_info_data.T

# Feature scaling using the scaler from the first 95000 data points
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit the scalers using the first 95000 data points
X_data_scaled = scaler_x.fit_transform(X_data[:95000])
Y_data_scaled = scaler_y.fit_transform(Y_data[:95000])

# Select the last 100 data points for validation
X_val_scaled = scaler_x.transform(X_data[-100:])
Y_val_scaled = scaler_y.transform(Y_data[-100:])

# Predict the output using the model
Y_pred_scaled = model.predict(X_val_scaled)

# Inverse transform the scaled predictions to the original scale
Y_pred = scaler_y.inverse_transform(Y_pred_scaled)

# Plot the actual and predicted values for the first row of output
output_index = 400
plt.figure(figsize=(10, 6))
plt.plot(Y_data[-100:, output_index], label='Actual')
plt.plot(Y_pred[:, output_index], label='Predicted')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.title(f'Actual vs Predicted for Output {output_index + 1}')
plt.legend()
plt.show()
